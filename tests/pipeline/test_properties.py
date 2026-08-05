from __future__ import annotations

import math
from dataclasses import replace

import numpy
import pytest
import tiktoken
from hypothesis import given
from hypothesis import strategies as st

from atlas.config import ChunkSettings
from atlas.contracts import DocName, Document, QuestionName
from atlas.corpus.chunk import get_chunker
from atlas.corpus.gold import Correct
from evals.ir_metrics import (
    bounded_precision_at_k,
    bounded_recall_at_k,
    graded_ndcg_at_k,
    graded_ndcg_ceiling_at_k,
    ndcg_at_k,
    ndcg_ceiling_at_k,
    precision_at_k,
    precision_ceiling_at_k,
    recall_at_k,
    recall_ceiling_at_k,
    reciprocal_rank_at_k,
    success_at_k,
)
from tests.strategies import build_producer, cut_settings, documents, documents_with_a_short_value

SPLITS_VALUES = ChunkSettings(strategy="fixed", max_tokens=32, overlap_tokens=0)
ranked_lists = st.lists(st.builds(lambda i: f"doc#{i:04d}", st.integers(0, 40)),
                        min_size=2, unique=True)
OVER_A_SET = (recall_at_k, precision_at_k, reciprocal_rank_at_k, ndcg_at_k)


@given(doc=documents(), settings=cut_settings())
def test_chunk_text_is_exactly_the_slice_its_range_names(doc, settings):
    for chunk in get_chunker(settings)(doc):
        # A strip, a whitespace collapse or a unicode fold leaves the range addressing text
        # the chunk no longer holds, so every correct answer resolved from it is wrong.
        assert chunk.text == doc.text[chunk.span.start:chunk.span.end]


@given(doc=documents(), settings=cut_settings())
def test_ranges_cover_the_document_with_no_gaps_beyond_the_overlap(doc, settings):
    # The resolver reads correct answers off character ranges, so a character landing in no
    # chunk reads as a retrieval failure rather than as a hole in the chunker.
    chunks = get_chunker(settings)(doc)
    assert chunks[0].span.start == 0
    assert chunks[-1].span.end == len(doc.text)
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.span.start <= earlier.span.end
        assert earlier.span.end <= later.span.end


@given(doc=documents(), settings=cut_settings())
def test_cutting_the_same_document_twice_gives_the_same_chunks(doc, settings):
    chunker = get_chunker(settings)
    assert chunker(doc) == chunker(doc)


@given(doc=documents(), settings=cut_settings())
def test_no_chunk_exceeds_the_token_limit(doc, settings):
    encoding = tiktoken.get_encoding(settings.encoding)
    for chunk in get_chunker(settings)(doc):
        assert len(encoding.encode(chunk.text)) <= settings.max_tokens


def test_an_overlap_at_or_above_the_token_limit_is_refused_by_the_settings():
    # A fixed size chunker whose overlap reaches its limit never advances, so the coverage
    # property above would hang rather than go red. The generator is bounded the same way.
    with pytest.raises(ValueError, match="overlap"):
        ChunkSettings(strategy="fixed", max_tokens=32, overlap_tokens=32)


@given(written=documents_with_a_short_value())
def test_the_short_value_generator_produces_documents_the_chunker_really_divides(written):
    # If this generator emits documents that come back as one chunk, the property below
    # passes trivially and its failure message accuses the chunker, not the generator.
    assert len(get_chunker(SPLITS_VALUES)(written.document)) > 1


@pytest.mark.parametrize("strategy", ["recursive", "sentence"])
@given(written=documents_with_a_short_value())
def test_a_value_shorter_than_the_limit_is_never_divided(written, strategy):
    # The spans come from the generator that wrote them: no production type carries offsets
    # inside a document, since tau2's gold set names required documents and nothing finer.
    # "fixed" is excluded because it genuinely divides values; see the test below.
    chunks = get_chunker(replace(SPLITS_VALUES, strategy=strategy))(written.document)
    for span in written.spans:
        # Containment, not a count of chunks touching the value: with overlap switched on a
        # value can legitimately sit inside two chunks, and the claim is only that some
        # single chunk holds it whole.
        whole = [f for f in chunks if f.span.start <= span.start and span.end <= f.span.end]
        assert whole, f"the value at {span.start}:{span.end} is held whole by no chunk"


def test_the_fixed_size_chunker_divides_a_value_that_lands_on_a_window_boundary() -> None:
    """A fixed size chunker cuts at a token count without looking at what it is cutting, so
    a value shorter than the limit is divided whenever a window boundary falls inside it.
    That is the trade the other two strategies exist to avoid.

    Constructed rather than left to Hypothesis as a strict xfail case, which flaked: every
    CI job starts with a cold example database, so a run that fails to rediscover the
    counterexample inside its budget reports XPASS(strict) and blocks the merge.
    """
    settings = replace(SPLITS_VALUES, strategy="fixed", max_tokens=8)
    filler = " ".join(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"] * 6)
    value = "SPLITME"

    for position in range(1, len(filler)):
        text = filler[:position] + value + filler[position:]
        chunks = get_chunker(settings)(Document(name=DocName("d"), kind="generated", text=text))
        if len(chunks) < 2:
            continue
        start, end = position, position + len(value)
        if not any(c.span.start <= start and end <= c.span.end for c in chunks):
            return  # the window closed inside the value, which is the claim
    raise AssertionError(
        "no insertion position divided the value, so the fixed size chunker no longer "
        "trades value integrity for a constant chunk size and the other two strategies "
        "have nothing left to be better at"
    )


@pytest.mark.parametrize("producer", ["vector", "keyword", "rerank", "fuse"])
@given(order=st.permutations(range(6)))
def test_ranking_is_the_same_however_chunks_were_added(producer, order, tie_heavy):
    """The tie heavy fixture holds two chunks with identical text, so every producer faces
    an exact tie. Python's sort keeps arrival order, so without a tie break on chunk name
    this relation fails intermittently, which reads as a real discovery."""
    reference = build_producer(producer, tie_heavy.chunks, tie_heavy.matrix)
    shuffled = build_producer(producer,
                              tuple(tie_heavy.chunks[i] for i in order),
                              numpy.stack([tie_heavy.matrix[i] for i in order]))
    expected = reference.search(tie_heavy.query_text, tie_heavy.query_vector, candidates=6)
    actual = shuffled.search(tie_heavy.query_text, tie_heavy.query_vector, candidates=6)
    if producer == "fuse":
        # Each arm resolves the tie by chunk name before rrf_fuse sees a score, so the
        # duplicates land on adjacent ranks but never on an equal fused score. The tie is
        # therefore proved on ranks here rather than on scores.
        duplicates = {tie_heavy.chunks[1].name, tie_heavy.chunks[2].name}
        ranks = sorted(h.rank for h in expected.hits if h.chunk in duplicates)
        assert ranks == [ranks[0], ranks[0] + 1], "tie_heavy did not place the duplicates adjacently"
    else:
        scores = [h.score for h in expected.hits]
        # Without a real tie this property is green for an implementation with no tie break
        # at all, leaving it to fail later on live data where the cause is harder to see.
        assert len(set(scores)) < len(scores), "tie_heavy produced no tied scores"
    assert [h.chunk for h in actual.hits] == [h.chunk for h in expected.hits]
    assert [h.rank for h in actual.hits] == [1, 2, 3, 4, 5, 6]


@given(ranked=ranked_lists, data=st.data())
def test_recall_ignores_ordering_within_the_fetched_set_while_the_ranking_measures_do_not(
        ranked, data) -> None:
    # Guards a ranking measure implemented as if order did not matter, which yields a
    # believable meaningless number. The limit is drawn inside the list rather than set to
    # its length: at the full length a shuffle cannot move anything across the boundary, so
    # the recall half holds for every implementation ever written.
    k = data.draw(st.integers(2, len(ranked)))
    correct = frozenset({ranked[k - 1]})
    shuffled = list(data.draw(st.permutations(ranked[:k]))) + ranked[k:]
    assert recall_at_k(ranked, correct, k) == recall_at_k(shuffled, correct, k) == 1.0
    front = [ranked[k - 1]] + [name for name in ranked if name != ranked[k - 1]]
    assert reciprocal_rank_at_k(front, correct, k) > reciprocal_rank_at_k(ranked, correct, k)
    assert ndcg_at_k(front, correct, k) > ndcg_at_k(ranked, correct, k)


def test_an_empty_correct_set_reads_as_not_a_number_rather_than_a_perfect_score() -> None:
    # Not a number is the half of the rule this module can honour, being handed a ranked list
    # and a set and nothing else. The other half, raising rather than scoring, lives in the
    # resolver, the only place that knows whether the question must have a correct answer.
    for measure in OVER_A_SET:
        assert math.isnan(measure(["a"], frozenset(), 1))


def test_a_limit_larger_than_the_ranked_list_is_evaluated_over_what_is_there() -> None:
    # Precision is the load bearing line. Recall cannot prove this: padding a one item list
    # with names that are not correct leaves recall at one either way, so it passes on the
    # exact implementation this was written to catch.
    assert precision_at_k(["a"], frozenset({"a"}), 4) == 1.0
    assert recall_at_k(["a"], frozenset({"a"}), 99) == 1.0


@given(
    ranked=ranked_lists,
    k=st.integers(1, 40),
    deeper=st.integers(1, 40),
    data=st.data(),
)
def test_precision_stops_moving_with_k_once_the_ranking_runs_out(ranked, k, deeper, data):
    # `precision_at_k` divides by what was fetched, so once k exceeds the ranking's length
    # every deeper k scores the same number while `recall_at_k` keeps responding to k. The
    # two sit in adjacent columns of one published table: on the recorded run precision@10,
    # @20 and @50 are all 0.363918 while recall moves. Deliberate, so it is pinned.
    correct = frozenset(data.draw(st.lists(st.sampled_from(ranked), min_size=1, unique=True)))
    beyond, further = sorted((len(ranked) + k, len(ranked) + deeper))
    assert precision_at_k(ranked, correct, beyond) == precision_at_k(ranked, correct, further)
    # The ceiling has to agree, or `Bounded` compares a value against a maximum derived from
    # a different denominator.
    assert precision_ceiling_at_k(ranked, correct, beyond) == pytest.approx(
        precision_ceiling_at_k(ranked, correct, further)
    )


def test_ranking_quality_reaches_one_when_the_ideal_is_capped() -> None:
    # Without the cap a perfect score is unreachable whenever there are fewer correct chunks
    # than the limit.
    assert ndcg_at_k(["a", "b", "c"], frozenset({"a"}), 3) == 1.0


@given(ranked=ranked_lists, k=st.integers(1, 12), data=st.data())
def test_the_graded_ceiling_is_exact_the_same_way_the_flat_ceilings_are(ranked, k, data):
    # The flat ceiling property covers recall, precision and flat nDCG, never the graded
    # one. A mutant sizing graded_ndcg_ceiling_at_k from k rather than from what the ranking
    # returned survived every other test in this suite.
    correct_chunks = frozenset(data.draw(st.lists(st.sampled_from(ranked), min_size=1, unique=True)))
    # Primary is drawn as a subset of correct, never independently: a primary chunk outside
    # the correct set is not a shape atlas.corpus.gold.resolve can produce.
    primary = frozenset(
        data.draw(st.lists(st.sampled_from(sorted(correct_chunks)), unique=True))
    )
    correct = Correct(question=QuestionName("q"), chunks=correct_chunks, primary=primary)
    if not correct.primary:
        return  # not-a-number without a primary set; there is no ceiling to bound
    value = graded_ndcg_at_k(ranked, correct, k)
    ceiling = graded_ndcg_ceiling_at_k(ranked, correct, k)
    assert value <= ceiling + 1e-9

    # Tightness, not just soundness: a perfect ranking has to reach its own ceiling exactly,
    # at both a list at least k long and one shorter than k.
    ordered_correct = sorted(correct.chunks, key=lambda name: name not in correct.primary)
    perfect = tuple(ordered_correct) + tuple(c for c in ranked if c not in correct.chunks)
    for candidate in (perfect, perfect[: max(1, k // 2)]):
        best = graded_ndcg_at_k(candidate, correct, k)
        cap = graded_ndcg_ceiling_at_k(candidate, correct, k)
        assert abs(best - cap) < 1e-9, (
            f"graded_ndcg_at_k on a perfect ranking of {len(candidate)} chunks at k={k} "
            f"reaches {best:.6f} but its ceiling claims {cap:.6f}; a ceiling nothing can "
            "reach is not a ceiling"
        )


@pytest.mark.parametrize("measure", OVER_A_SET, ids=lambda m: m.__name__)
def test_every_measurement_refuses_a_limit_of_zero_and_a_repeated_name(measure) -> None:
    # Parametrised because the guard is five separate pieces of code. A quietly dropped
    # duplicate hides a search returning the same chunk twice while inflating precision.
    with pytest.raises(ValueError):
        measure(["a"], frozenset({"a"}), 0)
    with pytest.raises(ValueError):
        measure(["a", "a"], frozenset({"a"}), 2)


@given(ranked=ranked_lists, k=st.integers(1, 12), data=st.data())
def test_no_capped_metric_ever_exceeds_the_ceiling_that_bounds_it(ranked, k, data):
    # The property behind evals.ir_metrics.Bounded: a ceiling that can come out below its
    # own value turns a good result into an impossible one. Generated rather than exampled
    # because the interesting cases are the awkward sizes, a correct set larger than k, of
    # exactly k, a single correct chunk, a k past the end of the list.
    correct = frozenset(data.draw(st.lists(st.sampled_from(ranked), min_size=1, unique=True))
                        + [f"unretrieved#{i:04d}" for i in range(data.draw(st.integers(0, 15)))])
    assert recall_at_k(ranked, correct, k) <= recall_ceiling_at_k(ranked, correct, k) + 1e-9
    assert precision_at_k(ranked, correct, k) <= precision_ceiling_at_k(ranked, correct, k) + 1e-9
    # Constructing it is the assertion: Bounded raises when the value is above the ceiling.
    bounded_recall_at_k(ranked, correct, k)

    # Tightness, not just soundness. `value <= ceiling` holds for any ceiling large enough,
    # so a ceiling of 1.0 satisfies it forever; both the recall and the nDCG ceilings shipped
    # too high with the soundness half green. A perfect ranking has to reach it exactly.
    perfect = tuple(sorted(correct))[:k] + tuple(c for c in ranked if c not in correct)
    # The short candidate is the whole point. A list at least k long makes min(k, ...) and
    # min(len(ranked[:k]), ...) the same number, so a ceiling sized from k looks correct;
    # the defect only exists where the retriever returns fewer chunks than k.
    for candidate in (perfect, perfect[: max(1, k // 2)]):
        for metric, ceiling in (
            (recall_at_k, recall_ceiling_at_k),
            (precision_at_k, precision_ceiling_at_k),
            (ndcg_at_k, ndcg_ceiling_at_k),
        ):
            best, cap = metric(candidate, correct, k), ceiling(candidate, correct, k)
            assert abs(best - cap) < 1e-9, (
                f"{metric.__name__} on a perfect ranking of {len(candidate)} chunks at k={k} "
                f"reaches {best:.6f} but its ceiling claims {cap:.6f}; a ceiling nothing "
                "can reach is not a ceiling"
            )
    bounded_precision_at_k(ranked, correct, k)


@given(ranked=ranked_lists, k=st.integers(1, 12), data=st.data())
def test_success_agrees_with_recall_about_whether_anything_was_found(ranked, k, data):
    # success_at_k is the coarse reading of the same fact recall reports finely, so the two
    # disagreeing about whether the top k held anything correct means one of them is wrong.
    correct = frozenset(data.draw(st.lists(st.sampled_from(ranked), min_size=1, unique=True)))
    assert (success_at_k(ranked, correct, k) == 1.0) == (recall_at_k(ranked, correct, k) > 0.0)
