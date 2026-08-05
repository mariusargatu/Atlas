from __future__ import annotations

import inspect
import math
from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from atlas.config import RerankSettings, RrfSettings
from atlas.contracts import ChunkId
from atlas.retrieval.fuse import rrf_fuse
from atlas.retrieval.rerank import get_reranker
from scripts.repo_checks import collected_test_functions
from tests.strategies import StubReranker, chunk_lists, ranked_lists, worse_chunk_lists

SETTINGS = RrfSettings(constant=60, vector_weight=1.0, keyword_weight=1.0)
RERANK = RerankSettings(backend="passthrough", depth_in=8, depth_out=3)


def build_reranker(backend: str, depth_in: int):
    # StubReranker is constructed directly rather than added to RerankSettings.backend,
    # which is public surface: a fourth value there is a breaking change that bumps the
    # schema version.
    if backend == "stub":
        return StubReranker(depth_in=depth_in, depth_out=RERANK.depth_out)
    return get_reranker(replace(RERANK, backend=backend, depth_in=depth_in))


def test_ranks_start_at_one(one_vector_hit, empty_keyword):
    result = rrf_fuse(one_vector_hit, empty_keyword, SETTINGS)
    # Zero-based ranks are fine arithmetically and make every published number
    # incomparable with the literature.
    assert result.hits[0].rank == 1 and result.hits[0].score == pytest.approx(1 / 61)


def test_a_chunk_missing_from_one_search_contributes_nothing_from_it(one_vector_hit, empty_keyword):
    result = rrf_fuse(one_vector_hit, empty_keyword, SETTINGS)
    # Not treated as sitting one place past the end of the keyword list; that choice is
    # what makes the scale invariance property below hold.
    assert result.arm_ranks[result.hits[0].chunk] == {"vector": 1}
    assert result.hits[0].score == pytest.approx(1 / 61)


def test_equal_scores_are_ordered_by_chunk_name(tied_searches):
    result = rrf_fuse(*tied_searches, SETTINGS)
    scores = [h.score for h in result.hits]
    # Without this precondition the assertion below passes for an implementation with no
    # tie break at all.
    assert len(set(scores)) == 1, "tied_searches produced no tie to break"
    names = [h.chunk for h in result.hits]
    assert names == sorted(names) and len(names) == len(set(names))


def test_the_arm_weights_are_read_rather_than_assumed(second_ranked_in_keyword_only):
    # Both weights default to one, so every test above passes for an implementation
    # that ignores them.
    vector, keyword = second_ranked_in_keyword_only
    even = rrf_fuse(vector, keyword, SETTINGS)
    keyword_heavy = rrf_fuse(vector, keyword, replace(SETTINGS, keyword_weight=10.0))
    assert [h.chunk for h in even.hits] != [h.chunk for h in keyword_heavy.hits]


@given(vector=ranked_lists(), keyword=ranked_lists())
def test_the_ranked_list_generator_produces_lists_the_contract_allows(vector, keyword):
    # A malformed generated list makes blending raise, and the failure then accuses
    # blending rather than the generator. Assert the precondition where it names the culprit.
    for result in (vector, keyword):
        names = [h.chunk for h in result.hits]
        assert len(names) == len(set(names))
        assert [h.rank for h in result.hits] == list(range(1, len(names) + 1))
        assert all(math.isfinite(h.score) for h in result.hits)
        assert list(result.hits) == sorted(result.hits, key=lambda h: (-h.score, h.chunk))


@given(vector=ranked_lists(), keyword=ranked_lists(), scale=st.floats(1e-3, 1e3))
def test_blending_is_unaffected_by_multiplying_either_search_scores(vector, keyword, scale):
    """Blending is a pure function of ranks. Keyword scores and cosine similarities live
    on incompatible scales, so anything reading a raw score here is tuning noise."""
    scaled = replace(vector, hits=tuple(replace(h, score=h.score * scale) for h in vector.hits))
    assert ([h.chunk for h in rrf_fuse(vector, keyword, SETTINGS).hits]
            == [h.chunk for h in rrf_fuse(scaled, keyword, SETTINGS).hits])


@given(result=ranked_lists())
def test_blending_is_monotonic_in_rank_and_identical_searches_reproduce_the_input_order(result):
    # The cheapest statement of monotonicity there is: a failure says the sum is reading
    # something other than rank.
    fused = rrf_fuse(replace(result, arm="vector"), replace(result, arm="keyword"), SETTINGS)
    assert [h.chunk for h in fused.hits] == [h.chunk for h in result.hits]
    assert [h.rank for h in fused.hits] == list(range(1, len(result.hits) + 1))


@pytest.mark.parametrize("backend", ["passthrough", "stub"])
@pytest.mark.parametrize("depth_in", [4, 40])
@given(candidates=chunk_lists(min_size=5, max_size=20))
def test_reranking_output_is_only_a_reordering_of_a_prefix_of_its_input(backend, depth_in, candidates):
    # Both depths are chosen against the generator's bounds: at four every generated list
    # is longer than the depth and at forty none is, so both regimes are exercised.
    assert len({f.name for f in candidates}) == len(candidates), "chunk_lists repeated a name"
    reranker = build_reranker(backend, depth_in)
    result = reranker("what does the plan cost", candidates)
    output = [h.chunk for h in result.hits]
    assert result.input_order == tuple(f.name for f in candidates[:depth_in])
    assert set(output) <= set(result.input_order)
    assert len(output) == min(RERANK.depth_out, len(result.input_order))
    assert len(set(output)) == len(output)
    assert [h.rank for h in result.hits] == list(range(1, len(output) + 1))
    # Nothing above reads a score. That is the rule, not an omission.


@given(candidates=chunk_lists(min_size=2, max_size=8), padding=worse_chunk_lists(max_size=8))
def test_appending_strictly_worse_candidates_does_not_change_the_top_result(candidates, padding):
    # Strictly worse is only definable when the test controls the scorer, hence the stub.
    # The depth is above the padded length so a green result cannot come from the padding
    # being truncated away before the reranker saw it.
    reranker = StubReranker(depth_in=40, depth_out=RERANK.depth_out)
    plain = reranker("what does the plan cost", candidates)
    padded = reranker("what does the plan cost", tuple(candidates) + tuple(padding))
    assert padded.hits[0].chunk == plain.hits[0].chunk


def test_no_reranking_test_reads_a_score():
    # Every collected test whose source reaches a reranker must say so in its name and
    # must not touch a score attribute. Per function rather than per file, because
    # blending tests read scores legitimately and live in the same module. This function
    # names both forbidden strings to check for them, so it excuses itself from its scan.
    this_test = f"{__name__.rsplit('.', 1)[-1]}.test_no_reranking_test_reads_a_score"
    for name, function in collected_test_functions():
        if name == this_test:
            continue
        source = inspect.getsource(function)
        # Any route to a reranker, not one literal: matching only "get_reranker" leaves
        # tests that go through build_reranker invisible to the scan.
        if not any(marker in source for marker in ("get_reranker", "build_reranker",
                                                   "OnnxReranker", "PassthroughReranker")):
            continue
        assert "rerank" in name, f"{name} reaches the reranker without saying so in its name"
        assert ".score" not in source, f"{name} asserts on a reranker score"


def test_the_onnx_reranker_reorders_a_prefix_of_a_fixed_list(deep_and_shallow):
    """One deterministic case against the real backend.

    Downloads ~23 MB from huggingface.co on a cold cache. That network access is a genuine
    prerequisite: HF_HUB_OFFLINE is read at import time, so no in-test monkeypatch avoids it.
    """
    settings = replace(RERANK, backend="onnx")
    result = get_reranker(settings)("what does the plan cost", deep_and_shallow)
    assert result.input_order == tuple(f.name for f in deep_and_shallow[: settings.depth_in])
    assert {h.chunk for h in result.hits} <= set(result.input_order)
    assert [h.rank for h in result.hits] == list(range(1, len(result.hits) + 1))
    # Everything above holds for a constant scorer: `ranked_hits` sorts on (-score, name),
    # so a model scoring every passage 0.0 degrades to alphabetical, which preserves the
    # input order, is a subset of it, and numbers 1..n. deep.0007 is the passage about plan
    # cost, so this is the one assertion that fails if the model stops working.
    assert result.hits[0].chunk == ChunkId("deep.0007"), (
        "the real cross encoder did not rank the one passage about plan cost first, which "
        "either means the model changed or that it is not being consulted at all"
    )
