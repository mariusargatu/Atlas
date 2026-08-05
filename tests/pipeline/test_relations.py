"""Metamorphic relations: what must stay true between two runs.

There is no oracle for "is this answer good", so these checks assert relations between
runs rather than values within one.
"""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import pytest

from atlas.config import (
    ChunkSettings,
    EmbeddingSettings,
    RerankSettings,
    RrfSettings,
    Settings,
    SparseSettings,
)
from atlas.contracts import Chunk, DocName
from atlas.corpus.chunk import get_chunker
from atlas.corpus.gold import resolve
from atlas.corpus.tau2 import Tau2Source
from atlas.models.embed import CachedEmbedder, get_embedder
from atlas.models.generate import ModelAnswerWriter
from atlas.retrieval.dense import VectorIndex
from atlas.retrieval.fuse import rrf_fuse
from atlas.retrieval.rerank import get_reranker
from atlas.retrieval.sparse import build_keyword_index

_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = Tau2Source()
_COLLECTION = _SOURCE(0)
_QUESTIONS = _SOURCE.questions()
_FRAGMENTS = tuple(f for d in _COLLECTION.documents for f in get_chunker(ChunkSettings())(d))
# Module scoped and memoising. The relations assert that two runs of the same function
# agree, which needs one embedder object: a fresh client per relation would be asserting
# the endpoint's determinism rather than the pipeline's.
_VECTORISER = CachedEmbedder(get_embedder(EmbeddingSettings()))
_MATRIX = _VECTORISER.encode([f.text for f in _FRAGMENTS])

# Questions whose gold set is small enough that "the top result" is meaningful. A task
# requiring thirty documents has no single correct answer to hold steady.
_FOCUSED = tuple(q for q in _QUESTIONS if len(q.required) <= 3)


def _ranked(chunks: tuple[Chunk, ...], matrix, question_text: str) -> tuple[str, ...]:
    vector_index = VectorIndex.build(chunks, matrix)
    keyword_index = build_keyword_index(chunks, SparseSettings())
    query_vector = _VECTORISER.encode([question_text])[0]
    vector = vector_index.search(question_text, query_vector, 50)
    keyword = keyword_index.search(question_text, 50)
    return tuple(h.chunk for h in rrf_fuse(vector, keyword, RrfSettings()).hits)


def test_the_focused_question_set_is_not_empty() -> None:
    # If every gold set were large this module would pass while exercising nothing.
    # Checked here, where the failure names the cause rather than looking like a ranking bug.
    assert _FOCUSED, "no task has a small enough gold set for a top-result relation"


def test_every_question_resolves_to_at_least_one_chunk() -> None:
    # Recall over an empty correct set is undefined and the obvious implementation returns
    # one, so a required document that reached no chunk would read as a perfect score.
    for question in _QUESTIONS:
        assert resolve(question, _FRAGMENTS).chunks, (
            f"{question.name} resolves to nothing")


def _dense_ranking(chunks: tuple[Chunk, ...], matrix, question_text: str) -> tuple[str, ...]:
    query_vector = _VECTORISER.encode([question_text])[0]
    return tuple(h.chunk for h in
                 VectorIndex.build(chunks, matrix).search(question_text, query_vector, 50).hits)


def test_adding_documents_the_question_does_not_need_never_promotes_a_chunk_in_the_dense_arm() -> None:
    """A dense score is a cosine against the chunk's own vector, so adding documents adds
    competitors and can only push a chunk down. A dense arm that broke this would be
    reading the collection rather than the query.

    The keyword arm is deliberately excluded: BM25 weights terms by inverse document
    frequency, which does read the whole collection, measured at 6 violations in 239
    comparisons against 0 in 195 for the dense arm. See docs/the-keyword-search.md.

    Asserted per arm, never on the fused ranking. Each arm contributes only its top fifty,
    so a chunk that falls out of one arm's cut loses that arm's whole contribution and can
    be overtaken by one that never had it; a fused version of this relation is false.
    """
    chunker = get_chunker(ChunkSettings())
    checked = 0
    for question in _FOCUSED:
        required = set(question.required)
        others = tuple(d for d in _COLLECTION.documents if d.name not in required)
        keep = tuple(d for d in _COLLECTION.documents if d.name in required)
        # A subset of the whole collection, so the only difference is the addition.
        small = tuple(f for d in keep + others[:100] for f in chunker(d))
        assert len(small) < len(_FRAGMENTS), "the smaller corpus is not smaller"

        before = _dense_ranking(small, _VECTORISER.encode([f.text for f in small]), question.text)
        after = _dense_ranking(_FRAGMENTS, _MATRIX, question.text)
        position = {name: i for i, name in enumerate(after)}
        for i, name in enumerate(before):
            if name not in position:
                continue
            checked += 1
            assert position[name] >= i, (
                f"{question.name}: the dense arm promoted {name} from {i} to "
                f"{position[name]} when unrelated documents arrived, so its score "
                "reads the corpus rather than the query"
            )
    assert checked > 100, f"only {checked} rank comparisons survived, so this proves little"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_insertion_order_gives_the_same_ranked_names(seed: int) -> None:
    # Shuffling the input and re-encoding is the only way to see a producer that leaks its
    # insertion order into its tie breaks.
    question = _FOCUSED[0]
    shuffled = list(_FRAGMENTS)
    random.Random(seed).shuffle(shuffled)
    matrix = _VECTORISER.encode([f.text for f in shuffled])
    assert _ranked(_FRAGMENTS, _MATRIX, question.text) == _ranked(
        tuple(shuffled), matrix, question.text)


def test_every_chunk_retrieves_itself_from_its_own_vector() -> None:
    # The join between a chunk name and its row of the matrix has no type and passes every
    # shape check there is. It also partially conceals its own breakage, so no metric can
    # find it: rolling the rows by one costs fused nDCG@10 only 0.4218 to 0.3779, above the
    # 0.2563 a ranking that never reads the question scores.
    #
    # The order invariance relation above cannot catch a corruption applied equally to both
    # branches, a matrix rolled inside `search` say, because then both sides are wrong the
    # same way and still agree. This asserts the pairing rather than that two pairings match.
    index = VectorIndex.build(_FRAGMENTS, _MATRIX)
    by_name = {f.name: f.text for f in _FRAGMENTS}
    for fragment, vector in zip(_FRAGMENTS, _MATRIX, strict=True):
        top = index.search(fragment.text, vector, 1).hits[0]
        # Compared by text rather than by name: two chunks carrying identical text have
        # identical vectors, tie at cosine 1.0, and break the tie by name.
        assert by_name[top.chunk] == fragment.text, (
            f"{fragment.name} did not retrieve itself: its own vector ranked "
            f"{top.chunk} first, so the matrix row and the chunk name disagree"
        )


def test_the_composed_pipeline_including_the_reranker_is_order_invariant() -> None:
    # The reranker sees a candidate list rather than the corpus, so it needs its own case:
    # a reranker that broke ties by arrival order satisfies every check above this one.
    question = _FOCUSED[0]
    # The backend is named rather than taken from the default, which is passthrough. A
    # reranker returning its input order is order dependent by definition, so passthrough
    # cannot show this relation. See docs/the-reranker.md.
    settings = replace(RerankSettings(), backend="onnx")
    reranker = get_reranker(settings)
    by_name = {f.name: f for f in _FRAGMENTS}
    ranked = _ranked(_FRAGMENTS, _MATRIX, question.text)
    # Cut to depth_in before reversing: the reranker keeps only the first depth_in
    # candidates, so reversing a longer list feeds it a different set, not the same set
    # in a different order.
    candidates = [by_name[name] for name in ranked][: settings.depth_in]
    assert len(candidates) == settings.depth_in, "too few candidates to fill the depth"

    forward = tuple(h.chunk for h in reranker(question.text, candidates).hits)
    backward = tuple(h.chunk for h in reranker(question.text, candidates[::-1]).hits)
    assert forward == backward


def test_withholding_every_required_document_leaves_nothing_correct_to_retrieve() -> None:
    # Whether a writer refuses when its evidence is withheld is a question about a model.
    # The half checkable without generating anything is this one: with the required
    # documents gone, the gold set must report that nothing correct was retrievable rather
    # than quietly scoring the remaining passages as if they were.
    question = _FOCUSED[0]
    required = set(question.required)
    withheld = tuple(f for f in _FRAGMENTS if DocName(f.document) not in required)
    assert len(withheld) < len(_FRAGMENTS), "no chunk belonged to a required document"

    ranked = _ranked(withheld, _VECTORISER.encode([f.text for f in withheld]), question.text)
    correct = resolve(question, _FRAGMENTS).chunks
    assert not (set(ranked) & correct), "a correct chunk survived being withheld"


def test_the_answer_writer_records_no_violations_on_a_clean_run(fake_model) -> None:
    # Under test is the writer's bookkeeping, not a model's judgement, so the client is a
    # fake: a real model would make the assertion depend on what it happened to cite.
    question = _FOCUSED[0]
    by_name = {f.name: f for f in _FRAGMENTS}
    shown = [by_name[n] for n in _ranked(_FRAGMENTS, _MATRIX, question.text)][
        : Settings().answer.max_shown]
    client = fake_model(
        text="See the first passage.", cited=[shown[0].name], outcome="answered", reason="",
    )
    answer = ModelAnswerWriter(Settings().answer, client)(question, shown)
    assert answer.violations == ()
    assert set(answer.cited) <= set(answer.shown)


def test_the_vendored_corpus_is_present_and_unmodified_in_shape() -> None:
    # The corpus is vendored and nothing here can rebuild it, so a partial checkout or a
    # stray edit must fail loudly rather than move every number in the repository quietly.
    assert len(_COLLECTION.documents) == 698, "tau2 ships 698 knowledge documents"
    assert len(_QUESTIONS) == 97, "tau2 ships 97 tasks carrying a gold document list"
    assert all(d.kind == "tau2_knowledge" for d in _COLLECTION.documents)
    assert (_ROOT / "data/tau2/banking_knowledge/documents").is_dir()
