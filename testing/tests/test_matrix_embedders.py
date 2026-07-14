"""`matrix.embedders`, hermetic (stage 1): retrieval only metrics, no LLM anywhere. A hand workable
fixture (known relevant sets, known rankings) so recall/nDCG are checkable by inspection; a
call-counting `search` wrapper proves the content hash cache skips recompute on a rerun rather than
merely asserting the cache directory has files in it.
"""
from __future__ import annotations

from atlas.ports.knowledge import Chunk

from matrix.cache import MatrixCache
from matrix.cases import MatrixCase
from matrix.embedders import (
    BASELINE_COMPONENT_IDS,
    BM25_COMPONENT_ID,
    EXACT_SCAN_COMPONENT_ID,
    EmbedderComponent,
    candidate_chunks,
    run_embedder_stage,
)

_CASE_A = MatrixCase("case-a", "query a", frozenset({"d1"}))
_CASE_B = MatrixCase("case-b", "query b", frozenset({"d2"}))
_CASES = (_CASE_A, _CASE_B)

_D1 = Chunk(chunk_id="d1", doc_id="d1", text="one")
_D2 = Chunk(chunk_id="d2", doc_id="d2", text="two")
_D3 = Chunk(chunk_id="d3", doc_id="d3", text="three")

_CORPUS_VERSION = "corpus-test-0.0.1"
_DATASET_VERSION = "dataset-test-0.0.1"


class _CountingSearch:
    """A fixture retrieval callable (the "seeded REPLAY fixture" this stage's own module docstring
    names) that counts its own calls, so a hermetic test can assert the cache genuinely skipped
    recompute rather than inferring it from timing."""

    def __init__(self, table: dict[str, list[Chunk]]) -> None:
        self._table = table
        self.calls = 0

    def __call__(self, case: MatrixCase) -> list[Chunk]:
        self.calls += 1
        return self._table[case.case_id]


def _perfect_search() -> _CountingSearch:
    # case-a's relevant d1 is retrieved first; case-b's relevant d2 is retrieved first: both hit at k=1.
    return _CountingSearch({"case-a": [_D1, _D3, _D2], "case-b": [_D2, _D3, _D1]})


def _bad_search() -> _CountingSearch:
    # neither case's relevant doc appears in the top 1 (both land at rank 2 or later): recall@1 = 0.
    return _CountingSearch({"case-a": [_D3, _D1, _D2], "case-b": [_D3, _D1, _D2]})


def test_perfect_embedder_scores_full_recall_and_ndcg_at_k1():
    search = _perfect_search()
    embedder = EmbedderComponent("perfect", search, embedding_model={"id": "perfect", "revision": "perfect"})
    cache = _fresh_cache()
    cells = run_embedder_stage(
        _CASES, [embedder], k=1, seed=1, cache=cache,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION,
    )
    report = cells["perfect"].report
    assert report.n == 2
    assert report.hit_rate_at_k == 1.0
    assert report.ndcg_at_k_ci[0] == 1.0


def test_bad_embedder_scores_zero_recall_at_k1():
    search = _bad_search()
    embedder = EmbedderComponent("bad", search, embedding_model={"id": "bad", "revision": "bad"})
    cache = _fresh_cache()
    cells = run_embedder_stage(
        _CASES, [embedder], k=1, seed=1, cache=cache,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION,
    )
    assert cells["bad"].report.hit_rate_at_k == 0.0


def test_baseline_component_ids_are_bm25_and_exact_scan():
    assert BASELINE_COMPONENT_IDS == {BM25_COMPONENT_ID, EXACT_SCAN_COMPONENT_ID}


def test_bm25_and_exact_scan_baseline_rows_are_present_in_a_stage_run():
    bm25 = EmbedderComponent(BM25_COMPONENT_ID, _perfect_search(), embedding_model=None, is_baseline=True)
    exact = EmbedderComponent(
        EXACT_SCAN_COMPONENT_ID, _perfect_search(),
        embedding_model={"id": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181"},
        is_baseline=True,
    )
    cache = _fresh_cache()
    cells = run_embedder_stage(
        _CASES, [bm25, exact], k=1, seed=1, cache=cache,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION,
    )
    assert set(cells) == BASELINE_COMPONENT_IDS
    assert cells[BM25_COMPONENT_ID].is_baseline is True
    assert cells[BM25_COMPONENT_ID].embedding_model is None  # lexical: no real embedder at all
    assert cells[EXACT_SCAN_COMPONENT_ID].embedding_model == {
        "id": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181",
    }


def test_content_hash_cache_skips_recompute_on_a_rerun():
    search = _perfect_search()
    embedder = EmbedderComponent("perfect", search, embedding_model={"id": "x", "revision": "x"})
    cache_dir = _tmp_cache_dir()
    cache1 = MatrixCache(cache_dir)
    run_embedder_stage(
        _CASES, [embedder], k=1, seed=1, cache=cache1,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION,
    )
    assert search.calls == len(_CASES)  # one call per case, the first (and only) computation

    cache2 = MatrixCache(cache_dir)  # a fresh instance over the SAME directory: simulates a rerun
    cells = run_embedder_stage(
        _CASES, [embedder], k=1, seed=1, cache=cache2,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION,
    )
    assert search.calls == len(_CASES)  # unchanged: the rerun never called search again
    assert cache2.hits == 1 and cache2.misses == 0
    assert cells["perfect"].report.hit_rate_at_k == 1.0  # the cached result still scores correctly


def test_candidate_chunks_rehydrates_chunk_objects_for_a_known_case():
    search = _perfect_search()
    embedder = EmbedderComponent("perfect", search, embedding_model={"id": "x", "revision": "x"})
    cache = _fresh_cache()
    cells = run_embedder_stage(
        _CASES, [embedder], k=3, seed=1, cache=cache,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION,
    )
    chunks = candidate_chunks(cells["perfect"], "case-a")
    assert [c.chunk_id for c in chunks] == ["d1", "d3", "d2"]
    assert all(isinstance(c, Chunk) for c in chunks)


def test_candidate_chunks_on_an_unknown_case_id_is_empty_never_a_keyerror():
    search = _perfect_search()
    embedder = EmbedderComponent("perfect", search, embedding_model={"id": "x", "revision": "x"})
    cache = _fresh_cache()
    cells = run_embedder_stage(
        _CASES, [embedder], k=1, seed=1, cache=cache,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION,
    )
    assert candidate_chunks(cells["perfect"], "no-such-case") == []


# ---- tiny helpers -------------------------------------------------------------------------------


def _tmp_cache_dir():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp(prefix="matrix-cache-"))


def _fresh_cache():
    return MatrixCache(_tmp_cache_dir())
