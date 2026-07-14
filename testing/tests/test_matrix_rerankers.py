"""`matrix.rerankers`, hermetic (stage 2): rerankers over stage 1's cached candidates, at swept
depths, still no LLM anywhere (a `CassetteReranker` -- a deterministic REPLAYED score table, the
hermetic CI adapter behind the `Reranker` port -- and the `none` identity axis). The depth axis is
exercised for real: a relevant chunk that a shallow depth truncates away before the reranker ever
sees it cannot be recovered, exactly the "reranker quality can degrade past a depth" property
research 14 names, here proven at the OTHER end (recall CAN'T improve past a depth that excludes the
chunk in the first place).
"""
from __future__ import annotations

from atlas.adapters.cassette_reranker import CassetteReranker
from atlas.ports.knowledge import Chunk

from matrix.cache import MatrixCache
from matrix.cases import MatrixCase
from matrix.chunks import deserialize_chunk
from matrix.embedders import EmbedderCell
from matrix.rerankers import DEPTHS, NONE_RERANKER_ID, RerankerComponent, config_id, run_reranker_stage

_QUERY = "query a"
_CASE_A = MatrixCase("case-a", _QUERY, frozenset({"d1"}))
_CASES = (_CASE_A,)

_D1 = Chunk(chunk_id="d1", doc_id="d1", text="one")
_D2 = Chunk(chunk_id="d2", doc_id="d2", text="two")
_D3 = Chunk(chunk_id="d3", doc_id="d3", text="three")
_D4 = Chunk(chunk_id="d4", doc_id="d4", text="four")

_CORPUS_VERSION = "corpus-test-0.0.1"
_DATASET_VERSION = "dataset-test-0.0.1"


def _embedder_cells() -> dict[str, EmbedderCell]:
    """A hand built stage 1 output: case-a's candidates in fused order [d3, d1, d2, d4] -- the
    relevant chunk (d1) sits at rank 2, recoverable by a reranker ONLY IF the depth it truncates to
    is wide enough to still include it."""
    from matrix.chunks import serialize_chunk
    from quality.retrieval_report import CaseRetrieval, evaluate

    candidates = {"case-a": tuple(serialize_chunk(c) for c in (_D3, _D1, _D2, _D4))}
    report = evaluate([CaseRetrieval("case-a", ("d3", "d1", "d2", "d4"), frozenset({"d1"}))], k=1, seed=1)
    return {
        "emb": EmbedderCell(
            component_id="emb", embedding_model={"id": "emb", "revision": "emb"},
            is_baseline=False, candidates=candidates, report=report,
        )
    }


def _boost_d1_reranker() -> RerankerComponent:
    scores = {_QUERY: {"d1": 10.0, "d3": 1.0, "d2": 0.5, "d4": 0.1}}
    return RerankerComponent("bge-reranker-v2-m3", CassetteReranker(scores))


def _none_reranker() -> RerankerComponent:
    return RerankerComponent(NONE_RERANKER_ID, None)


def test_depths_constant_is_the_named_research_14_sweep():
    assert DEPTHS == (20, 50, 100)


def test_config_id_names_embedder_reranker_and_depth():
    assert config_id("emb", "bge-reranker-v2-m3", 20) == "emb::bge-reranker-v2-m3@20"


def test_shallow_depth_truncates_the_relevant_chunk_away_before_rerank_can_recover_it():
    """depth=1 keeps only [d3] from the fused candidates: d1 (the relevant chunk, fused rank 2) is
    truncated away before the reranker ever sees it, so no reranker score table can recover it."""
    cache = MatrixCache(_tmp_dir())
    cells = run_reranker_stage(
        _CASES, _embedder_cells(), [_boost_d1_reranker()], k_final=1, seed=1, cache=cache,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION, depths=(1,),
    )
    cell = cells[config_id("emb", "bge-reranker-v2-m3", 1)]
    assert cell.report.hit_rate_at_k == 0.0
    assert [c["doc_id"] for c in cell.candidates["case-a"]] == ["d3"]


def test_wider_depth_lets_the_reranker_recover_the_relevant_chunk():
    """depth=2 keeps [d3, d1]: d1 IS in the truncated set now, and the reranker's own score table
    (d1=10.0 > d3=1.0) puts it first, so it survives truncation to k_final=1."""
    cache = MatrixCache(_tmp_dir())
    cells = run_reranker_stage(
        _CASES, _embedder_cells(), [_boost_d1_reranker()], k_final=1, seed=1, cache=cache,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION, depths=(2,),
    )
    cell = cells[config_id("emb", "bge-reranker-v2-m3", 2)]
    assert cell.report.hit_rate_at_k == 1.0
    assert [c["doc_id"] for c in cell.candidates["case-a"]] == ["d1"]


def test_none_axis_is_an_identity_pass_through_never_reorders():
    """The `none` axis at the SAME depth=2 that let the real reranker recover d1 leaves the fused
    order untouched (d3 stays first), proving `none` really is a no-op, not a silently-applied
    default reranker."""
    cache = MatrixCache(_tmp_dir())
    cells = run_reranker_stage(
        _CASES, _embedder_cells(), [_none_reranker()], k_final=1, seed=1, cache=cache,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION, depths=(2,),
    )
    cell = cells[config_id("emb", NONE_RERANKER_ID, 2)]
    assert cell.report.hit_rate_at_k == 0.0
    assert [c["doc_id"] for c in cell.candidates["case-a"]] == ["d3"]


def test_reranker_and_none_axis_both_present_across_multiple_depths():
    cache = MatrixCache(_tmp_dir())
    cells = run_reranker_stage(
        _CASES, _embedder_cells(), [_boost_d1_reranker(), _none_reranker()], k_final=1, seed=1,
        cache=cache, corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION, depths=(1, 2, 4),
    )
    assert len(cells) == 2 * 3  # 2 rerankers x 3 depths, over the ONE embedder cell


def test_candidates_rehydrate_into_real_chunk_objects():
    cache = MatrixCache(_tmp_dir())
    cells = run_reranker_stage(
        _CASES, _embedder_cells(), [_boost_d1_reranker()], k_final=1, seed=1, cache=cache,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION, depths=(2,),
    )
    cell = cells[config_id("emb", "bge-reranker-v2-m3", 2)]
    chunks = [deserialize_chunk(d) for d in cell.candidates["case-a"]]
    assert chunks == [_D1]


def test_content_hash_cache_skips_recompute_on_a_rerun():
    calls = {"n": 0}

    class _CountingCassetteReranker(CassetteReranker):
        def rerank(self, query, chunks):
            calls["n"] += 1
            return super().rerank(query, chunks)

    reranker = RerankerComponent("bge-reranker-v2-m3", _CountingCassetteReranker({_QUERY: {"d1": 10.0}}))
    cache_dir = _tmp_dir()
    cache1 = MatrixCache(cache_dir)
    run_reranker_stage(
        _CASES, _embedder_cells(), [reranker], k_final=1, seed=1, cache=cache1,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION, depths=(2,),
    )
    assert calls["n"] == 1

    cache2 = MatrixCache(cache_dir)  # same directory, fresh instance: simulates a rerun
    run_reranker_stage(
        _CASES, _embedder_cells(), [reranker], k_final=1, seed=1, cache=cache2,
        corpus_version=_CORPUS_VERSION, dataset_version=_DATASET_VERSION, depths=(2,),
    )
    assert calls["n"] == 1  # unchanged: the rerun never called rerank() again
    assert cache2.hits == 1 and cache2.misses == 0


def _tmp_dir():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp(prefix="matrix-rerank-cache-"))
