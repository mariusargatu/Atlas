"""`matrix.select`, hermetic: ranking (never re-running) the retrieval configs stage 1/2 already
computed, and handing the top 1 to 2 to stage 3 -- the mechanism that keeps the staged design from
ever becoming a cross product on its most expensive axis.
"""
from __future__ import annotations

from quality.retrieval_report import CaseRetrieval, evaluate

from matrix.embedders import EmbedderCell
from matrix.rerankers import RerankerCell
from matrix.select import RetrievalConfigResult, all_retrieval_configs, select_top_configs


def _report(ndcg: float):
    """A report whose `ndcg_at_k_ci` point is exactly `ndcg`: one perfect case plus enough
    zero-score padding cases to hit any target mean, built from real `evaluate()` output (never a
    hand-faked `RetrievalReport`, so the fixture stays honest about what the function actually
    returns)."""
    if ndcg <= 0.0:
        cases = [CaseRetrieval("only", (), frozenset({"x"}))]
    elif ndcg >= 1.0:
        cases = [CaseRetrieval("only", ("x",), frozenset({"x"}))]
    else:
        # ndcg = hits / total at k=1 (binary relevance, so nDCG@1 reduces to hit rate): pick a
        # denominator that reproduces the target ratio exactly.
        hits = round(ndcg * 4)
        cases = [CaseRetrieval(f"h{i}", ("x",), frozenset({"x"})) for i in range(hits)]
        cases += [CaseRetrieval(f"m{i}", (), frozenset({"x"})) for i in range(4 - hits)]
    return evaluate(cases, k=1, seed=1)


def _embedder_cell(component_id: str, ndcg: float) -> EmbedderCell:
    return EmbedderCell(component_id, {"id": "x", "revision": "x"}, False, {}, _report(ndcg))


def _reranker_cell(cid: str, ndcg: float) -> RerankerCell:
    return RerankerCell("emb", "rr", 20, {}, _report(ndcg))


def test_all_retrieval_configs_includes_both_embedder_and_reranker_cells():
    embedder_cells = {"bge-m3": _embedder_cell("bge-m3", 0.5)}
    reranker_cells = {"bge-m3::bge-reranker@20": _reranker_cell("bge-m3::bge-reranker@20", 0.75)}
    configs = all_retrieval_configs(embedder_cells, reranker_cells, k_final=5)
    assert {c.config_id for c in configs} == {"bge-m3", "bge-m3::bge-reranker@20"}


def test_all_retrieval_configs_truncates_a_bare_embedder_cells_wide_pool_to_k_final():
    """A bare embedder cell's own `candidates` is the WIDE pool stage 2 needs (headroom for the
    depth sweep); handed to stage 3 as a "no rerank" config, it must be truncated to k_final, never
    handed over untruncated (that would silently grant a "no rerank" config a wider context window
    than a real no-rerank retrieval path would ever produce)."""
    from atlas.ports.knowledge import Chunk

    from matrix.chunks import serialize_chunk

    wide = {"case-a": tuple(serialize_chunk(Chunk(chunk_id=f"c{i}", doc_id=f"c{i}")) for i in range(10))}
    cell = EmbedderCell("bge-m3", {"id": "x", "revision": "x"}, False, wide, _report(0.5))
    configs = all_retrieval_configs({"bge-m3": cell}, {}, k_final=3)
    assert len(configs[0].candidates["case-a"]) == 3


def test_select_top_configs_ranks_by_ndcg_point_descending():
    configs = [
        RetrievalConfigResult("low", {}, 0.25),
        RetrievalConfigResult("high", {}, 0.9),
        RetrievalConfigResult("mid", {}, 0.5),
    ]
    top = select_top_configs(configs, n=2)
    assert [c.config_id for c in top] == ["high", "mid"]


def test_select_top_configs_breaks_ties_by_config_id_ascending():
    configs = [RetrievalConfigResult("zebra", {}, 0.5), RetrievalConfigResult("alpha", {}, 0.5)]
    top = select_top_configs(configs, n=2)
    assert [c.config_id for c in top] == ["alpha", "zebra"]


def test_select_top_configs_caps_at_n():
    configs = [RetrievalConfigResult(f"c{i}", {}, float(i)) for i in range(5)]
    assert len(select_top_configs(configs, n=1)) == 1
    assert len(select_top_configs(configs, n=2)) == 2


def test_select_top_configs_on_empty_list_is_empty():
    assert select_top_configs([], n=2) == []
