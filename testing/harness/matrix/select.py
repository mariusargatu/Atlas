"""Selecting the top 1 to 2 retrieval configs stage 3 spends generator calls on (D17: staged, never a
cross product). Stage 1 and stage 2 together produce every candidate retrieval config (each embedder
alone, plus every embedder x reranker x depth triple); this module never runs anything again, it only
ranks the configs those two stages already computed and hands the top few, by name, to stage 3.
"""
from __future__ import annotations

from dataclasses import dataclass

from quality.retrieval_report import RetrievalReport

from matrix.embedders import EmbedderCell
from matrix.rerankers import RerankerCell


@dataclass(frozen=True)
class RetrievalConfigResult:
    """One retrieval config's identity (`config_id`, an embedder component id alone or
    `matrix.rerankers.config_id(...)`'s embedder::reranker@depth form), its per case candidate
    chunks (what stage 3 actually generates from), and the nDCG point estimate `select_top_configs`
    ranks by."""

    config_id: str
    candidates: dict[str, tuple[dict, ...]]
    ndcg_point: float


def _ndcg_point(report: RetrievalReport) -> float:
    return report.ndcg_at_k_ci[0]


def _truncated(candidates: dict[str, tuple[dict, ...]], k_final: int) -> dict[str, tuple[dict, ...]]:
    return {case_id: chunks[:k_final] for case_id, chunks in candidates.items()}


def all_retrieval_configs(
    embedder_cells: dict[str, EmbedderCell],
    reranker_cells: dict[str, RerankerCell],
    *,
    k_final: int,
) -> list[RetrievalConfigResult]:
    """Every stage 1 embedder cell (unreranked) PLUS every stage 2 reranker cell, as one flat list of
    candidate configs. Order follows the caller's own dict insertion order (stage 1's own embedders
    sequence, then stage 2's own embedder x reranker x depth walk); `select_top_configs`'s sort is
    what makes the final ranking independent of order, not this function.

    A bare embedder cell's own `candidates` is intentionally the WIDE pool stage 2 needs headroom to
    sweep depths over (`matrix.embedders.run_embedder_stage`'s `pool_size`); handed to stage 3
    UNTRUNCATED, a "no rerank" config would generate over every pooled candidate instead of the top
    `k_final`, exactly the width a real reranker's absence should NOT grant. `k_final` truncates a
    bare embedder cell's candidates here, matching what `run_reranker_stage`'s own `none` axis
    already does explicitly; a reranker cell's own candidates are already `k_final` truncated by that
    stage, so truncating here again changes nothing for them."""
    configs = [
        RetrievalConfigResult(component_id, _truncated(cell.candidates, k_final), _ndcg_point(cell.report))
        for component_id, cell in embedder_cells.items()
    ]
    configs.extend(
        RetrievalConfigResult(cid, cell.candidates, _ndcg_point(cell.report)) for cid, cell in reranker_cells.items()
    )
    return configs


def select_top_configs(configs: list[RetrievalConfigResult], *, n: int = 2) -> list[RetrievalConfigResult]:
    """Best nDCG point estimate first, ties broken by `config_id` ascending (never dict/insertion
    order), capped at `n` (the plan's own "top 1 to 2 retrieval configs" wording). `n=0` or an empty
    `configs` both return `[]`, the honest empty case."""
    ordered = sorted(configs, key=lambda c: (-c.ndcg_point, c.config_id))
    return ordered[:n]


__all__ = ["RetrievalConfigResult", "all_retrieval_configs", "select_top_configs"]
