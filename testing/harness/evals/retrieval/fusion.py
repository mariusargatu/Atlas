"""Reciprocal Rank Fusion (RRF): the deterministic hybrid that fuses two ranked lists into one.

Used for the BM25 + dense comparison (doc 07's advanced-retrieval layer) and any with/without
fusion experiment. Score-scale agnostic, which is the point: it fuses by RANK, not by each backend's
incommensurable similarity score, so a lexical list and a dense list combine without calibration.
score(d) = sum over lists of 1 / (k + rank_in_list(d)), rank 1-indexed, k=60 (Cormack et al. 2009).

Pure and deterministic: a doc's fused score depends only on its ranks, and ties are broken by the
best (lowest) rank the doc achieved in any list, then by id, so the output never depends on dict
iteration order. The honest claim it supports is modest, RRF should not underperform its inputs, not
that it delivers a large lift (that needs field-boost tuning); the gate asserts the former.
"""
from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60  # the canonical constant; damps the weight of any single list's top ranks


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[str]], k: int = RRF_K) -> list[str]:
    """Fuse ranked id lists into one ranked id list, best-first. Deterministic and order-stable."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            best_rank[doc_id] = min(best_rank.get(doc_id, rank), rank)
    # score descending, then best rank ascending, then id ascending: a total, reproducible order.
    return sorted(scores, key=lambda d: (-scores[d], best_rank[d], d))
