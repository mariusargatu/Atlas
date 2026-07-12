"""Information retrieval metrics: the judge-free half of RAG evaluation (doc 07, "retrieval first").

Given a ranked list of retrieved chunk ids and the set (or graded map) of relevant ids, these are
closed-form arithmetic over positions, no LLM in the loop. That determinism is why they gate the
hermetic PR lane while the framework "contextual precision/recall" metrics, which decide relevance
with a model call, stay in the operator eval lane.

Pure stdlib by design, the same discipline as ``evals.stats``: the interval math and the ranking
math run under the same ``task test``, no numpy, no network. NDCG follows the trec_eval / BEIR
convention (the field's primary metric) exactly: LINEAR gain and a ``log2(rank + 1)`` discount with
rank 1-indexed, so a graded-relevance score is comparable to a published benchmark rather than to a
subtly different exponential-gain implementation. Relevance is matched by exact id membership, which
is *exactly* deterministic (not approximately, as an embedding or string-similarity match would be),
the primitive the fixed-id corpus makes available.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = [
    "average_precision_at_k",
    "dcg_at_k",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _relevance_map(relevant: Mapping[str, float] | frozenset[str] | set[str]) -> Mapping[str, float]:
    """Accept either a set of relevant ids (binary gain 1) or a graded id->gain map, uniformly."""
    if isinstance(relevant, Mapping):
        return relevant
    return {doc_id: 1.0 for doc_id in relevant}


def precision_at_k(retrieved: Sequence[str], relevant: frozenset[str] | set[str], k: int) -> float:
    """Of the top-k retrieved chunks, the fraction that are relevant. Divides by k (trec_eval P@k),
    so a short list that pads below k is penalised rather than flattered. Order-unaware within k."""
    _validate_k(k)
    top = retrieved[:k]
    return sum(1 for doc_id in top if doc_id in relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: frozenset[str] | set[str], k: int) -> float:
    """Of the relevant chunks, the fraction that appear in the top-k. Guards an empty relevant set
    (returns 0.0, never a ZeroDivision): the golden slice always labels at least one, but a caller
    that does not should get a defined value, not a crash."""
    _validate_k(k)
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return sum(1 for doc_id in relevant if doc_id in top) / len(relevant)


def hit_rate_at_k(retrieved: Sequence[str], relevant: frozenset[str] | set[str], k: int) -> float:
    """1.0 if any relevant chunk appears anywhere in the top-k, else 0.0. Per-query {0, 1}; average
    over a query set for the corpus-level hit rate. The coarsest safety net: did the needle appear."""
    _validate_k(k)
    return 1.0 if any(doc_id in relevant for doc_id in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: frozenset[str] | set[str]) -> float:
    """1 / rank of the first relevant chunk (1-indexed), or 0.0 if none is retrieved. Rewards putting
    the one needed chunk high, where the model's attention still is."""
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def average_precision_at_k(retrieved: Sequence[str], relevant: frozenset[str] | set[str], k: int) -> float:
    """Precision averaged over the ranks where relevant chunks appear, within the top-k, normalised by
    the number of relevant chunks. Order-aware over binary relevance; the judge-free twin of a
    reranker's contextual precision. Averaging by |relevant| (not by hits found) charges for misses."""
    _validate_k(k)
    if not relevant:
        return 0.0
    hits = 0
    summed = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            hits += 1
            summed += hits / i
    return summed / len(relevant)


def dcg_at_k(retrieved: Sequence[str], relevance: Mapping[str, float] | frozenset[str] | set[str], k: int) -> float:
    """Discounted cumulative gain over the top-k: sum of gain / log2(rank + 1), rank 1-indexed.
    Linear gain (the relevance value itself), the trec_eval convention BEIR is scored with."""
    _validate_k(k)
    gains = _relevance_map(relevance)
    return sum(gains.get(doc_id, 0.0) / math.log2(i + 1) for i, doc_id in enumerate(retrieved[:k], start=1))


def ndcg_at_k(retrieved: Sequence[str], relevance: Mapping[str, float] | frozenset[str] | set[str], k: int) -> float:
    """DCG@k over the ideal DCG@k (the same gains sorted best-first). 0.0 when no relevance exists to
    normalise against. Handles binary relevance (a set) and graded relevance (an id->gain map); the
    two coincide on binary labels, since 2^1 - 1 == 1 makes exponential and linear gain identical."""
    _validate_k(k)
    gains = _relevance_map(relevance)
    ideal_gains = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_gains, start=1))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, gains, k) / idcg
