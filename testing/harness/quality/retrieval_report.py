"""Dataset level retrieval measurement (SP7 Task 7): aggregate per case IR metrics into one honest
report, CI widths included, over the seed set's own real sample size, never an invented one.

Pure aggregation, no network and no retriever call anywhere in this module: a caller (a hermetic
fixture, or a live test against the real pgvector/TEI stack) supplies each case's already computed
`(retrieved, relevant)` pair as a `CaseRetrieval`, and `evaluate` turns that sequence into a
`RetrievalReport` using only `quality.ir_metrics` (the per case closed form metrics) and
`quality.stats` (the interval and power sizing math). Fully deterministic and unit tested against a
hand computed fixture; the live half of SP7 Task 7 imports this module rather than duplicating its
arithmetic (`testing/tests/test_sp7_retrieval_metrics_live.py`).

Interval shape mirrors `quality.stats`'s own convention rather than inventing a third one: a binary
per case outcome (`hit_rate_at_k`, did any relevant chunk appear in the top k at all) gets a Wilson
interval, `(lo, hi)`, the well behaved choice at the edges; a continuous per case score (`recall_at_k`,
`ndcg_at_k`, the reciprocal rank feeding `mrr`) gets a percentile bootstrap, `(point, lo, hi)`, since
none of those three have a clean closed form standard error. `detectable_effect_ndcg` answers the
honesty question directly: at this dataset's actual n, what nDCG delta could a future regression
check even see, so a report never claims power the case count does not have.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import stdev

from quality import ir_metrics, stats

DEFAULT_N_RESAMPLES = 2000


@dataclass(frozen=True)
class CaseRetrieval:
    """One case's already computed retrieval outcome: the ranked chunk ids a retriever returned for
    the case's own query (real or replayed), and the relevant set its `expected_doc_ids` declares.
    Holds no query text and makes no call itself; the caller decides where `retrieved` came from."""

    case_id: str
    retrieved: tuple[str, ...]
    relevant: frozenset[str]


@dataclass(frozen=True)
class RetrievalReport:
    """A dataset level retrieval measurement, honest about its own sample size (`n`). Every interval
    field is a `quality.stats` interval, never a bare point: `hit_rate_at_k_ci` is a Wilson `(lo, hi)`
    pair, `recall_at_k_ci`/`mrr_ci`/`ndcg_at_k_ci` are bootstrap `(point, lo, hi)` triples.
    `detectable_effect_ndcg` is `None` only when `n < 2` (no spread to estimate a standard deviation
    from at all); a `None` here must read as "cannot size", never as "zero delta detectable"."""

    n: int
    k: int
    hit_rate_at_k: float
    hit_rate_at_k_ci: tuple[float, float]
    recall_at_k_ci: tuple[float, float, float]
    mrr_ci: tuple[float, float, float]
    ndcg_at_k_ci: tuple[float, float, float]
    detectable_effect_ndcg: float | None


def evaluate(
    cases: Sequence[CaseRetrieval],
    *,
    k: int,
    seed: int,
    n_resamples: int = DEFAULT_N_RESAMPLES,
) -> RetrievalReport:
    """Aggregate `cases` into one `RetrievalReport`. Every per case metric is `quality.ir_metrics`
    unmodified (no third variant of recall/nDCG/reciprocal rank lives here); every interval is
    `quality.stats` unmodified (no third variant of Wilson/bootstrap/power sizing lives here either).
    Raises `ValueError` on an empty `cases` sequence: an interval over zero cases is not an honestly
    wide interval, it is a meaningless one, so this refuses to manufacture one."""
    if not cases:
        raise ValueError("evaluate needs at least one case")

    hits = [ir_metrics.hit_rate_at_k(c.retrieved, c.relevant, k) for c in cases]
    recalls = [ir_metrics.recall_at_k(c.retrieved, c.relevant, k) for c in cases]
    ndcgs = [ir_metrics.ndcg_at_k(c.retrieved, c.relevant, k) for c in cases]
    reciprocal_ranks = [ir_metrics.reciprocal_rank(c.retrieved, c.relevant) for c in cases]

    n = len(cases)
    hit_successes = sum(1 for h in hits if h == 1.0)

    detectable_effect_ndcg = stdev(ndcgs) if n >= 2 else None
    if detectable_effect_ndcg is not None and detectable_effect_ndcg > 0.0:
        detectable_effect_ndcg = stats.detectable_effect(n, detectable_effect_ndcg)
    else:
        # n < 2 (no spread to measure) or every case tied exactly (sd == 0, the formula's own
        # division would be by zero): both read as "cannot size a detectable effect here", never
        # as "any nonzero delta is detectable", the wrong direction to fail silently toward.
        detectable_effect_ndcg = None

    return RetrievalReport(
        n=n,
        k=k,
        hit_rate_at_k=hit_successes / n,
        hit_rate_at_k_ci=stats.wilson_interval(hit_successes, n),
        recall_at_k_ci=stats.bootstrap_ci(recalls, seed=seed, n_resamples=n_resamples),
        mrr_ci=stats.bootstrap_ci(reciprocal_ranks, seed=seed, n_resamples=n_resamples),
        ndcg_at_k_ci=stats.bootstrap_ci(ndcgs, seed=seed, n_resamples=n_resamples),
        detectable_effect_ndcg=detectable_effect_ndcg,
    )


__all__ = ["CaseRetrieval", "RetrievalReport", "evaluate"]
