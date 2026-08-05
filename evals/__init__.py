"""MEASUREMENTS is an explicit registry of scored metrics, not everything the
package exports. Every entry here must be caught by a broken system in
evals.adversaries or carry a written exemption reason in EXEMPT_BY_DESIGN."""

from __future__ import annotations

from evals.calibration import agreement
from evals.ir_metrics import (
    graded_ndcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    success_at_k,
)
from evals.stats import paired_comparison

MEASUREMENTS = (
    recall_at_k,
    precision_at_k,
    reciprocal_rank_at_k,
    ndcg_at_k,
    graded_ndcg_at_k,
    success_at_k,
    agreement,
    paired_comparison,
)

EXEMPT_BY_DESIGN = {
    "recall_at_k": (
        "a search that returns everything scores perfectly on recall by construction; "
        "search_returns_everything in evals.adversaries.BROKEN_SYSTEMS is wired to "
        "precision instead"
    ),
    "ndcg_at_k": (
        "rank-based, so a full-corpus ranking still places gold reasonably high and this "
        "cannot fire on search_returns_everything; precision is the metric that notices"
    ),
    "reciprocal_rank_at_k": (
        "no broken system moves the rank of the first correct chunk without also moving "
        "precision, which is wired"
    ),
    "graded_ndcg_at_k": (
        "scores only the 22 tau2 tasks whose primary documents the benchmark's own criteria "
        "name, and shares ndcg_at_k's rank-based blind spot"
    ),
    "agreement": (
        "catches a judge that approves everything, and a judge is not a field of "
        "evals.adversaries.System (which carries a retrieval and an answer, not a verdict). "
        "Asserted directly on hand-computable values by tests/measurement/test_calibration.py"
        "::test_an_always_approving_judge_reads_high_raw_agreement_and_chance_corrected_near_zero"
    ),
    "paired_comparison": (
        "catches a reranker that returns its input and is credited with a gain, but the "
        "shipped reranker backend is passthrough by design, so that system is production "
        "configuration rather than broken. Asserted by tests/measurement/test_report.py"
        "::test_the_reranking_verdict_reports_quality_money_and_response_time_together"
    ),
}
