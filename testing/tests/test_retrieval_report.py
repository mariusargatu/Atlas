"""`quality.retrieval_report`, hermetic: the SP7 Task 7 aggregation layer over hand computed
`CaseRetrieval` fixtures (no retriever, no network, no live TEI/pgvector anywhere in this file).
The live half (`test_sp7_retrieval_metrics_live.py`) supplies real per case results to the same
`evaluate` function this module tests in isolation.
"""
from __future__ import annotations

import math

import pytest

from quality import ir_metrics, stats
from quality.retrieval_report import CaseRetrieval, evaluate

# Four cases, k=3, hand worked so recall/hit rate are checkable by inspection and nDCG/MRR are
# cross checked against `quality.ir_metrics` directly (never a second, hand rederived formula).
#   A: relevant {a}, retrieved (a, b, c)  -> hit, recall 1.0, first relevant at rank 1
#   B: relevant {a}, retrieved (b, a, c)  -> hit, recall 1.0, first relevant at rank 2
#   C: relevant {a}, retrieved (b, c, d)  -> miss, recall 0.0, never found
#   D: relevant {a, b}, retrieved (a, c, b) -> hit, recall 1.0 (both found), first relevant at rank 1
_CASES = (
    CaseRetrieval("case-a", ("a", "b", "c"), frozenset({"a"})),
    CaseRetrieval("case-b", ("b", "a", "c"), frozenset({"a"})),
    CaseRetrieval("case-c", ("b", "c", "d"), frozenset({"a"})),
    CaseRetrieval("case-d", ("a", "c", "b"), frozenset({"a", "b"})),
)
_K = 3
_SEED = 20260720


def test_evaluate_needs_at_least_one_case():
    with pytest.raises(ValueError, match="at least one case"):
        evaluate([], k=_K, seed=_SEED)


def test_n_and_k_are_recorded_on_the_report():
    report = evaluate(_CASES, k=_K, seed=_SEED)
    assert report.n == 4
    assert report.k == _K


def test_hit_rate_at_k_is_the_plain_fraction_of_cases_with_any_relevant_chunk_in_top_k():
    # A, B, D hit; C misses: 3 of 4.
    report = evaluate(_CASES, k=_K, seed=_SEED)
    assert report.hit_rate_at_k == pytest.approx(3 / 4)


def test_hit_rate_ci_is_the_real_wilson_interval_not_a_reinvented_one():
    report = evaluate(_CASES, k=_K, seed=_SEED)
    assert report.hit_rate_at_k_ci == stats.wilson_interval(3, 4)


def test_recall_ci_point_is_the_mean_of_the_real_per_case_recall_at_k():
    report = evaluate(_CASES, k=_K, seed=_SEED)
    recalls = [ir_metrics.recall_at_k(c.retrieved, c.relevant, _K) for c in _CASES]
    assert recalls == [1.0, 1.0, 0.0, 1.0]
    point, lo, hi = report.recall_at_k_ci
    assert point == pytest.approx(sum(recalls) / len(recalls))
    assert lo <= point <= hi


def test_recall_ci_delegates_to_the_real_bootstrap_ci_with_the_given_seed():
    # Composition, not reinvention: the exact same seed must reproduce the exact same interval.
    report = evaluate(_CASES, k=_K, seed=_SEED)
    recalls = [ir_metrics.recall_at_k(c.retrieved, c.relevant, _K) for c in _CASES]
    expected = stats.bootstrap_ci(recalls, seed=_SEED)
    assert report.recall_at_k_ci == expected


def test_mrr_ci_point_matches_the_real_per_case_reciprocal_rank():
    report = evaluate(_CASES, k=_K, seed=_SEED)
    rrs = [ir_metrics.reciprocal_rank(c.retrieved, c.relevant) for c in _CASES]
    assert rrs == [1.0, 0.5, 0.0, 1.0]
    point, _, _ = report.mrr_ci
    assert point == pytest.approx(sum(rrs) / len(rrs))


def test_ndcg_ci_point_matches_the_real_per_case_ndcg_at_k():
    report = evaluate(_CASES, k=_K, seed=_SEED)
    ndcgs = [ir_metrics.ndcg_at_k(c.retrieved, c.relevant, _K) for c in _CASES]
    # Hand worked: A = 1.0; B = (1/log2(3)) / 1.0; C = 0.0;
    # D = (1/log2(2) + 1/log2(4)) / (1/log2(2) + 1/log2(3)).
    expected_b = (1 / math.log2(3)) / 1.0
    expected_d = (1 / math.log2(2) + 1 / math.log2(4)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcgs == pytest.approx([1.0, expected_b, 0.0, expected_d])
    point, _, _ = report.ndcg_at_k_ci
    assert point == pytest.approx(sum(ndcgs) / len(ndcgs))


def test_seeded_bootstraps_are_byte_reproducible_across_two_calls():
    first = evaluate(_CASES, k=_K, seed=_SEED)
    second = evaluate(_CASES, k=_K, seed=_SEED)
    assert first == second


def test_a_different_seed_may_change_the_interval_but_never_the_point():
    a = evaluate(_CASES, k=_K, seed=1)
    b = evaluate(_CASES, k=_K, seed=2)
    assert a.recall_at_k_ci[0] == b.recall_at_k_ci[0]  # the point is the real data, seed free
    assert a.ndcg_at_k_ci[0] == b.ndcg_at_k_ci[0]


def test_detectable_effect_ndcg_matches_stats_detectable_effect_at_this_n():
    from statistics import stdev

    report = evaluate(_CASES, k=_K, seed=_SEED)
    ndcgs = [ir_metrics.ndcg_at_k(c.retrieved, c.relevant, _K) for c in _CASES]
    sd = stdev(ndcgs)
    assert report.detectable_effect_ndcg == pytest.approx(stats.detectable_effect(4, sd))


def test_detectable_effect_ndcg_is_none_for_a_single_case_no_spread_to_measure():
    report = evaluate([_CASES[0]], k=_K, seed=_SEED)
    assert report.detectable_effect_ndcg is None


def test_detectable_effect_ndcg_is_none_when_every_case_ties_exactly_sd_zero():
    # Two cases with identical nDCG (both perfect hits at rank 1): sd is 0, the formula's own
    # division would be by zero. "Cannot size" (None), never "any nonzero delta is detectable".
    tied = (
        CaseRetrieval("t1", ("a", "b"), frozenset({"a"})),
        CaseRetrieval("t2", ("x", "y"), frozenset({"x"})),
    )
    report = evaluate(tied, k=2, seed=_SEED)
    assert report.detectable_effect_ndcg is None


def test_a_worse_retriever_scores_lower_hit_rate_than_a_better_one():
    # Sanity: the aggregation direction is not accidentally inverted.
    good = (
        CaseRetrieval("g1", ("a",), frozenset({"a"})),
        CaseRetrieval("g2", ("b",), frozenset({"b"})),
    )
    bad = (
        CaseRetrieval("b1", ("z",), frozenset({"a"})),
        CaseRetrieval("b2", ("z",), frozenset({"b"})),
    )
    good_report = evaluate(good, k=1, seed=_SEED)
    bad_report = evaluate(bad, k=1, seed=_SEED)
    assert good_report.hit_rate_at_k > bad_report.hit_rate_at_k
