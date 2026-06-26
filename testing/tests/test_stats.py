"""P2, statistics: a score without an interval is an anecdote."""
from __future__ import annotations

import pytest

from evals.stats import cohen_kappa, intervals_overlap, wilson_interval


def test_close_scores_have_overlapping_intervals():
    # 84% vs 81% on 100 items: you cannot yet claim one beats the other.
    assert intervals_overlap(wilson_interval(84, 100), wilson_interval(81, 100))


def test_clearly_different_scores_do_not_overlap():
    assert not intervals_overlap(wilson_interval(95, 100), wilson_interval(50, 100))


def test_interval_is_within_bounds():
    lo, hi = wilson_interval(84, 100)
    assert 0.0 <= lo < 0.84 < hi <= 1.0


def test_zero_trials_is_the_widest_interval_not_false_certainty():
    # No data cannot claim a 0% pass rate with certainty; the honest interval is the whole range.
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_invalid_counts_raise():
    with pytest.raises(ValueError):
        wilson_interval(5, 3)  # successes > n is nonsense, not a silent 1.0+ rate


def test_kappa_pins_the_formula_on_an_asymmetric_case():
    # Golden value over UNEQUAL marginals (pa=0.3, pb=0.2), the way wilson is pinned. A mutant that
    # drops the (1-pa)(1-pb) term from the expected agreement reads 0.89 here, so this pin kills it.
    a = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    b = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert round(cohen_kappa(a, b), 4) == 0.7368


def test_kappa_is_one_when_both_raters_always_agree():
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_kappa_rejects_mismatched_or_empty_input():
    with pytest.raises(ValueError):
        cohen_kappa([1, 0], [1])
    with pytest.raises(ValueError):
        cohen_kappa([], [])
