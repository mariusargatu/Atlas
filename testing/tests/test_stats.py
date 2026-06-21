"""P2, statistics: a score without an interval is an anecdote (07-statistics.md)."""
from __future__ import annotations

import pytest

from evals.stats import intervals_overlap, wilson_interval


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
