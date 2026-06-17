"""P6, judge calibration: raw agreement flatters, Cohen's kappa reveals (06-judge-calibration)."""
from __future__ import annotations

from stats import cohen_kappa


def test_perfect_agreement_is_one():
    assert cohen_kappa([1, 0, 1, 1, 0], [1, 0, 1, 1, 0]) == 1.0


def test_raw_agreement_flatters_but_kappa_reveals_the_lying_judge():
    # Humans: 8 good, 2 bad. The judge calls everything good: 80% raw agreement on the
    # dashboard, but no better than chance once corrected.
    humans = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    judge = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    raw = sum(1 for h, j in zip(humans, judge) if h == j) / len(humans)
    assert raw == 0.8
    assert abs(cohen_kappa(humans, judge)) < 0.01


def test_a_calibrated_judge_clears_the_point_six_bar():
    humans = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    judge = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]  # 8/10 agree, balanced
    assert cohen_kappa(humans, judge) >= 0.6
