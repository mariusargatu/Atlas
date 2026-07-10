"""P2, statistics: gate on the lower bound of the interval, never the point.

A point of 0.84 with a floor of 0.78 has not cleared a 0.80 bar. Two companions ride
along: a variance budget (an interval wider than the decision tolerates is an unproven
claim, not a pass) and a quarantine (a result too wide to call gets rerun, not shipped).
"""
from __future__ import annotations

import dataclasses

import pytest

from evals.gate import GateVerdict, gate_on_lower_bound
from evals.stats import wilson_interval


def test_floor_above_threshold_passes():
    d = gate_on_lower_bound((0.82, 0.90), threshold=0.80, variance_budget=0.15)
    assert d.verdict is GateVerdict.PASS


def test_point_above_but_floor_below_fails_closed():
    # The cold open shape: a best guess above the line and an honest floor below it.
    # Shipping on the best guess is shipping on optimism, so the gate fails closed.
    d = gate_on_lower_bound((0.78, 0.89), threshold=0.80, variance_budget=0.15)
    assert d.verdict is GateVerdict.FAIL


def test_confidently_below_threshold_fails():
    d = gate_on_lower_bound((0.60, 0.70), threshold=0.80, variance_budget=0.15)
    assert d.verdict is GateVerdict.FAIL


def test_interval_wider_than_the_variance_budget_quarantines():
    # 0.70..0.95 straddles the bar with a 0.25 spread: too wide to call, rerun it.
    d = gate_on_lower_bound((0.70, 0.95), threshold=0.80, variance_budget=0.15)
    assert d.verdict is GateVerdict.QUARANTINE
    assert "rerun" in d.reason


def test_a_wide_interval_is_an_unproven_claim_even_when_the_floor_clears():
    # Width over budget quarantines BEFORE the threshold is consulted: the budget is a
    # statement about how much spread the decision tolerates, not about where it sits.
    d = gate_on_lower_bound((0.81, 0.99), threshold=0.80, variance_budget=0.15)
    assert d.verdict is GateVerdict.QUARANTINE


def test_width_exactly_at_budget_is_within_budget():
    d = gate_on_lower_bound((0.82, 0.97), threshold=0.80, variance_budget=0.15)
    assert d.verdict is GateVerdict.PASS


def test_lower_bound_exactly_at_threshold_clears():
    d = gate_on_lower_bound((0.80, 0.90), threshold=0.80, variance_budget=0.15)
    assert d.verdict is GateVerdict.PASS


def test_decision_carries_the_numbers_it_was_made_from():
    d = gate_on_lower_bound((0.78, 0.89), threshold=0.80, variance_budget=0.15)
    assert d.lower_bound == 0.78
    assert d.width == 0.11  # stored already rounded: the exact value the verdict used
    assert d.threshold == 0.80
    assert d.variance_budget == 0.15
    assert d.reason  # never a bare verdict, the reason is part of the decision


def test_stored_width_agrees_with_the_verdict():
    # 0.97 - 0.82 is 0.15000000000000002 in floats. The verdict reads the rounded width,
    # so the decision must carry that same number or a downstream
    # `width <= variance_budget` re-check contradicts the PASS it sits next to.
    d = gate_on_lower_bound((0.82, 0.97), threshold=0.80, variance_budget=0.15)
    assert d.verdict is GateVerdict.PASS
    assert d.width <= d.variance_budget


def test_float_noisy_budget_does_not_quarantine():
    # 0.7 - 0.4 is 0.29999999999999993 in floats: a width the caller wrote as 0.30
    # against a budget the caller wrote as 0.30 must not read as "0.300 exceeds the
    # 0.300 budget". The comparison rounds the difference, not just one side.
    d = gate_on_lower_bound((0.30, 0.60), threshold=0.50, variance_budget=0.7 - 0.4)
    assert d.verdict is GateVerdict.FAIL  # the floor missed the bar, not a quarantine


def test_nonfinite_interval_raises():
    # NaN compares False against everything: unvalidated, (0.85, nan) would sail past
    # the inversion, width, and threshold checks and PASS, the gate failing OPEN.
    with pytest.raises(ValueError):
        gate_on_lower_bound((0.85, float("nan")), threshold=0.80, variance_budget=0.15)
    with pytest.raises(ValueError):
        gate_on_lower_bound((float("-inf"), 0.90), threshold=0.80, variance_budget=0.15)


def test_nonfinite_threshold_and_budget_raise():
    with pytest.raises(ValueError):
        gate_on_lower_bound((0.80, 0.90), threshold=float("nan"), variance_budget=0.15)
    with pytest.raises(ValueError):
        gate_on_lower_bound((0.80, 0.90), threshold=0.80, variance_budget=float("nan"))
    with pytest.raises(ValueError):
        gate_on_lower_bound((0.80, 0.90), threshold=0.80, variance_budget=float("inf"))


def test_decision_is_immutable():
    d = gate_on_lower_bound((0.82, 0.90), threshold=0.80, variance_budget=0.15)
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(d, "verdict", GateVerdict.PASS)


def test_gate_composes_with_wilson_directly():
    # The intended call shape: a pass rate's Wilson pair feeds the gate unmodified.
    d = gate_on_lower_bound(wilson_interval(81, 100), threshold=0.80, variance_budget=0.20)
    assert d.verdict is GateVerdict.FAIL  # point 0.81, floor ~0.722: not cleared


def test_inverted_interval_raises():
    with pytest.raises(ValueError):
        gate_on_lower_bound((0.9, 0.8), threshold=0.80, variance_budget=0.15)


def test_nonpositive_variance_budget_raises():
    with pytest.raises(ValueError):
        gate_on_lower_bound((0.8, 0.9), threshold=0.80, variance_budget=0.0)
