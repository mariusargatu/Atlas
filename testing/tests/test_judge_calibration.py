"""`judge.calibration`, hermetic (SP8 task 2): the agreement report against a labelled set, the
kappa deployment gate routed through the SHARED `quality.gate.gate_on_lower_bound` (never a second
hand rolled `kappa_ci[1] >= bar` copy), and the AC1/prevalence companions D15 names alongside kappa.

Every fixture below is HAND TYPED, seeded labels, never real human gold: this module has no opinion
on where `human_labels` comes from (see `judge/calibration.py`'s own KAPPA HONESTY note), so a
"licensing report" here proves the ARITHMETIC licenses, not that anything may deploy on it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from judge.calibration import (
    AUTOMATION_BAR,
    KAPPA_VARIANCE_BUDGET,
    AgreementRow,
    CalibrationReport,
    calibrate,
    order_swap_flip_rate,
)
from judge.contract import JudgeContract
from quality.gate import GateVerdict, gate_on_lower_bound

_CONTRACT = JudgeContract("gpt-judge", "groundedness-v1", "tmpl-abc123")
_CLOCK_INSTANT = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

# A near perfect, moderately sized labelled set (one disagreement in ten, repeated): kappa=0.80 and
# its lower bound clears 0.6 at this n, so this is a "licensing report" candidate.
_LICENSING_IDS = [f"case-{i:02d}" for i in range(50)]
_LICENSING_HUMAN = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 5
_LICENSING_JUDGE = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1] * 5

# A near chance labelled set (alternating judge labels against a skewed human split): kappa=0.20 and
# its lower bound sits well under 0.6, a "rejected report" candidate.
_REJECTED_IDS = [f"case-{i:02d}" for i in range(100)]
_REJECTED_HUMAN = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 10
_REJECTED_JUDGE = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10


def _licensing_report() -> CalibrationReport:
    return calibrate(_CONTRACT, _LICENSING_IDS, _LICENSING_HUMAN, _LICENSING_JUDGE, generated_at=_CLOCK_INSTANT)


def _rejected_report() -> CalibrationReport:
    return calibrate(_CONTRACT, _REJECTED_IDS, _REJECTED_HUMAN, _REJECTED_JUDGE, generated_at=_CLOCK_INSTANT)


# ---- a licensing report: the kappa lower bound clears the bar ----------------------------------


def test_a_licensing_report_clears_the_bar_on_the_lower_bound_not_just_the_point():
    report = _licensing_report()
    assert round(report.kappa, 2) == 0.80
    assert report.kappa_ci[1] >= AUTOMATION_BAR   # the floor, the licence bearing quantity
    assert report.gate_decision.verdict is GateVerdict.PASS
    assert report.licensed is True


# ---- a rejected report: the kappa lower bound sits below the bar -------------------------------


def test_a_rejected_report_misses_the_bar_on_the_lower_bound():
    report = _rejected_report()
    assert round(report.kappa, 2) == 0.20
    assert report.kappa_ci[1] < AUTOMATION_BAR
    assert report.gate_decision.verdict is GateVerdict.FAIL
    assert report.licensed is False


def test_raw_agreement_flatters_the_rejected_report_more_than_kappa_does():
    report = _rejected_report()
    assert report.raw_agreement > report.kappa + 0.25


# ---- the gate routes through the SHARED gate_on_lower_bound, never a second copy ---------------


def test_licensed_calls_the_shared_gate_on_lower_bound_not_a_hand_rolled_comparison():
    report = _licensing_report()
    with patch("judge.calibration.gate_on_lower_bound", wraps=gate_on_lower_bound) as spy:
        assert report.licensed is True
    spy.assert_called_once()
    _, lo, hi = report.kappa_ci
    called_interval, called_kwargs = spy.call_args.args[0], spy.call_args.kwargs
    assert called_interval == (lo, hi)
    assert called_kwargs == {"threshold": report.bar, "variance_budget": report.variance_budget}


def test_a_quarantined_interval_is_not_licensed_either():
    # A tiny n with a real disagreement: the interval is wide enough to exceed the module's own
    # variance budget, so the gate reads QUARANTINE ("too wide to call"), not a licence, whichever
    # side of the bar the point estimate sits on.
    report = calibrate(
        _CONTRACT, ["a", "b", "c", "d"], [1, 1, 0, 0], [1, 0, 0, 0], generated_at=_CLOCK_INSTANT
    )
    assert report.gate_decision.verdict is GateVerdict.QUARANTINE
    assert report.licensed is False


# ---- AC1 and prevalence, D15's companions to kappa ----------------------------------------------


def test_ac1_and_prevalence_are_reported_alongside_kappa_never_instead_of_it():
    report = _licensing_report()
    assert 0.0 <= report.ac1 <= 1.0
    assert 0.0 <= report.prevalence <= 1.0
    # AC1 and kappa need not be pinned to the same value; both are read, neither substitutes.
    assert isinstance(report.kappa, float)


def test_ac1_stays_higher_than_kappa_at_the_same_prevalence_paradox_shape_calibration_uses():
    # Mirrors quality/stats.py's own kappa paradox pin: extreme prevalence, one disagreement.
    ids = [f"c{i}" for i in range(10)]
    human = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    judge = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    report = calibrate(_CONTRACT, ids, human, judge, generated_at=_CLOCK_INSTANT)
    assert report.kappa < 0
    assert report.ac1 > 0.7


# ---- byte reproducible under the frozen clock ---------------------------------------------------


def test_report_render_is_byte_reproducible_under_the_same_frozen_instant():
    first = calibrate(
        _CONTRACT, _LICENSING_IDS, _LICENSING_HUMAN, _LICENSING_JUDGE, generated_at=_CLOCK_INSTANT
    )
    second = calibrate(
        _CONTRACT, _LICENSING_IDS, _LICENSING_HUMAN, _LICENSING_JUDGE, generated_at=_CLOCK_INSTANT
    )
    assert first.render() == second.render()
    assert "2026-06-15T12:00:00+00:00" in first.render()


def test_report_render_differs_when_the_frozen_instant_differs():
    other_instant = datetime(2026, 7, 20, 9, 30, 0, tzinfo=timezone.utc)
    first = calibrate(
        _CONTRACT, _LICENSING_IDS, _LICENSING_HUMAN, _LICENSING_JUDGE, generated_at=_CLOCK_INSTANT
    )
    second = calibrate(
        _CONTRACT, _LICENSING_IDS, _LICENSING_HUMAN, _LICENSING_JUDGE, generated_at=other_instant
    )
    assert first.render() != second.render()


# ---- report rendering and input validation -------------------------------------------------------


def test_calibration_report_renders_the_contract_and_verdict():
    text = _rejected_report().render()
    assert "gpt-judge" in text and "NOT licensed" in text and "kappa" in text.lower()


def test_calibrate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        calibrate(_CONTRACT, ["only-one-id"], [1, 0], [1], generated_at=_CLOCK_INSTANT)


def test_calibrate_rejects_an_empty_calibration():
    with pytest.raises(ValueError):
        calibrate(_CONTRACT, [], [], [], generated_at=_CLOCK_INSTANT)


def test_agreement_row_agree_reads_human_vs_judge():
    assert AgreementRow("c1", human=1, judge=1).agree is True
    assert AgreementRow("c1", human=1, judge=0).agree is False


def test_calibrate_uses_the_default_bar_and_variance_budget_when_not_given():
    report = _licensing_report()
    assert report.bar == AUTOMATION_BAR
    assert report.variance_budget == KAPPA_VARIANCE_BUDGET


# ---- order swap flip rate (position bias), absorbed unchanged -----------------------------------


def test_order_swap_flip_rate_counts_only_inconsistent_pairs():
    assert order_swap_flip_rate([(0, 0), (1, 1)]) == 0.0
    assert order_swap_flip_rate([(0, 1), (1, 1)]) == 0.5
    assert order_swap_flip_rate([]) == 0.0
