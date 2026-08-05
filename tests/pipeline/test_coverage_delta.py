from __future__ import annotations

from scripts.coverage_delta import _TOLERANCE_PERCENT, verdict


def test_no_baseline_passes_and_says_how_to_record_one():
    code, message = verdict(current=87.3, baseline=None)
    assert code == 0 and "no baseline recorded" in message and "87.30" in message


def test_a_drop_within_tolerance_passes():
    code, _ = verdict(current=80.0, baseline=80.0 + _TOLERANCE_PERCENT)
    assert code == 0


def test_a_drop_past_tolerance_fails_and_names_the_baseline_file():
    code, message = verdict(current=80.0, baseline=80.0 + _TOLERANCE_PERCENT + 0.01)
    assert code == 1 and "REGRESSION" in message and "coverage_baseline.json" in message


def test_a_rise_past_tolerance_still_passes_but_says_to_update_the_baseline():
    code, message = verdict(current=90.0, baseline=90.0 - _TOLERANCE_PERCENT - 0.01)
    assert code == 0 and "rose" in message and "coverage_baseline.json" in message


def test_this_is_a_regression_check_never_an_absolute_floor():
    code, _ = verdict(current=5.0, baseline=5.0)
    assert code == 0
