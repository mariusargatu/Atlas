from __future__ import annotations

import math

import pytest

from atlas.contracts import QuestionName
from evals.report import comparison_header
from evals.stats import (
    average_by_kind,
    paired_comparison,
    resample_interval,
    smallest_resolvable_difference,
)

QUESTIONS = tuple(QuestionName(f"q{i:04d}") for i in range(200))


def paired_inputs() -> tuple[dict[QuestionName, float], dict[QuestionName, float]]:
    # Large variation between questions, a small and varying change within each question: the
    # shape pairing exists for. A constant offset instead makes the paired interval zero width,
    # which the comparison below then satisfies even for an implementation that never paired.
    before = {q: 0.20 + 0.60 * ((i * 37) % 200) / 199 for i, q in enumerate(QUESTIONS)}
    after = {q: before[q] + 0.04 + 0.02 * ((i * 13) % 7) / 6 for i, q in enumerate(QUESTIONS)}
    return before, after


def test_averaging_partitions_by_question_kind_before_it_averages() -> None:
    scored = {(QuestionName("q1"), "lookup"): 1.0, (QuestionName("q2"), "lookup"): 0.0,
              (QuestionName("q3"), "absent_entity"): float("nan")}
    result = average_by_kind(scored)      # the probe kind is its own number, never mixed into recall
    assert result["lookup"] == 0.5 and math.isnan(result["absent_entity"])


def test_averaging_refuses_a_non_number_inside_a_kind_that_should_have_an_answer() -> None:
    # The probe kinds may carry not a number and are reported on their own. A lookup question
    # carrying one is a resolver bug, and averaging it would return not a number for the whole
    # kind, which reads as a tooling problem rather than as a measurement problem.
    with pytest.raises(ValueError, match="lookup"):
        average_by_kind({(QuestionName("q1"), "lookup"): float("nan")})


@pytest.mark.parametrize("values", [[0.5, float("nan"), 0.7], [0.5, float("inf"), 0.7], []])
def test_the_interval_function_refuses_input_it_cannot_resample(values) -> None:
    # A single not-a-number silently poisons every headline figure. An empty list is the same
    # failure from the other end: an interval built from nothing still prints as two numbers.
    with pytest.raises(ValueError):
        resample_interval(values, seed=0)


def test_a_paired_comparison_is_tighter_than_two_independent_intervals() -> None:
    before, after = paired_inputs()
    paired = paired_comparison(before, after, seed=0)
    low, high = resample_interval(list(before.values()), seed=0)
    # A width strictly above zero rules out the degenerate interval an unvarying input gives. An
    # implementation that resamples the two sides independently lands at about the width on the
    # right or wider, so the middle comparison is the one that detects a missing pairing.
    assert 0.0 < (paired.high - paired.low) < (high - low)
    assert paired.low > 0.0 and paired.questions == 200
    assert paired.low <= 0.05 <= paired.high      # it covers the change it was handed


def test_pairing_is_on_question_names_and_not_on_list_length() -> None:
    before, after = paired_inputs()
    renamed = {QuestionName(f"other{i:04d}"): v for i, v in enumerate(after.values())}
    for other in (renamed, dict(list(after.items())[:199])):
        with pytest.raises(ValueError):
            paired_comparison(before, other, seed=0)


def test_resample_interval_reproduces_exactly_from_its_seed() -> None:
    # An implementation that drops the seed (np.random.default_rng() instead of
    # np.random.default_rng(seed)) passes every other test of this function.
    values = [0.12, 0.44, 0.44, 0.71, 0.93, 0.20, 0.55, 0.08]
    assert resample_interval(values, seed=7) == resample_interval(values, seed=7)
    assert resample_interval(values, seed=7) != resample_interval(values, seed=8)


def test_smallest_resolvable_difference_matches_its_documented_value_at_97_questions() -> None:
    # 0.100 shipped once, from using 0.5 as the worst case standard deviation of a paired
    # difference instead of 1.0. The relative scaling checks pass under both constants,
    # since each gives the same sqrt(5) ratio, so only an absolute pin catches the regression.
    assert smallest_resolvable_difference(97) == pytest.approx(0.199, abs=0.0005)


def test_every_comparison_header_carries_the_smallest_resolvable_difference() -> None:
    forty, two_hundred = smallest_resolvable_difference(40), smallest_resolvable_difference(200)
    # Derived rather than pinned to two remembered figures: any decreasing sequence passes the
    # first line, only the square root relationship passes the second.
    assert forty > two_hundred > 0.0
    assert forty / two_hundred == pytest.approx(math.sqrt(5), rel=0.05)
    header = comparison_header("recursive", "fixed", questions=200)
    assert header.resolvable_difference == two_hundred
    assert f"{two_hundred:.2f}" in header.text
    assert "recursive" in header.text and "fixed" in header.text
