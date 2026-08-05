from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import numpy as np
import pytest

from atlas.config import Settings
from atlas.contracts import Question, QuestionName
from evals.calibration import (
    NOISE_FLOOR_SCHEMA_VERSION,
    agreement,
    agreement_against_labels,
    agreement_with_reference_noise,
    noise_floor,
    record_noise_floor,
    recorded_noise_floor,
)
from evals.labels import LABEL_SCHEMA_VERSION, Label

FIRST = Label(schema=LABEL_SCHEMA_VERSION, question=QuestionName("q0042"),
              rubric_version="1.1.0", verdict="fail", reason="price disagrees with the record",
              run_id="3f9c1a2b7d04", judge_verdict="pass", rater="human")


def test_an_always_approving_judge_reads_high_raw_agreement_and_chance_corrected_near_zero():
    # Why a raw agreement figure is never published on its own: a judge using one category
    # only scores 0.9 raw and exactly 0.0 once chance is taken out.
    result = agreement(["pass"] * 36 + ["fail"] * 4, ["pass"] * 40, seed=0)
    assert result.raw == pytest.approx(0.9)
    assert result.chance_corrected == pytest.approx(0.0)


def test_the_interval_is_reproducible_from_its_seed():
    # A published interval that cannot be recomputed is an anecdote.
    args = (["pass", "fail"] * 20, ["pass"] * 18 + ["fail"] * 22)
    assert agreement(*args, seed=0) == agreement(*args, seed=0)
    assert agreement(*args, seed=1) != agreement(*args, seed=0)


def test_forty_labels_flag_themselves_underpowered_and_two_thousand_do_not():
    # Both directions: a flag only tested where it is true is satisfied by a constant.
    # Three tenths is the width the calibration module defines as underpowered.
    human, judged = ["pass", "fail"] * 20, ["pass"] * 18 + ["fail"] * 22
    forty = agreement(human, judged, seed=0)
    plenty = agreement(human * 50, judged * 50, seed=0)
    assert forty.underpowered is True and forty.high - forty.low > 0.3
    assert plenty.underpowered is False
    assert plenty.high - plenty.low < forty.high - forty.low


def test_reference_noise_at_a_zero_floor_reproduces_plain_agreement_exactly():
    # At flip_floor=0.0 the implementation skips the extra draw entirely rather than
    # drawing and discarding it, so this is the same resample and not merely the same
    # expected value from a different one.
    reference, comparison = ["pass", "fail"] * 15, ["pass"] * 18 + ["fail"] * 12
    assert agreement_with_reference_noise(reference, comparison, flip_floor=0.0, seed=3) == (
        agreement(reference, comparison, seed=3)
    )


def test_reference_noise_pulls_the_interval_toward_zero_as_the_floor_rises():
    # Direction, not width: flipping a reference biases kappa toward zero rather than only
    # adding spread. Width grows on this fixture because the comparison agrees perfectly and
    # there is nowhere to go but down; it does not grow in general.
    #
    # 0.247 is the floor recorded in data/noise_floor.json. At 0.5 the resampled reference
    # is a fair coin regardless of what it said, so the function distinguishes nothing.
    reference = ["pass"] * 15 + ["fail"] * 15
    comparison = list(reference)  # perfect agreement before any reference noise
    low_floor = agreement_with_reference_noise(reference, comparison, flip_floor=0.05, seed=0)
    high_floor = agreement_with_reference_noise(reference, comparison, flip_floor=0.247, seed=0)
    assert high_floor.high < low_floor.high and high_floor.low < low_floor.low
    assert (high_floor.high - high_floor.low) > (low_floor.high - low_floor.low)
    # The point estimate is computed on the real, unflipped data, so only the interval moves.
    assert low_floor.chance_corrected == high_floor.chance_corrected == pytest.approx(1.0)


def test_reference_noise_refuses_a_reference_with_other_than_two_categories():
    # flip_floor describes how often a judge changes ITS OWN verdict, which has one
    # meaning only when there are exactly two verdicts to change between.
    with pytest.raises(ValueError, match="binary reference"):
        agreement_with_reference_noise(
            ["pass", "fail", "unsure"], ["pass", "fail", "pass"], flip_floor=0.2, seed=0
        )
    with pytest.raises(ValueError, match="binary reference"):
        agreement_with_reference_noise(["pass", "pass"], ["pass", "fail"], flip_floor=0.2, seed=0)


def test_reference_noise_refuses_a_flip_floor_outside_zero_to_one():
    with pytest.raises(ValueError, match=r"proportion in \[0, 1\]"):
        agreement_with_reference_noise(["pass", "fail"], ["pass", "fail"], flip_floor=1.5, seed=0)


def test_labels_written_against_another_scoring_guide_version_are_refused(verdicts):
    # Comparing labels written against one guide with verdicts produced under another
    # measures the edit to the guide and reports it as human agreement, and the number
    # looks entirely normal.
    stale = (replace(FIRST, rubric_version="0.9.0"),)
    with pytest.raises(ValueError, match="0.9.0"):
        agreement_against_labels(stale, verdicts, Settings())


@dataclass(frozen=True, slots=True)
class _Call:
    question: QuestionName
    randomness: float


@dataclass(slots=True)
class _ScriptedJudge:
    """Answers pass for every question except those named as flipping, where it
    alternates starting with fail, and records every call it received."""

    flipping: frozenset[QuestionName]
    calls: int = 0
    received: list[_Call] = field(default_factory=list)
    _next: dict[QuestionName, str] = field(default_factory=dict)

    def __call__(self, question, randomness: float) -> str:
        self.calls += 1
        self.received.append(_Call(question=question.name, randomness=randomness))
        if question.name not in self.flipping:
            return "pass"
        verdict = self._next.get(question.name, "fail")
        self._next[question.name] = "pass" if verdict == "fail" else "fail"
        return verdict


@pytest.fixture
def scripted_judge():
    return lambda flipping: _ScriptedJudge(flipping=frozenset(flipping))


def test_the_noise_floor_reports_both_randomness_settings_and_per_question_self_agreement(
    small, scripted_judge
):
    # Zero is not determinism for a hosted model, so both columns are reported. And an
    # average can look stable while every verdict flips, which only per question reveals.
    # The stub alternates the first question's verdict and no other, so both numbers are
    # known before the call rather than merely bounded.
    flipping = {small[0].name}
    floor = noise_floor(small, scripted_judge(flipping=flipping), repeats=5, randomness=0.7)
    assert floor.at_configured.randomness == 0.7 and floor.at_zero.randomness == 0.0
    assert len(floor.at_configured.run_averages) == 5
    assert floor.at_configured.per_question_self_agreement == pytest.approx(1 - 1 / len(small))
    assert floor.at_configured.spread > 0.0


def test_the_noise_floor_reruns_the_judge_at_zero_rather_than_copying(small, scripted_judge):
    stub = scripted_judge(flipping=frozenset())
    noise_floor(small, stub, repeats=5, randomness=0.7)
    assert stub.calls == 2 * 5 * len(small)                     # two columns, five repeats
    assert sorted({call.randomness for call in stub.received}) == [0.0, 0.7]


_FLIP_PROBABILITY = 0.15
_SIMULATED_QUESTIONS = 800


def _flip_prone_judge(probability: float, seed: int):
    """One settled verdict per question and a fixed chance of contradicting it on any single
    call, so two independent readings differ with probability 2p(1-p) at any repeat count."""
    rng = np.random.default_rng(seed)

    def call(question, randomness: float) -> str:
        settled = "pass" if int(str(question.name)[1:]) % 2 == 0 else "fail"
        flipped = bool(rng.random() < probability)
        return ("fail" if settled == "pass" else "pass") if flipped else settled

    return call


def test_the_flip_floor_does_not_move_with_the_number_of_repeats():
    """A floor that grows with the repeat count means measuring the judge more carefully
    raises the bar it has to clear, since every judge relation is a multiple of this figure.
    `1 - unanimity_rate` has that defect: against this simulated judge it reads 0.251 at two
    repeats and 0.961 at twenty."""
    questions = tuple(
        Question(name=QuestionName(f"q{i:04d}"), text="", kind="lookup", required=())
        for i in range(_SIMULATED_QUESTIONS)
    )
    columns = {
        repeats: noise_floor(
            questions, _flip_prone_judge(_FLIP_PROBABILITY, seed=0), repeats, 0.0
        ).at_configured
        for repeats in (2, 3, 5, 10, 20)
    }
    floors = {repeats: column.flip_floor for repeats, column in columns.items()}
    settled = 2 * _FLIP_PROBABILITY * (1 - _FLIP_PROBABILITY)   # 0.255
    assert all(value == pytest.approx(settled, abs=0.02) for value in floors.values()), floors
    assert max(floors.values()) - min(floors.values()) < 0.03, floors

    # The statistic it replaced, on the same runs: invariance says nothing without the
    # thing it is invariant unlike.
    unanimity_floors = {r: 1 - column.unanimity_rate for r, column in columns.items()}
    assert unanimity_floors[2] == pytest.approx(floors[2]), "the two agree at two repeats"
    assert unanimity_floors[20] > 0.9 > unanimity_floors[2], unanimity_floors


def test_the_flip_floor_refuses_a_column_that_cannot_hold_one():
    # Returning 0.0 for a single reading would publish an unmeasured judge as one that
    # never changes its mind.
    questions = (Question(name=QuestionName("q0000"), text="", kind="lookup", required=()),)
    once = noise_floor(questions, _flip_prone_judge(0.15, seed=0), repeats=1, randomness=0.0)
    with pytest.raises(ValueError, match="two or more repeats"):
        once.at_configured.flip_floor


def test_the_noise_floor_is_recorded_to_a_named_tracked_file(tmp_path, small, scripted_judge):
    # The judge bias relations read this exact path back, so the name has to be a fact
    # both files share rather than a convention neither states.
    floor = noise_floor(small, scripted_judge(flipping=frozenset()), repeats=5, randomness=0.7)
    path = tmp_path / "noise_floor.json"
    record_noise_floor(floor, path)
    assert recorded_noise_floor(path) == floor
    assert json.loads(path.read_text())["schema"] == NOISE_FLOOR_SCHEMA_VERSION
