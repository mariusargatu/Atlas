"""Agreement between two verdict sequences, raw and chance corrected, with a resampled
interval. Both figures are reported because raw agreement alone looks convincing for a
judge that always approves; only the chance-corrected figure sees through that.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from atlas.config import Settings
from atlas.contracts import Question
from evals.judge import JudgeVerdict
from evals.labels import Label

_RESAMPLES = 2000
_UNDERPOWERED_WIDTH = 0.3
NOISE_FLOOR_SCHEMA_VERSION = "2.0.0"

# Where a recorded floor is written; not shipped in the repository. Run `noise_floor`
# against a real judge to produce one.
NOISE_FLOOR_PATH = "data/noise_floor.json"


def _kappa(human: Sequence[str], judged: Sequence[str]) -> float:
    n = len(human)
    raw = sum(1 for h, j in zip(human, judged, strict=True) if h == j) / n
    categories = set(human) | set(judged)
    chance = sum((human.count(c) / n) * (judged.count(c) / n) for c in categories)
    if chance >= 1.0:
        return 0.0
    return (raw - chance) / (1 - chance)


@dataclass(frozen=True, slots=True)
class Agreement:
    raw: float
    chance_corrected: float
    low: float
    high: float
    underpowered: bool


def agreement(human: Sequence[str], judged: Sequence[str], seed: int) -> Agreement:
    if not human:
        raise ValueError(
            "agreement needs at least one labelled answer, and was given none. An empty "
            "label store is the usual cause: record human verdicts with evals.labels."
        )
    n = len(human)
    raw = sum(1 for h, j in zip(human, judged, strict=True) if h == j) / n
    chance_corrected = _kappa(list(human), list(judged))

    rng = np.random.default_rng(seed)
    human_list, judged_list = list(human), list(judged)
    samples = []
    for _ in range(_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        samples.append(_kappa([human_list[i] for i in idx], [judged_list[i] for i in idx]))
    low, high = (float(v) for v in np.percentile(samples, [2.5, 97.5]))

    # A zero-width interval from n < 2 or a constant column means the resample could not
    # vary, not that the estimate is precise; flag those as underpowered explicitly rather
    # than let a degenerate interval read as a confident one.
    degenerate = n < 2 or len(set(human)) < 2 or len(set(judged)) < 2
    return Agreement(
        raw=raw, chance_corrected=chance_corrected, low=low, high=high,
        underpowered=degenerate or (high - low) > _UNDERPOWERED_WIDTH,
    )


def agreement_with_reference_noise(
    reference: Sequence[str], comparison: Sequence[str], flip_floor: float, seed: int,
) -> Agreement:
    """`agreement`, with the reference's own measured unreliability folded into the
    interval: for a judge whose verdict is used as a reference, `agreement` alone treats
    each question's recorded reference verdict as fixed and misses that a second reading
    of the same answer can disagree with the first. Each resample here also flips the
    reference verdict independently per question with probability `flip_floor`.
    `comparison` never flips: it is not the noise source being modeled.

    `flip_floor` has no default so a caller cannot pass 0.0 by omission and read the
    result as "measured and found to be zero." Only sensible at a floor well under 0.5;
    at 0.5 the resampled reference is a fair coin regardless of the judge's verdict and
    every scorer looks alike.
    """
    if not (0.0 <= flip_floor <= 1.0):
        raise ValueError(f"flip_floor must be a proportion in [0, 1], got {flip_floor}")
    n = len(reference)
    if n == 0:
        raise ValueError("agreement_with_reference_noise needs at least one paired verdict")
    categories = sorted(set(reference))
    if len(categories) != 2:
        raise ValueError(
            f"agreement_with_reference_noise needs a binary reference, got "
            f"{len(categories)} categories {categories}: flip_floor describes how often a "
            "judge changes ITS OWN verdict, which only has one meaning when there are "
            "exactly two verdicts to change between"
        )
    other = {categories[0]: categories[1], categories[1]: categories[0]}

    reference_list, comparison_list = list(reference), list(comparison)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        # Skips the flip draw at 0.0 so this is byte identical to `agreement`'s own
        # resample, not merely equal in expectation while consuming the RNG differently.
        if flip_floor > 0.0:
            flips = rng.random(n) < flip_floor
            resampled_reference = [
                other[reference_list[i]] if flip else reference_list[i]
                for i, flip in zip(idx, flips, strict=True)
            ]
        else:
            resampled_reference = [reference_list[i] for i in idx]
        resampled_comparison = [comparison_list[i] for i in idx]
        samples.append(_kappa(resampled_reference, resampled_comparison))
    low, high = (float(v) for v in np.percentile(samples, [2.5, 97.5]))

    raw = sum(1 for r, c in zip(reference, comparison, strict=True) if r == c) / n
    chance_corrected = _kappa(list(reference), list(comparison))
    degenerate = n < 2 or len(set(reference)) < 2 or len(set(comparison)) < 2
    return Agreement(
        raw=raw, chance_corrected=chance_corrected, low=low, high=high,
        underpowered=degenerate or (high - low) > _UNDERPOWERED_WIDTH,
    )


def agreement_against_labels(
    labels: Sequence[Label], verdicts: Sequence[JudgeVerdict], settings: Settings, seed: int = 0
) -> Agreement:
    """Refuses to mix a label written against one rubric version with verdicts produced
    under another, and refuses a mixed set of raters: either would silently average two
    different things under one number.
    """
    raters = {label.rater for label in labels}
    if len(raters) > 1:
        raise ValueError(
            f"labels come from {len(raters)} raters ({', '.join(sorted(raters))}), and one "
            "kappa over the union describes none of them. Group by rater and report each."
        )
    for label in labels:
        if label.rubric_version != settings.judge.rubric_version:
            raise ValueError(
                f"label {label.question} was written against rubric {label.rubric_version!r}, "
                f"settings name {settings.judge.rubric_version!r}"
            )
    by_question = {v.question: v.verdict for v in verdicts}
    human = [label.verdict for label in labels]
    judged = [by_question[label.question] for label in labels]
    return agreement(human, judged, seed)


@dataclass(frozen=True, slots=True)
class NoiseColumn:
    randomness: float
    run_averages: tuple[float, ...]
    per_question_self_agreement: float
    spread: float
    # How many repeats disagreed with each question's own majority verdict. Load bearing,
    # not just diagnostic: `flip_floor` is counted from this map because the scalar fields
    # above cannot be turned into a per-reading flip rate at any repeat count except two.
    disagreement: Mapping[str, int] = field(default_factory=dict)

    @property
    def unstable(self) -> tuple[str, ...]:
        """The questions the judge did not repeat itself on, worst first."""
        return tuple(
            name for name, count in
            sorted(self.disagreement.items(), key=lambda kv: (-kv[1], kv[0])) if count
        )

    @property
    def unanimity_rate(self) -> float:
        """The share of questions every one of the repeats agreed on exactly.

        Not a good threshold: unanimity over more readings is strictly harder to reach, so
        this describes run length as much as judge stability. Use `flip_floor` instead.
        """
        return self.per_question_self_agreement

    @property
    def flip_floor(self) -> float:
        """How often two readings of the same answer disagree, per pair of readings.

        Deliberately not `spread` (the range of aggregate pass rates, which cancels flips
        in opposite directions) and not `1 - unanimity_rate` (the chance any one of R
        repeats broke ranks, which only equals the pairwise disagreement rate at R=2).
        Both alternatives are biased upward by repeat count; this one is invariant to it
        (see tests/measurement/test_calibration.py).

        Counted from `disagreement`, where `m` is how many of the R repeats differed from
        that question's own majority: exactly `m * (R - m)` of that question's
        `R * (R - 1) / 2` unordered pairs disagree, unbiased for the per-call flip rate at
        every R.
        """
        repeats = len(self.run_averages)
        if repeats < 2 or not self.disagreement:
            raise ValueError(
                "a flip floor needs two or more repeats over one or more questions, and this "
                f"column holds {repeats} over {len(self.disagreement)}. A single reading of an "
                "answer cannot disagree with itself, and returning 0.0 for that would publish "
                "an unmeasured judge as one that never changes its mind."
            )
        pairs = repeats * (repeats - 1) / 2
        return sum(
            m * (repeats - m) / pairs for m in self.disagreement.values()
        ) / len(self.disagreement)


@dataclass(frozen=True, slots=True)
class NoiseFloor:
    at_configured: NoiseColumn
    at_zero: NoiseColumn


def _noise_column(
    questions: Sequence[Question], judge_fn: Callable[[Question, float], str],
    repeats: int, randomness: float,
) -> NoiseColumn:
    runs = [{q.name: judge_fn(q, randomness) for q in questions} for _ in range(repeats)]
    run_averages = tuple(sum(1 for v in run.values() if v == "pass") / len(run) for run in runs)
    agreeing = sum(1 for q in questions if len({run[q.name] for run in runs}) == 1)
    disagreement = {
        str(q.name): len(runs) - Counter(run[q.name] for run in runs).most_common(1)[0][1]
        for q in questions
    }
    return NoiseColumn(
        randomness=randomness, run_averages=run_averages,
        per_question_self_agreement=agreeing / len(questions),
        spread=max(run_averages) - min(run_averages),
        disagreement=disagreement,
    )


def noise_floor(
    questions: Sequence[Question], judge_fn: Callable[[Question, float], str],
    repeats: int, randomness: float,
) -> NoiseFloor:
    """Reruns the judge at both the configured randomness and at zero, as two independent
    sets of real calls: zero is not determinism for a hosted model.

    A model that rejects `temperature` (as every current Anthropic model does) makes these
    two columns the same request twice, indistinguishable from in here. `scripts/
    noise_floor.py` detects that case before spending money; the flip floor is still a
    real number either way, since it measures variation between repeats at a fixed
    setting.
    """
    return NoiseFloor(
        at_configured=_noise_column(questions, judge_fn, repeats, randomness),
        at_zero=_noise_column(questions, judge_fn, repeats, 0.0),
    )


def _column_from_payload(payload: dict[str, Any]) -> NoiseColumn:
    return NoiseColumn(
        randomness=float(payload["randomness"]),
        run_averages=tuple(float(v) for v in payload["run_averages"]),
        per_question_self_agreement=float(payload["per_question_self_agreement"]),
        spread=float(payload["spread"]),
        disagreement={str(k): int(v) for k, v in payload.get("disagreement", {}).items()},
    )


def record_noise_floor(floor: NoiseFloor, path: str | Path = NOISE_FLOOR_PATH) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": NOISE_FLOOR_SCHEMA_VERSION, "floor": asdict(floor)}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def recorded_noise_floor(path: str | Path = NOISE_FLOOR_PATH) -> NoiseFloor:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload["schema"] != NOISE_FLOOR_SCHEMA_VERSION:
        raise ValueError(
            f"noise floor file holds schema {payload['schema']!r}, expected {NOISE_FLOOR_SCHEMA_VERSION!r}"
        )
    data = payload["floor"]
    return NoiseFloor(
        at_configured=_column_from_payload(data["at_configured"]),
        at_zero=_column_from_payload(data["at_zero"]),
    )
