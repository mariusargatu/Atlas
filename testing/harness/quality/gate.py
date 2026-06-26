"""Release gating on the honest bound: gate on the lower end of the interval, never
the point. A point of 0.84 with a floor of 0.78 has not cleared a 0.80 bar. It has a
best guess above the line and an honest floor below it, and shipping on the best guess
is shipping on optimism.

Two companions ride along:

- a **variance budget**: an interval wider than the decision tolerates is an unproven
  claim, not a pass, whatever side of the bar it sits on.
- a **quarantine**: a result too wide to call gets rerun with more items or more
  trials instead of shipped as a coin flip.

The eval lane does not hard gate a merge the way regression does, but where a release
turns on a tracked metric this is the rule, and it is conservative by design: "we
cannot tell yet" fails closed. The gate consumes a plain (lo, hi) interval, so it
composes directly with `stats.wilson_interval`. The (point, lo, hi) triples from
`stats.mean_interval` and the bootstraps unpack to (lo, hi) first.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class GateVerdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    QUARANTINE = "quarantine"


@dataclass(frozen=True)
class GateDecision:
    """The verdict plus every number it was made from, so a report never shows a bare word.

    `width` is the value the verdict was actually computed on (rounded to the same
    9 decimal discipline as the comparison), so a downstream `width <= variance_budget`
    recheck agrees with the verdict instead of contradicting it on float noise.
    """

    verdict: GateVerdict
    reason: str
    lower_bound: float
    width: float
    threshold: float
    variance_budget: float


def gate_on_lower_bound(
    interval: tuple[float, float],
    *,
    threshold: float,
    variance_budget: float,
) -> GateDecision:
    """Gate a tracked metric's confidence interval against a release bar.

    Checked in this order, because each rule is a stronger statement than the next:

    1. width > variance_budget -> QUARANTINE. The spread exceeds what the decision
       tolerates, so the measurement is under resourced whatever it says. Rerun it.
    2. lower bound >= threshold -> PASS. The honest floor clears the bar.
    3. otherwise -> FAIL, closed. The point may sit above the bar, but the floor does not.
    """
    lo, hi = interval
    # Finite or refused: NaN compares False against everything, so an uncomputable
    # interval would sail past every check below and fail OPEN, the one direction this
    # gate must never fail. Raising keeps "we cannot tell" closed.
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"interval must be finite, got ({lo}, {hi})")
    if lo > hi:
        raise ValueError(f"interval is inverted: lo={lo} > hi={hi}")
    if not math.isfinite(threshold):
        raise ValueError(f"threshold must be finite, got {threshold}")
    if not (math.isfinite(variance_budget) and variance_budget > 0):
        raise ValueError(f"variance_budget must be positive and finite, got {variance_budget}")
    # Rounded once, here, the same discipline as stats._level_bounds: hi - lo is not
    # always exact as a float (0.97 - 0.82 = 0.15000000000000002), a width exactly at
    # the budget must read as within it, and the stored width must agree with the
    # verdict. The budget enters through the rounded DIFFERENCE so a float noisy budget
    # (0.7 - 0.4 = 0.29999999999999993) cannot flip the verdict either.
    width = round(hi - lo, 9)
    if round(width - variance_budget, 9) > 0:
        verdict = GateVerdict.QUARANTINE
        reason = (
            f"interval width {width:.3f} exceeds the {variance_budget:.3f} variance budget: "
            "too wide to call, rerun with more items or more trials"
        )
    elif lo >= threshold:
        verdict = GateVerdict.PASS
        reason = f"lower bound {lo:.3f} clears the {threshold:.3f} bar"
    else:
        verdict = GateVerdict.FAIL
        reason = (
            f"lower bound {lo:.3f} is below the {threshold:.3f} bar: "
            "the floor has not cleared, so the gate fails closed"
        )
    return GateDecision(
        verdict=verdict,
        reason=reason,
        lower_bound=lo,
        width=width,
        threshold=threshold,
        variance_budget=variance_budget,
    )


__all__ = ["GateDecision", "GateVerdict", "gate_on_lower_bound"]
