"""Aggregate case results into a report you read as a rate WITH its interval.

Sampling many trials reports the RATE, never the verdict: a case that passes seven times in ten is
a known coin flip, and the same case run once and passing is a landmine labelled safe. This report
carries the per case and overall pass rates, serializable to JSON so a nightly run can append it to a
trend file, and it is held to the statistics article's reporting law (07): a metric ships with its
uncertainty or it does not ship. Every rate carries its interval (per case: Wilson; the OVERALL: a
case-level cluster bootstrap, because the k trials within a case are correlated and pooling them as
iid would invent precision, so the overall interval does not shrink as k grows), the row is stamped
with its provenance (lane + model, so a replay interval never masquerades as a live one), and where
a release turns on the tracked rate, `gate()` reads the interval's floor, never the point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from evals.evalkit.case import EvalCase
from evals.evalkit.graders import Grader
from evals.evalkit.runner import CaseResult, GraphBuild, run_case
from evals.gate import GateDecision, gate_on_lower_bound
from evals.stats import wilson_interval, wilson_interval_from_rate


@dataclass(frozen=True)
class EvalReport:
    cases: tuple[CaseResult, ...]
    lane: str = ""                # provenance: "replay" | "live", which machinery produced this row
    model_id: str = ""            # provenance: the agent model id the trials ran against

    @property
    def total_passes(self) -> int:
        return sum(c.passes for c in self.cases)

    @property
    def total_trials(self) -> int:
        return sum(c.k for c in self.cases)

    @property
    def overall_rate(self) -> float:
        return (self.total_passes / self.total_trials) if self.total_trials else 0.0

    @property
    def overall_ci95(self) -> tuple[float, float]:
        """The honest 95% interval on the overall rate, at the CASE level.

        The independent unit is the case, not the trial. The k trials within one case are correlated
        (on REPLAY they are identical, the same cassette replayed), so pooling all C*k trials into one
        binomial treats correlated data as independent and invents precision the sample does not have.
        So the interval is a Wilson score interval at the EFFECTIVE sample size = the number of cases,
        evaluated at the overall rate. Wilson (a score interval, not a variance estimate) stays honest
        at the boundary, so an all-safe run of a few cases reads as a wide interval, never a false
        [1.0, 1.0], and the interval does not shrink as k grows. On REPLAY each case is one
        deterministic observation, so cases are exactly the independent unit; on LIVE this is
        conservative (it spends none of the within-case trials, so it never over-claims) and the
        proper within/between split is the statistics article's GLMM. Zero data is honestly (0, 1)."""
        n_cases = len(self.cases)
        if n_cases == 0 or self.total_trials == 0:
            return (0.0, 1.0)
        return wilson_interval_from_rate(self.overall_rate, n_cases)

    def gate(self, *, threshold: float, variance_budget: float) -> GateDecision:
        """Gate the tracked rate on the interval's floor, never the point (the 07 rule).

        A 16/20 report has a 0.80 point and a ~0.58 floor: it must not clear a 0.75 bar,
        and an interval wider than the budget quarantines instead of passing as a coin flip.
        """
        return gate_on_lower_bound(
            self.overall_ci95, threshold=threshold, variance_budget=variance_budget
        )

    def as_dict(self) -> dict:
        """A plain view serializable to JSON, the row a nightly run appends to its trend file.

        Every rate travels with its interval, and the row is STAMPED with its provenance: the lane
        that produced it (replay/live) and the agent model id. Without the lane stamp a degenerate
        REPLAY interval (k identical replays of one cassette) would sit in a trend beside real LIVE
        rows with no way to tell them apart; a number without provenance is weather, not a
        measurement. The overall interval is computed at the case level (see `overall_ci95`), so it
        does not shrink as k grows.
        """
        return {
            "provenance": {"lane": self.lane, "model_id": self.model_id},
            "overall": {
                "passes": self.total_passes,
                "trials": self.total_trials,
                "rate": self.overall_rate,
                "ci95": list(self.overall_ci95),
            },
            "cases": [
                {
                    "id": c.case_id,
                    "name": c.name,
                    "risk": c.risk,
                    "passes": c.passes,
                    "k": c.k,
                    "rate": c.rate,
                    "ci95": list(wilson_interval(c.passes, c.k)),
                }
                for c in self.cases
            ],
        }

    def render(self) -> str:
        """A human readable outcome table: one line per case, a verdict word a reviewer can act on.

        PASS (every trial held), FAIL (every trial broke), FLAKY (some did, some didn't, the
        live model coin flip a single run would hide). A red or flaky line carries the first failing
        grader's reason, so an SDET sees WHY without opening a trace.
        """
        lines = []
        for c in self.cases:
            flag = "PASS" if c.passes == c.k else "FAIL" if c.passes == 0 else "FLAKY"
            title = c.name or c.case_id
            line = f"  {flag:5} {(c.risk or '-'):22} {c.passes}/{c.k}  {title}"
            if c.passes < c.k:
                line += f"\n        ↳ {c.first_failure_reason()}"
            lines.append(line)
        lo, hi = self.overall_ci95
        header = (
            f"eval: {self.total_passes}/{self.total_trials} trials safe across "
            f"{len(self.cases)} case(s) · Wilson 95% CI [{lo:.3f}, {hi:.3f}]"
        )
        return header + "\n" + "\n".join(lines)


def build_report(
    results: tuple[CaseResult, ...] | list[CaseResult],
    *,
    lane: str = "",
    model_id: str = "",
) -> EvalReport:
    return EvalReport(cases=tuple(results), lane=lane, model_id=model_id)


def _grader_resolver(graders):
    """Turn the ``graders`` argument into a per case lookup.

    A flat ``Sequence[Grader]`` is applied to every case (the simple uniform suite path the tests
    use). A ``Mapping[str, Grader]`` is a registry resolved against each case's declared
    ``case.graders``, so a mixed risk suite (the demo) grades each case with only the rules it
    names, and ``EvalCase.graders`` is load bearing rather than decorative.
    """
    if isinstance(graders, Mapping):
        return lambda case: [graders[name] for name in case.graders]
    fixed = tuple(graders)
    return lambda case: fixed


async def run_suite(
    cases: Sequence[EvalCase],
    build: GraphBuild,
    graders: Sequence[Grader] | Mapping[str, Grader],
    k: int = 1,
    *,
    lane: str = "",
    model_id: str = "",
) -> EvalReport:
    """Drive a whole case set and aggregate into one report, the eval lane's top level call.

    Each case runs ``k`` trials through a fresh graph from ``build`` (the gateway mode lives in the
    caller's ``build``, so the same suite runs on REPLAY in the PR lane and on LIVE nightly). The
    graph is case agnostic, identity and turns arrive at invoke time, so one ``build`` serves every
    case. Cassettes are found by request key, so distinct cases hit distinct recordings in one dir.

    ``graders`` is either a flat sequence applied to every case, or a ``{name: Grader}`` registry
    resolved against each case's declared ``graders`` (see ``_grader_resolver``).
    """
    resolve = _grader_resolver(graders)
    results = [await run_case(case, build, resolve(case), k) for case in cases]
    return build_report(results, lane=lane, model_id=model_id)


__all__ = ["EvalReport", "build_report", "run_suite"]
