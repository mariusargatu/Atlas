"""Aggregate case results into a report you read as a rate, not a single pass/fail.

Multi-trial sampling reports the RATE, never the verdict: a case that passes seven times in ten is
a known coin-flip, and the same case run once and passing is a landmine labelled safe. This report
carries the per-case and overall pass rates, JSON-serializable so a nightly run can append it to a
trend file. Turning a rate into a confidence interval, and gating on its lower bound, is the
statistics article's job (07); the harness stops at the honest rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from evals.evalkit.case import EvalCase
from evals.evalkit.graders import Grader
from evals.evalkit.runner import CaseResult, GraphBuild, run_case


@dataclass(frozen=True)
class EvalReport:
    cases: tuple[CaseResult, ...]

    @property
    def total_passes(self) -> int:
        return sum(c.passes for c in self.cases)

    @property
    def total_trials(self) -> int:
        return sum(c.k for c in self.cases)

    @property
    def overall_rate(self) -> float:
        return (self.total_passes / self.total_trials) if self.total_trials else 0.0

    def as_dict(self) -> dict:
        """A plain, JSON-serializable view, the row a nightly run appends to its trend file."""
        return {
            "overall": {
                "passes": self.total_passes,
                "trials": self.total_trials,
                "rate": self.overall_rate,
            },
            "cases": [
                {
                    "id": c.case_id,
                    "name": c.name,
                    "risk": c.risk,
                    "passes": c.passes,
                    "k": c.k,
                    "rate": c.rate,
                }
                for c in self.cases
            ],
        }

    def render(self) -> str:
        """A human-readable outcome table: one line per case, a verdict word a reviewer can act on.

        PASS (every trial held), FAIL (every trial broke), FLAKY (some did, some didn't, the
        live-model coin-flip a single run would hide). A red or flaky line carries the first failing
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
        header = f"eval: {self.total_passes}/{self.total_trials} trials safe across {len(self.cases)} case(s)"
        return header + "\n" + "\n".join(lines)


def build_report(results: tuple[CaseResult, ...] | list[CaseResult]) -> EvalReport:
    return EvalReport(cases=tuple(results))


def _grader_resolver(graders):
    """Turn the ``graders`` argument into a per-case lookup.

    A flat ``Sequence[Grader]`` is applied to every case (the simple uniform-suite path the tests
    use). A ``Mapping[str, Grader]`` is a registry resolved against each case's declared
    ``case.graders`` — so a mixed-risk suite (the demo) grades each case with only the rules it
    names, and ``EvalCase.graders`` is load-bearing rather than decorative.
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
) -> EvalReport:
    """Drive a whole case set and aggregate into one report, the eval lane's top-level call.

    Each case runs ``k`` trials through a fresh graph from ``build`` (the gateway mode lives in the
    caller's ``build``, so the same suite runs on REPLAY in the PR lane and on LIVE nightly). The
    graph is case-agnostic, identity and turns arrive at invoke time, so one ``build`` serves every
    case; cassettes are found by request key, so distinct cases hit distinct recordings in one dir.

    ``graders`` is either a flat sequence applied to every case, or a ``{name: Grader}`` registry
    resolved against each case's declared ``graders`` (see ``_grader_resolver``).
    """
    resolve = _grader_resolver(graders)
    results = [await run_case(case, build, resolve(case), k) for case in cases]
    return build_report(results)


__all__ = ["EvalReport", "build_report", "run_suite"]
