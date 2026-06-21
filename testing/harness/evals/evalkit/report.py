"""Aggregate case results into a report: a rate that always ships with its interval.

Every rate carries a Wilson interval, per case at the trial count and overall at the case count
(the k trials inside one case are correlated, so pooling them as independent trials would invent
precision the sample does not have). Each row is stamped with its provenance, the lane and model
id, so a replay interval never sits in a trend next to a live one with no way to tell them apart.
`gate()` reads only the interval's floor, never the point; `append_trend_row()` appends a row's
`as_dict()` to a committed JSONL trend file, and `read_trend_rows()` reads one back.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from evals.evalkit.case import EvalCase
from evals.evalkit.graders import Grader
from evals.evalkit.runner import CaseResult, GraphBuild, run_case
from quality.gate import GateDecision, gate_on_lower_bound
from quality.stats import wilson_interval, wilson_interval_from_rate


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
        """The 95% interval on the overall rate, computed at the case level, not the trial level.

        The independent unit is the case: the k trials within one case are correlated (on replay
        they are identical, the same cassette replayed), so pooling all trials into one binomial
        treats correlated data as independent and invents precision the sample does not have. The
        interval is a Wilson score interval evaluated at the overall rate with the effective sample
        size set to the case count, so it does not shrink as k grows and stays honest at the
        boundary (an all-safe run of a few cases reads as a wide interval, never a false [1.0, 1.0]).
        On replay each case is one deterministic observation, so this is exact; on live it is
        conservative, since it spends none of the within-case trials. Zero data returns (0.0, 1.0),
        never a false certainty."""
        n_cases = len(self.cases)
        if n_cases == 0 or self.total_trials == 0:
            return (0.0, 1.0)
        return wilson_interval_from_rate(self.overall_rate, n_cases)

    def gate(self, *, threshold: float, variance_budget: float) -> GateDecision:
        """Gate the tracked rate on the interval's floor, never the point.

        A 16/20 report has a 0.80 point and a ~0.58 floor: it must not clear a 0.75 bar,
        and an interval wider than the budget quarantines instead of passing as a coin flip.
        """
        return gate_on_lower_bound(
            self.overall_ci95, threshold=threshold, variance_budget=variance_budget
        )

    def as_dict(self) -> dict:
        """A plain dict serializable to JSON: one row for `append_trend_row()` to add to the trend file.

        Every rate travels with its interval, and the row is stamped with its provenance: the lane
        that produced it (replay or live) and the agent model id. Without that stamp, a degenerate
        replay interval (k identical replays of one cassette) would sit in the trend beside a real
        live row with no way to tell them apart. The overall interval is computed at the case level
        (see `overall_ci95`), so it does not shrink as k grows.
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
        grader's reason, so an SDET sees why without opening a trace.
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


# The committed trend file the eval entrypoint appends to and consumers (tests, a staleness workflow)
# read back. Defined next to the reader/writer so one public path is shared, not re-derived per caller.
# report.py lives in evalkit/, so this resolves to evalkit/artifacts/trend.jsonl.
TREND_PATH = Path(__file__).parent / "artifacts" / "trend.jsonl"


def append_trend_row(path: Path, row: Mapping) -> None:
    """Append one JSON row to a JSONL trend file, creating the parent directory if needed.

    One line, one JSON object, no rotation and no schema registry: the caller (an operator
    entrypoint) builds the row and stamps whatever provenance it needs (see `as_dict()`); this
    function only owns the file mechanics, so it stays clock free like the rest of this module.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def read_trend_rows(path: Path) -> list[dict]:
    """Read a JSONL trend file back into a list of row dicts, one per non blank line."""
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


__all__ = [
    "TREND_PATH",
    "EvalReport",
    "append_trend_row",
    "build_report",
    "read_trend_rows",
    "run_suite",
]
