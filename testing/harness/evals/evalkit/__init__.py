"""The eval harness: the second of the two machines.

The regression lane (REPLAY) asks "did a pinned behaviour change?" — binary, gating, never
flickering. This package is the OTHER machine: the eval harness that asks "how good is the
agent, and is it getting better or worse?" It drives the agent over seeded cases, grades each
run with a layered grader stack, and reports a RATE, never a single verdict, because the live
model is stochastic and one sample lies.

Same runner, two gateway modes:
- LIVE (nightly): the model varies; multi-trial sampling measures the variance.
- REPLAY (PR lane): the model is pinned; the same runner proves its own wiring with zero egress.

Three roles kept separate (the three-agent harness): a planner designs the tasks, the generator
(the Atlas graph, driven by the runner) produces the runs, and the evaluator (the grader stack)
grades them. Named ``evalkit`` (not ``eval``) so it never shadows the builtin.
"""
from __future__ import annotations

from evals.evalkit.case import EvalCase
from evals.evalkit.graders import Composite, GradeContext, Grader, PredicateGrader, Verdict
from evals.evalkit.planner import Planner, StaticPlanner
from evals.evalkit.report import EvalReport, build_report, run_suite
from evals.evalkit.runner import CaseResult, TrialResult, run_case

__all__ = [
    "Composite",
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "GradeContext",
    "Grader",
    "Planner",
    "PredicateGrader",
    "StaticPlanner",
    "TrialResult",
    "Verdict",
    "build_report",
    "run_case",
    "run_suite",
]
