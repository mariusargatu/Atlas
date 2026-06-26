"""The planner: the first role in a harness of three agents.

The pattern for generating and grading at scale keeps three roles separate: a PLANNER that designs
the task, a GENERATOR that drives the agent under test, and a calibrated EVALUATOR that grades the
result. The separation is the point, not bureaucracy: an agent that plans its own work, does it,
and grades it is a student marking their own exam, and scoring it predictably well. In this harness
the generator is the Atlas graph (driven by the runner) and the evaluator is the grader stack. This
module holds the planner seam open.

This pass ships the TRIVIAL planner: it hands back a fixed, hand authored case set. A model driven
planner that designs novel tasks (and the rich case format it would emit) belongs to the
golden dataset article (04). Naming the role now keeps the three agent shape honest without
preempting that design.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from evals.evalkit.case import EvalCase


class Planner(Protocol):
    """The planner port. Adapters: ``StaticPlanner`` now, and a generative planner in 04."""

    def plan(self) -> tuple[EvalCase, ...]: ...


class StaticPlanner:
    """Returns a fixed, hand authored case set, unchanged every run. The simplest thing that keeps
    the planner a SEPARATE role from the generator and evaluator: the cases are designed up front
    by a human, and the planner just supplies them. Deterministic by construction, so it never
    introduces variance into the eval lane.
    """

    def __init__(self, cases: Sequence[EvalCase]) -> None:
        self._cases = tuple(cases)

    def plan(self) -> tuple[EvalCase, ...]:
        return self._cases


__all__ = ["Planner", "StaticPlanner"]
