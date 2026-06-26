"""The grader stack: grade with the cheapest, strictest tool that can do the job.

The harness owns the stack shape, an ordered set of graders run cheapest first, short circuiting at
the first hard fail, so an expensive grader never runs once a cheaper one has already failed the
run. The concrete graders are deliberately not here: rules over an oracle, programmatic value
checks, and an LLM judge each carry their own domain (faithfulness/correctness, the judge and its
calibration) and live in their own modules. This module ships only the machinery plus one
trivial, domain free grader, so the stack is demonstrable without preempting that work.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from tracing import Span


@dataclass(frozen=True)
class Verdict:
    """One grader's reading of one run."""

    grader: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class GradeContext:
    """What every grader reads: the run's outcome and the trace it emitted.

    ``final_response`` is the text the agent would have shipped. ``trace`` is the turn's span tree
    (read only), the eval harness's substrate. Concrete graders add the oracle/judge fields a
    real grader needs. A grader takes only what it reads, so new fields never break existing ones.
    """

    customer_id: str
    final_response: str
    trace: tuple[Span, ...]
    expected_doc_ids: tuple[str, ...] = ()
    expected_tool_calls: tuple[Mapping[str, object], ...] = ()
    retrieved_doc_ids: tuple[str, ...] = ()


class Grader(Protocol):
    """The grader port. The harness provides the slot, and concrete graders fill it as they are built.

    A ``TrajectoryGrader`` that scores the tool order on ``ctx.trace`` against a case's expected
    trajectory belongs with the golden dataset layer, which owns that field. See ``case.py``.
    """

    name: str

    def grade(self, ctx: GradeContext) -> Verdict: ...


class PredicateGrader:
    """The trivial grader: wraps any predicate over the run into a ``Verdict``. Generic harness
    machinery with NO domain knowledge baked in, so it shows the grader stack slot and the
    short circuit behaviour without standing in for the real graders. The cheapest, strictest check
    that can decide a run, expressed as a plain callable, plugs in exactly here.
    """

    def __init__(self, name: str, predicate: Callable[[GradeContext], bool], *, reason: str = "") -> None:
        self.name = name
        self._predicate = predicate
        self._reason = reason

    def grade(self, ctx: GradeContext) -> Verdict:
        ok = bool(self._predicate(ctx))
        return Verdict(self.name, passed=ok, reason=self._reason or ("ok" if ok else "predicate failed"))


class Composite:
    """Run graders cheapest first, and stop at the first hard fail (short circuit)."""

    def __init__(self, graders: Sequence[Grader]) -> None:
        self._graders = tuple(graders)

    def grade(self, ctx: GradeContext) -> tuple[Verdict, ...]:
        verdicts: list[Verdict] = []
        for grader in self._graders:
            verdict = grader.grade(ctx)
            verdicts.append(verdict)
            if not verdict.passed:
                break  # an expensive grader never runs once a cheaper one has failed the run
        return tuple(verdicts)


__all__ = ["Composite", "GradeContext", "Grader", "PredicateGrader", "Verdict"]
