"""The LLM judge, routed through the model gateway (ADR-007).

The judge is itself a model call, so it goes through the SAME record/replay gateway as the agent
under test. Two consequences fall out, and both are the point. First, the grading is as reproducible
as the thing it grades: a model grading a model, both on tape, so rerunning last month's suite
returns last month's scores exactly. Second, the judge can be a DIFFERENT model family than the
agent (Atlas runs on Claude, the judge on a GPT-class model) without any new machinery, because the
gateway does not care which provider sits behind a ``model_id`` once a cassette exists. Cross family
judging is the cheapest defence against self-enhancement bias, and here it costs one ``model_id``.

The judge answers a binary question (PASS / FAIL) because the calibration set is human labelled the
same way. A one word verdict is the smallest contract that a rule can parse without a second judge.
"""
from __future__ import annotations

import re

from langchain_core.messages import BaseMessage

from evals.evalkit.graders import GradeContext, Verdict
from evals.judge.rubric import Rubric, compare_prompt, prompt
from tracing import spans_of_kind

# A compliant pairwise verdict is a bare "A" or "B". This also survives a verbose live reply by taking
# the first standalone A/B token, so "Answer B is better" reads as B, not A (the word "Answer" is not
# a standalone "A"). Anything with no clear token falls through to B at the call site.
_AB = re.compile(r"\b([AB])\b")


def _parse_label(text: str) -> int:
    """Map a one word verdict to 1 (PASS) / 0 (FAIL). Anything that is not a clear PASS fails closed,
    because an unparseable judge verdict is not evidence the answer was good."""
    head = text.strip().upper()
    return 1 if head.startswith("PASS") else 0


def judge_label(gateway, rubric: Rubric, question: str, answer: str) -> int:
    """Run one answer past the judge and return its binary label, through the gateway (REPLAY in CI).

    ``gateway`` is a ``GatewayChatModel``. In the PR lane it is pinned to REPLAY and a missing
    cassette is a hard fail, so the judge never reaches the network. The judge's ``model_id`` is the
    judge contract's first field, deliberately a different family than the agent's.
    """
    messages: list[BaseMessage] = prompt(rubric, question, answer)
    reply = gateway.invoke(messages)
    return _parse_label(getattr(reply, "content", "") or "")


def order_swap(gateway, rubric: Rubric, question: str, answer_a: str, answer_b: str) -> tuple[int, int]:
    """Score a pair both ways for position bias testing: judge (A,B) then (B,A).

    Returns ``(winner_first, winner_second)`` as 0=A, 1=B. A consistent judge picks the same answer
    regardless of order. A flip between the two readings is a reading order artifact, not a
    preference, which ``calibration.order_swap_flip_rate`` then quantifies.
    """
    ab = _winner(gateway, rubric, question, answer_a, answer_b)
    ba_raw = _winner(gateway, rubric, question, answer_b, answer_a)
    # ba_raw is in the swapped frame (0 means "the one shown first", which is B). Map back to A/B.
    ba = 1 - ba_raw
    return ab, ba


def _winner(gateway, rubric: Rubric, question: str, first: str, second: str) -> int:
    """Ask which of two answers is better. 0 = the first shown, 1 = the second shown.

    The first standalone A/B token decides, so a verbose verdict ("Answer B is better") is read as B,
    not misparsed as A by a leading letter test. No clear token falls through to 1 (the second)."""
    reply = gateway.invoke(compare_prompt(rubric, question, first, second))
    match = _AB.search((getattr(reply, "content", "") or "").upper())
    return 0 if (match and match.group(1) == "A") else 1


class LlmJudgeGrader:
    """A judge that grades a driven run as a ``Grader`` in the eval stack, for the subjective lane.

    It reads the question from the turn span (so a case does not have to thread it through) and the
    shipped reply from ``ctx.final_response``, and returns a ``Verdict``. It sits LAST in a
    ``Composite`` stack, after the cheap rules, so the expensive model call never runs once a rule
    has already failed the run (the grader stack short circuits). High consequence checks stay on
    rules. This is for "is the explanation actually helpful", where no rule can be written.
    """

    def __init__(self, gateway, rubric: Rubric, *, name: str = "llm-judge") -> None:
        self.name = name
        self._gateway = gateway
        self._rubric = rubric

    def grade(self, ctx: GradeContext) -> Verdict:
        turns = spans_of_kind(ctx.trace, "turn")
        question = str(turns[0].attributes.get("input", "")) if turns else ""
        label = judge_label(self._gateway, self._rubric, question, ctx.final_response)
        return Verdict(self.name, passed=bool(label), reason="judge: PASS" if label else "judge: FAIL")


__all__ = ["LlmJudgeGrader", "judge_label", "order_swap"]
