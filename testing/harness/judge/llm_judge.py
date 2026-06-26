"""The LLM judge, routed through the model gateway (ADR-007), plus the trace boundary translation.

A model call through the same record/replay gateway as the agent under test: grading is as
reproducible as the thing it grades, and the judge can be a different model family (Atlas runs on
Claude, the judge on a model from the GPT family, D15's cross family default) with no new machinery, since the
gateway does not care which provider sits behind a ``model_id`` once a cassette exists.

The judge answers PASS/FAIL, its OWN prompt vocabulary (``rubric.RUBRIC_GROUNDEDNESS``'s own
contract); a one word verdict is the smallest contract a rule can parse without a second judge.
``translate_verdict`` is the ONE place that vocabulary crosses into the frozen trace contract's wire
vocabulary (``grounded``/``ungrounded``, ``contracts/trace/schema.json``'s own pinned enum),
deliberately a separate step from parsing, mirroring how ``_parse_label`` already isolates "what did
the model say" from "what label does that mean."

Absorbed from the pre rewrite `evals/judge/llm_judge.py`'s parsing mechanics (`judge_label`/
`order_swap`/`_parse_label`, per the digest: "first standalone token wins, unparseable fails
closed"); the old `LlmJudgeGrader` (the retired `evalkit` `Composite`/`GradeContext` wiring) is not
carried forward, since D29 runs the judge as a batch teardown stage reading traces and dataset cases
directly, never as an `evalkit` `Grader`.
"""
from __future__ import annotations

import re
import string

from langchain_core.messages import BaseMessage

from judge.rubric import Rubric, compare_prompt, prompt

# Punctuation trimmed off the ends of the first token before comparing it to PASS/FAIL, so a
# trailing period or a leading quote does not defeat an otherwise clean verdict (`_parse_label`'s own
# "PASS." example). Trimmed only from the ends, never the middle, so a token that glues extra text
# onto PASS through INTERNAL punctuation still fails the exact match, the point of this whole guard.
_TOKEN_PUNCTUATION = string.punctuation

# A compliant pairwise verdict is a bare "A" or "B". This also survives a verbose live reply by taking
# the first standalone A/B token, so "Answer B is better" reads as B, not A (the word "Answer" is not
# a standalone "A"). Anything with no clear token falls through to B at the call site.
_AB = re.compile(r"\b([AB])\b")

# The trace boundary's own pinned vocabulary (`contracts/trace/schema.json`'s `atlas.judge.verdict`
# enum) -- named here as constants so nothing downstream ever hand types the literal strings.
VERDICT_GROUNDED = "grounded"
VERDICT_UNGROUNDED = "ungrounded"


def _parse_label(text: str) -> int:
    """Map a one word verdict to 1 (PASS) / 0 (FAIL). The FIRST STANDALONE token decides (the same
    design `_winner`'s `_AB` regex already uses for the pairwise verdict): the first whitespace
    delimited token, stripped of surrounding punctuation, must be the exact word PASS. Anything else,
    including a longer word that merely starts with those four letters (PASSABLE, or PASS glued to
    trailing text through internal punctuation), fails closed, because a prefix match is not evidence
    the model actually returned the PASS token, and an unparseable judge verdict is not evidence the
    answer was grounded either."""
    head = text.strip().upper()
    if not head:
        return 0
    first_token = head.split(None, 1)[0].strip(_TOKEN_PUNCTUATION)
    return 1 if first_token == "PASS" else 0


def translate_verdict(label: int) -> str:
    """The trace boundary translation: the judge's own binary label (1/0, `_parse_label`'s own
    output) becomes the frozen wire vocabulary (`grounded`/`ungrounded`), independent of whatever
    prompt vocabulary (PASS/FAIL) produced it. The one function every trace emitter of
    `atlas.judge.verdict` must route through, never a hand typed string at the call site."""
    return VERDICT_GROUNDED if label else VERDICT_UNGROUNDED


def judge_label(gateway, rubric: Rubric, question: str, answer: str, context: str) -> int:
    """Run one answer past the judge and return its binary label, through the gateway (REPLAY in CI).

    ``gateway`` is a ``GatewayChatModel``. In the PR lane it is pinned to REPLAY and a missing
    cassette is a hard fail, so the judge never reaches the network. The judge's ``model_id`` is the
    judge contract's first field, deliberately a different family than the agent's. ``context`` is
    the cited retrieved content (plus registry entity_ids where relevant) the rubric grades against.
    """
    messages: list[BaseMessage] = prompt(rubric, question, answer, context)
    reply = gateway.invoke(messages)
    return _parse_label(getattr(reply, "content", "") or "")


def order_swap(
    gateway, rubric: Rubric, question: str, answer_a: str, answer_b: str, context: str,
) -> tuple[int, int]:
    """Score a pair both ways for position bias testing: judge (A,B) then (B,A).

    Returns ``(winner_first, winner_second)`` as 0=A, 1=B. A consistent judge picks the same answer
    regardless of order. A flip between the two readings is a reading order artifact, not a
    preference.
    """
    ab = _winner(gateway, rubric, question, answer_a, answer_b, context)
    ba_raw = _winner(gateway, rubric, question, answer_b, answer_a, context)
    # ba_raw is in the swapped frame (0 means "the one shown first", which is B). Map back to A/B.
    ba = 1 - ba_raw
    return ab, ba


def _winner(gateway, rubric: Rubric, question: str, first: str, second: str, context: str) -> int:
    """Ask which of two answers is better. 0 = the first shown, 1 = the second shown.

    The first standalone A/B token decides, so a verbose verdict ("Answer B is better") is read as B,
    not misparsed as A by a leading letter test. No clear token falls through to 1 (the second)."""
    reply = gateway.invoke(compare_prompt(rubric, question, first, second, context))
    match = _AB.search((getattr(reply, "content", "") or "").upper())
    return 0 if (match and match.group(1) == "A") else 1


__all__ = [
    "VERDICT_GROUNDED",
    "VERDICT_UNGROUNDED",
    "judge_label",
    "order_swap",
    "translate_verdict",
]
