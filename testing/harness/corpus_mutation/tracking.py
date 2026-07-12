"""The answer-tracks-mutated-truth assertion (SP8 task 7, D32): the one check that actually makes
this a MUTATION lane rather than Task 6's metamorphic one. Metamorphic asks: same truth, different
wording, does the answer stay the same? This asks the inverse: different truth (one registry fact
changed), same wording, does the answer CHANGE to track it? An agent that answers from parametric or
training knowledge, or a stale cache or index, instead of the freshly re-indexed retrieved context
fails this check by repeating the STALE (pre mutation) value instead of the new one.

Reuses `quality.agent_metrics.is_fact_grounded` as a LIBRARY, SP7's own reference based grounding
check (the same one `metamorphic.report.registry_answer_equivalence_holds` already reuses for Task
6): nothing here re-derives grounding logic.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from quality.agent_metrics import is_fact_grounded

from corpus_mutation.selection import FactMutation

__all__ = ["AnswerTrackingResult", "answer_tracks_mutated_truth"]


@dataclass(frozen=True)
class AnswerTrackingResult:
    """`tracks_new_truth`: does the answer ground the MUTATED (new) fact value, via SP7's own
    reference based grounding check. `repeats_stale_truth`: does the answer ALSO ground the OLD (pre
    mutation) value, the diagnostic that distinguishes "answered from parametric/training knowledge
    or a stale index" (stale value present, new value absent) from "simply wrong" (neither present)."""

    fact_ref: str
    new_value: object
    old_value: object
    tracks_new_truth: bool
    repeats_stale_truth: bool

    @property
    def holds(self) -> bool:
        """This lane's own pass/fail: the answer must track the new truth. Repeating the stale
        value is recorded for diagnosis only and is never part of `holds` on its own: an answer
        could, in principle, mention the old value in an explicit comparison ("the term used to be
        N months, it is now M") and still correctly track the new truth."""
        return self.tracks_new_truth


def answer_tracks_mutated_truth(
    mutation: FactMutation, response_text: str, tool_results: Sequence[object] = ()
) -> AnswerTrackingResult:
    """Dereferences `mutation.new_value` (and, for diagnosis, `mutation.old_value`) against the
    agent's answer through SP7's own `is_fact_grounded`, never a second, home rolled substring
    check. `tool_results` mirrors `is_fact_grounded`'s own signature: a correct answer's value may
    come from the response text itself or from a tool result the agent actually received."""
    new_fact = {"fact_id": mutation.fact_ref, "value": mutation.new_value}
    old_fact = {"fact_id": mutation.fact_ref, "value": mutation.old_value}
    return AnswerTrackingResult(
        fact_ref=mutation.fact_ref,
        new_value=mutation.new_value,
        old_value=mutation.old_value,
        tracks_new_truth=is_fact_grounded(new_fact, response_text, tool_results),
        repeats_stale_truth=is_fact_grounded(old_fact, response_text, tool_results),
    )
