"""The groundedness rubric: does every claim in the answer trace back to what was actually
retrieved, or did the model say something the cited context never supports.

D15's judge is ONE binary check, reference free: not helpfulness, not style, not truth against some
external oracle. It reads the cited retrieved context (the chunks the agent actually cited, plus
their registry ``entity_ids`` where the caller supplies them) and the answer under review, and asks
whether every factual claim in the answer is entailed by that context. An answer that abstains
(admits it does not know, hands off to a person) PASSES: it makes no unsupported claim. An answer
that states anything the cited context does not support FAILS, even when the claim happens to be
true by coincidence -- groundedness measures whether the ANSWER is anchored to what was retrieved,
not whether it is right by some other test SP7's reference based metrics already own.

The scaffolding (``Rubric``, ``template_hash``, ``prompt``/``compare_prompt``'s shape) is absorbed
from the pre rewrite `evals/judge/rubric.py`; the CONTENT is fresh, per the planning digest's own
disposition ("absorb the scaffolding, discard the content... SP8 authors a fresh rubric text against
retrieved context"). The prompt still asks for a bare PASS/FAIL token (`llm_judge._parse_label`'s own
parsing contract, unchanged): that is the judge's OWN prompt vocabulary, deliberately independent of
the frozen trace contract's wire vocabulary (`grounded`/`ungrounded`), translated at the trace
boundary by `llm_judge.translate_verdict`, never by this module.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from determinism.canonical import digest


@dataclass(frozen=True)
class Rubric:
    """A versioned scoring instruction. ``version`` rides into the judge contract."""

    version: str
    system: str


# The one rubric SP8 ships (D15: "one judge (binary groundedness)"). Grades the answer against the
# CITED retrieved context only, never against account state or general helpfulness -- SP7 owns
# reference based correctness, and the pre rewrite lane's own RUBRIC_V1 (helpfulness)/RUBRIC_V2
# (account truth) are both out of scope here, discarded per the digest's own disposition.
RUBRIC_GROUNDEDNESS = Rubric(
    version="groundedness-v1",
    system=(
        "You score a customer-support answer for GROUNDEDNESS against its cited retrieved context, "
        "never for style, tone, or general helpfulness. Reply PASS only if every factual claim in "
        "the answer is entailed by the cited context (the retrieved chunks and, where given, their "
        "registry entity_ids). An answer that abstains, admits it does not know, or hands off to a "
        "person PASSES: it makes no claim the context does not support. Reply FAIL if the answer "
        "states anything the cited context does not support, even if the claim happens to be true. "
        "Answer with one word: PASS or FAIL."
    ),
)

# SP10 task 4's Simulator lane (`judge.simulator_lane`): the SAME `prompt()`/`judge_label` shape
# above, reused unchanged, just re purposed -- `question` carries the persona spec (never a
# retrieved context question), `answer` carries the transcript text under review, `context` stays
# empty (there is no cited passage concept for a persona conversation). Two rubrics, not one,
# because they grade two genuinely different things: whether the SIMULATOR stayed in character
# (a question about the quality of the simulation, never about the agent under test) versus
# whether the AGENT served the customer's goal (a question about the quality of the agent).
# Conflating them into one rubric would make a broken simulator and a broken agent
# indistinguishable in the same verdict.
RUBRIC_PERSONA_ADHERENCE = Rubric(
    version="persona-adherence-v1",
    system=(
        "You score whether a simulated customer's turns in a support transcript stayed IN CHARACTER "
        "for a stated persona (disposition and goal), never whether the support agent handled the "
        "conversation well. Reply PASS only if every customer turn is consistent with the stated "
        "disposition and goal, in voice and behaviour, with no turn that reads like a different "
        "persona or breaks character (for example, an impatient persona suddenly being endlessly "
        "patient, or a confused persona suddenly naming the exact plan by its catalog name). Reply "
        "FAIL if any customer turn drifts out of character. Answer with one word: PASS or FAIL."
    ),
)

RUBRIC_TASK_SUCCESS = Rubric(
    version="task-success-v1",
    system=(
        "You score whether a support agent's handling of a whole conversation reasonably served the "
        "customer's stated goal, given the full transcript and the account action(s) the agent "
        "actually executed. Reply PASS if the agent's final handling reasonably serves the goal, or "
        "safely and correctly declines/hands off when the goal cannot or should not be met, and any "
        "executed action matches what the customer actually settled on by the end of the "
        "conversation (never one she raised and then walked back). Reply FAIL if the agent took no "
        "action a settled goal required, took an action inconsistent with what the customer settled "
        "on, or otherwise failed to serve a reasonably achievable goal. Answer with one word: PASS "
        "or FAIL."
    ),
)


def prompt(rubric: Rubric, question: str, answer: str, context: str) -> list[BaseMessage]:
    """The judge prompt: the rubric as system, the question plus cited context plus the answer under
    review as user, a one word verdict expected back. ``context`` is the caller's own rendering of
    the cited retrieved chunks (plus registry entity_ids where relevant); this module has no opinion
    on how that string is built, only that it is what the judge is asked to ground the answer against."""
    return [
        SystemMessage(rubric.system),
        HumanMessage(
            f"Question: {question}\nCited context: {context}\nAnswer under review: {answer}\n"
            "Verdict (PASS or FAIL):"
        ),
    ]


def compare_prompt(rubric: Rubric, question: str, answer_a: str, answer_b: str, context: str) -> list[BaseMessage]:
    """The pairwise judge prompt for the order swap probe: the rubric plus the cited context plus
    two answers, an A/B verdict expected back. One definition shared by the judge and every cassette
    seeder, so the replayed request can never drift from the seeded one (a drift would be a hard
    cassette miss)."""
    return [
        SystemMessage(rubric.system + " You are comparing two answers; reply with only A or B."),
        HumanMessage(
            f"Question: {question}\nCited context: {context}\nAnswer A: {answer_a}\nAnswer B: {answer_b}\n"
            "Better answer (A or B):"
        ),
    ]


def template_hash(rubric: Rubric) -> str:
    """A digest over the rubric text and the template shape, the third field of the judge contract.

    Folds in the fixed scaffolding string so a change to the template (not just the rubric) also
    moves the hash, which is the contract's promise: the instrument is the whole prompt, not the
    rubric alone.
    """
    return digest(
        {
            "version": rubric.version,
            "system": rubric.system,
            "shape": "system(rubric)+user(question,context,answer)->PASS|FAIL",
        }
    )


__all__ = [
    "RUBRIC_GROUNDEDNESS",
    "RUBRIC_PERSONA_ADHERENCE",
    "RUBRIC_TASK_SUCCESS",
    "Rubric",
    "compare_prompt",
    "prompt",
    "template_hash",
]
