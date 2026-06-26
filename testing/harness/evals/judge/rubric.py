"""Two rubrics, and the documented correction between them.

The kappa before/after study turns on a single, named change to the instrument. ``RUBRIC_V1`` is the
naive judge the cold open shipped: it scores HELPFULNESS and FLUENCY, so it rewards a confident,
well written answer and is blind to whether the answer is TRUE for the customer's account. That is
the judge that scored the grounded but false legacy plan answer a nine.

``RUBRIC_V2`` is the correction: it makes truth against the account the pass condition and says, in
words, that a fluent answer which contradicts the customer's plan terms FAILS. Same judge model,
same calibration set, one rubric edit. The contract (``contract.py``) treats that edit as a new
instrument, which is the whole point: the agreement number has to be earned again.

The prompt template is deliberately tiny and identical in shape across rubrics (a system message
carrying the rubric, a user message carrying the case, a one word PASS/FAIL verdict), so the only
thing that moves between V1 and V2 is the rubric text, and ``template_hash`` proves it.
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


# V1, the lying judge: fluency and helpfulness, truth blind. Rewards the confident, well written
# answer the cold open shipped, regardless of whether it is true for this customer.
RUBRIC_V1 = Rubric(
    version="naive-helpfulness-v1",
    system=(
        "You score a customer-support answer for HELPFULNESS and FLUENCY. "
        "Reply PASS if the answer is clear, confident, complete, and well written. "
        "Reply FAIL only if it is vague, evasive, or hard to read. "
        "Answer with one word: PASS or FAIL."
    ),
)

# V2, the correction: truth against the account is the pass condition. A fluent answer that
# contradicts the customer's plan terms FAILS, which is exactly the case V1 waved through.
RUBRIC_V2 = Rubric(
    version="account-truth-v2",
    system=(
        "You score a customer-support answer for CORRECTNESS against the customer's own account and "
        "the catalog, not for how well it is written. Reply PASS only if every claim is TRUE for "
        "THIS customer's plan and record. A confident, fluent, well-grounded answer that "
        "contradicts the customer's actual plan terms (a term, a fee, a data cap) must FAIL. "
        "Answer with one word: PASS or FAIL."
    ),
)


def prompt(rubric: Rubric, question: str, answer: str) -> list[BaseMessage]:
    """The judge prompt: the rubric as system, the case as user, a one word verdict expected back."""
    return [
        SystemMessage(rubric.system),
        HumanMessage(f"Question: {question}\nAnswer under review: {answer}\nVerdict (PASS or FAIL):"),
    ]


def compare_prompt(rubric: Rubric, question: str, answer_a: str, answer_b: str) -> list[BaseMessage]:
    """The pairwise judge prompt for the order swap probe: the rubric plus two answers, an A/B verdict
    expected back. One definition shared by the judge and every cassette seeder, so the replayed
    request can never drift from the seeded one (a drift would be a hard cassette miss)."""
    return [
        SystemMessage(rubric.system + " You are comparing two answers; reply with only A or B."),
        HumanMessage(f"Question: {question}\nAnswer A: {answer_a}\nAnswer B: {answer_b}\nBetter answer (A or B):"),
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
            "shape": "system(rubric)+user(question,answer)->PASS|FAIL",
        }
    )


__all__ = ["RUBRIC_V1", "RUBRIC_V2", "Rubric", "compare_prompt", "prompt", "template_hash"]
