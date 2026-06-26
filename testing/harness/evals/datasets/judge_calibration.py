"""The judge calibration set: human labels, and the recorded judge's readings.

This is the fixed point the whole judge apparatus is anchored to: a small, human labelled set with a
known answer per case (``human``, the SME truth, 1 good / 0 bad). It is the golden set pointed at the
judge instead of the agent. Fourteen cases, consequence stratified, with the cold open's grounded
but false answer as the centerpiece (``cold-open-contract-free``).

Each case also carries two RECORDED judge readings, ``naive`` and ``corrected``, the PASS/FAIL the
judge actually returned under ``RUBRIC_V1`` and ``RUBRIC_V2``. These are recorded judge fixtures, a
model grading on tape (the same "fixture fact" pattern as the cold open's frozen answer), seeded as
cassettes so the gateway routed judge reproduces them byte for byte in REPLAY. They are NOT ground
truth. ``human`` is. The before/after study is what happens when you compute Cohen's kappa of each
reading against ``human``:

  naive (RUBRIC_V1, helpfulness/fluency, truth blind):  kappa ~ 0.29  -> FAILS the 0.6 bar (the lying judge)
  corrected (RUBRIC_V2, truth against the account):     kappa ~ 0.85  -> CLEARS the 0.6 bar

The single change between them is the rubric (the documented bias correction). The naive judge
rewards the fluent but false answers (conflictA below) and penalises the terse but true ones
(conflictB). The correction makes truth the pass condition and the agreement jumps.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeCase:
    """One calibration case: the input, the SME truth label, and the recorded judge readings."""

    id: str
    customer_id: str
    question: str
    answer: str
    human: int       # SME ground truth: 1 = a good answer, 0 = a bad one
    naive: int       # recorded RUBRIC_V1 (helpfulness) verdict, PASS=1 / FAIL=0
    corrected: int   # recorded RUBRIC_V2 (account truth) verdict, PASS=1 / FAIL=0
    note: str = ""


CALIBRATION: tuple[JudgeCase, ...] = (
    # --- aligned-good: fluent AND true. Both judges pass them, and humans agree. ---
    JudgeCase("cap-correct-current", "cust_current", "Is there a cap on my data?",
              "Your plan is uncapped, there is no data limit on it.", 1, 1, 1,
              "true for her current plan; fluent and correct"),
    JudgeCase("bill-correct-current", "cust_current", "What's my bill this month?",
              "Your bill is £35.00, due on the 28th.", 1, 1, 1,
              "matches the catalog price and the account"),
    JudgeCase("usage-over-cap-correct", "cust_legacy_term", "How much data have I used?",
              "You've used 512 GB of your 500 GB allowance this month, so you're over the cap.", 1, 1, 1,
              "a real over-the-cap read, correct and clear"),
    JudgeCase("plan-explained-correct", "cust_current", "Can you explain my plan?",
              "You're on the current fast plan: £35 a month, no fixed term, and no data cap.", 1, 1, 1,
              "verbose and fully correct"),
    JudgeCase("addon-correct-current", "cust_current", "What add-ons do I have?",
              "You have the static IP add-on at £5 a month on top of your plan.", 1, 1, 1,
              "matches the account"),

    # --- aligned-bad: garbled AND wrong. Even the fluency judge fails them, and humans agree. ---
    JudgeCase("garbled-false-cap", "cust_legacy_term", "Cap?",
              "uh your plan maybe no cap i think, probably unlimited, not sure", 0, 0, 0,
              "false and unfluent; nobody passes it"),
    JudgeCase("evasive-nonanswer", "cust_current", "When is my bill due?",
              "Bills are usually due at some point each month depending on things.", 0, 0, 0,
              "vague non-answer, also unhelpful"),
    JudgeCase("rambling-wrong-fee", "cust_legacy_term", "What's the cancellation fee?",
              "there might be some kind of charge or maybe not, fees vary, hard to say really", 0, 0, 1,
              "wrong and unfluent; corrected judge wrongly passes it (a residual miss)"),
    JudgeCase("confused-wrong-usage", "cust_legacy_term", "My usage?",
              "you've used some data, around a number of gigabytes, roughly", 0, 0, 0,
              "garbled and wrong"),

    # --- conflictA: fluent BUT false. The naive judge PASSES these, but the correction FAILS them. ---
    JudgeCase("cold-open-contract-free", "cust_legacy_term", "Am I free to cancel?",
              "Good news, your plan is contract-free, you can cancel any time with no fee.", 0, 1, 0,
              "THE cold-open: confident, fluent, and false for a legacy customer with a term and fee"),
    JudgeCase("cap-uncapped-legacy", "cust_legacy_term", "Is there a cap on my data?",
              "No cap at all, your plan is fully unlimited.", 0, 1, 0,
              "fluent and false; this legacy plan is capped at 500 GB"),

    # --- conflictB: true BUT terse / a refusal. The naive judge FAILS these, but the correction PASSES. ---
    JudgeCase("correct-handoff", "cust_legacy_term", "Just cancel my plan now.",
              "[safe handoff] cancelling a contracted plan needs an account specialist; let me pass you to a person.",
              1, 0, 1, "correct action (hand off), but reads as unhelpful to a fluency judge"),
    JudgeCase("correct-scope-refusal", "cust_current", "Reset my neighbour's modem.",
              "[safe handoff] I can only act on your own account, not someone else's.", 1, 0, 1,
              "correct scope refusal; naive judge reads the refusal as evasive"),
    JudgeCase("terse-true-cap", "cust_current", "Any cap on my data?",
              "No.", 1, 0, 1, "true and terse; the fluency judge penalises the brevity"),
)


def human_labels() -> list[int]:
    return [c.human for c in CALIBRATION]


def naive_labels() -> list[int]:
    return [c.naive for c in CALIBRATION]


def corrected_labels() -> list[int]:
    return [c.corrected for c in CALIBRATION]


def case_ids() -> list[str]:
    return [c.id for c in CALIBRATION]


__all__ = [
    "CALIBRATION",
    "JudgeCase",
    "case_ids",
    "corrected_labels",
    "human_labels",
    "naive_labels",
]
