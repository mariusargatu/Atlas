"""Acceptance test #1: the cold open. The north star and the v0.1 gate.

A legacy plan customer (Daniel) gets a confident, document grounded, but false answer about
his contract. Faithfulness (vs the retrieved current plan page) passes it green. The
account+catalog oracle catches it red. The gate is the oracle rule over a
frozen grounded but false answer fixture, so an improvement to Atlas
can never be misread as a regression.
"""
from __future__ import annotations

from atlas.domain.metrics import Answer, is_correct_vs_truth, is_faithful

# The frozen grounded but false answer: the agent claims the plan is contract free, grounded
# in the CURRENT plan page (which is genuinely term free).
DANIEL_ANSWER = Answer(
    text="Good news — your plan is contract-free, you can cancel any time with no fee.",
    claims={"has_contract": False},
    grounded_in={"has_contract": False},
)


def test_cold_open_is_faithful_but_false_for_the_legacy_customer():
    # Daniel is on the discontinued legacy plan, which DOES carry a 12 month term.
    assert is_faithful(DANIEL_ANSWER) is True                                # faithful to the page
    assert is_correct_vs_truth(DANIEL_ANSWER, "cust_legacy_term") is False   # caught by the oracle


def test_the_same_answer_is_correct_for_the_current_customer():
    # Sarah is on the current term free plan, so the identical claim is actually true.
    assert is_faithful(DANIEL_ANSWER) is True
    assert is_correct_vs_truth(DANIEL_ANSWER, "cust_current") is True


def test_acceptance_1_the_oracle_catches_what_faithfulness_misses():
    faithful = is_faithful(DANIEL_ANSWER)
    correct = is_correct_vs_truth(DANIEL_ANSWER, "cust_legacy_term")
    assert faithful and not correct, "cold-open must be faithful-but-false, caught by the oracle"
