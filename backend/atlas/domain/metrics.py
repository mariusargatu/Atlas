"""The two metrics the cold open turns on: faithfulness vs correctness vs truth.

Faithfulness asks whether the answer agrees with the document it retrieved (RAGAS style).
Correctness vs truth asks whether it agrees with the source of truth (this customer's
account+catalog). The gap between them is where the expensive failures live, and the gate
is the second one, a rule over structured claims (principle 10), not a fuzzy judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from atlas.domain.oracle import truth_for


@dataclass
class Answer:
    text: str
    claims: dict = field(default_factory=dict)        # structured assertions, e.g. {"has_contract": False}
    grounded_in: dict = field(default_factory=dict)   # the retrieved document's facts


def is_faithful(answer: Answer) -> bool:
    """Every claim is consistent with the document the answer was grounded in."""
    return all(answer.grounded_in.get(k) == v for k, v in answer.claims.items())


def is_correct_vs_truth(answer: Answer, customer_id: str) -> bool:
    """Every claim is consistent with the source of truth for this customer."""
    truth = truth_for(customer_id)
    truth_facts = {"has_contract": truth.has_contract, "has_data_cap": truth.has_data_cap}
    return all(truth_facts.get(k) == v for k, v in answer.claims.items() if k in truth_facts)
