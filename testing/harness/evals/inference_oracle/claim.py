"""A structured claim the agent made about an inference-truth question.

The differential oracle compares this against a rules engine derivation. The `kind` names the
derivation. `value` is what the agent asserted. `args` carries any parameters the derivation needs
(e.g. the target plan for a cost change). Extracting a `Claim` from free text prose is a separate,
fuzzy problem that belongs to the metrics article. Here the claim is already structured, the same way
the cold open test works from a structured answer rather than parsing the sentence.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Claim:
    kind: str                       # a key in rules.RULES, e.g. "over_allowance"
    value: object                   # what the agent asserted (bool | Decimal)
    args: tuple = field(default_factory=tuple)  # extra derivation args, e.g. (new_plan_id,)


__all__ = ["Claim"]
