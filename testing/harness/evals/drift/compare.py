"""Compare two decision records to classify a change as behavioural drift or only prose drift.

Replay pins a proxy of the model; nothing re-checks whether that proxy still matches the live one.
A provider can update the model behind a stable version string and the request stays byte
identical, so replay keeps returning the old response while production moves. A reworded answer is
prose drift, low signal. A changed tool call, a flipped guard, or a different outcome is
behavioural drift, the move a green suite would not show.
"""
from __future__ import annotations

from dataclasses import dataclass

from evals.drift.record import DECISION_KEYS, DecisionRecord


@dataclass(frozen=True)
class DriftReport:
    changed_decisions: tuple[str, ...]   # decision fields that moved: intent / tools / guards / outcome
    prose_changed: bool                  # the shipped text changed (claim_digest differs)
    old: DecisionRecord
    new: DecisionRecord

    def severity(self) -> str:
        """"behavioural" if any decision moved, else "prose" if only the wording did, else "none"."""
        if self.changed_decisions:
            return "behavioural"
        return "prose" if self.prose_changed else "none"

    def render(self) -> str:
        sev = self.severity()
        if sev in ("none", "prose"):
            return f"drift={sev}"
        parts = " ".join(
            f"{field}={getattr(self.old, field)!r}->{getattr(self.new, field)!r}"
            for field in self.changed_decisions
        )
        return f"drift=behavioural {parts}"


def compare(old: DecisionRecord, new: DecisionRecord) -> DriftReport:
    """Field by field over the decisions. The prose is compared only as a digest, kept separate.

    Iterates `DECISION_KEYS` (record.py's single source of truth) so the diff can never fall out of
    sync with the set of decision fields.
    """
    changed = tuple(
        field
        for field in DECISION_KEYS
        if getattr(old, field) != getattr(new, field)
    )
    return DriftReport(
        changed_decisions=changed,
        prose_changed=old.claim_digest != new.claim_digest,
        old=old,
        new=new,
    )


__all__ = ["DriftReport", "compare"]
