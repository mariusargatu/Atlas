"""Compare two decision records: did behaviour drift, or only the prose?

This is the lane's whole point. Replay pins a PROXY of the model (the cassette), and nothing re-checks
whether that proxy still matches the live model. When a provider silently updates the model behind a
stable version string, the request stays byte identical, replay returns the old response forever, and
the suite stays green while production moves. The drift lane re-records against the new model and
compares the DECISIONS. A reworded answer is prose drift (low signal). A changed tool call, a flipped
guard, a different outcome is BEHAVIOURAL drift, the silent move a green suite would never show you.
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
        if sev == "none":
            return "DRIFT none        — decisions and prose identical (the green that is actually green)"
        if sev == "prose":
            return "DRIFT prose       — wording moved, decisions held (low signal)"
        parts = []
        for field in self.changed_decisions:
            parts.append(f"{field}: {getattr(self.old, field)!r} → {getattr(self.new, field)!r}")
        return "DRIFT BEHAVIOURAL — " + " ; ".join(parts)


def compare(old: DecisionRecord, new: DecisionRecord) -> DriftReport:
    """Field by field over the DECISIONS. The prose is compared only as a digest, kept separate.

    Iterates `DECISION_KEYS` (record.py's single source of truth) so the diff can never fall out of
    sync with the set of fields the decision digest covers.
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
