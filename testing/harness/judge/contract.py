"""The judge contract: the versioned serial number of a measuring instrument.

A precise triple: the judge model id (pinned, not a floating alias), the rubric version, and a hash
of the prompt template. Changing any field means a new instrument with an unknown calibration, so a
kappa earned against the old triple is void. The fingerprint is a canonical digest of the triple (the
same machinery the cassette key uses), so every score is stamped with the instrument that produced it.

Absorbed verbatim from the pre rewrite `evals/judge/contract.py` (SP8 task 1, per the planning
digest's own disposition: "keep verbatim... D15's own identity rule word for word").
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from determinism.canonical import digest


@dataclass(frozen=True)
class JudgeContract:
    """The unit of judge versioning: ``(judge_model_id, rubric_version, prompt_template_hash)``.

    Frozen, because a contract is an identity, not a mutable config. To change the instrument you
    construct a new contract, which yields a new ``fingerprint`` and forces a recalibration.
    """

    judge_model_id: str
    rubric_version: str
    prompt_template_hash: str

    def fingerprint(self) -> str:
        """A stable digest of the triple. Two judges share a fingerprint iff all three fields match,
        so a vendor model bump, a rubric edit, or a template tweak each produce a distinct id."""
        return digest(asdict(self))


__all__ = ["JudgeContract"]
