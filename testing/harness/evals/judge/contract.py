"""The judge contract: the versioned serial number of a measuring instrument.

An LLM judge is not "GPT, roughly." It is a precise triple, and that triple is a contract:
the judge model id (pinned, not a floating alias), the rubric version (the exact wording of what
you asked it to score), and a hash of the prompt template (the scaffolding around the rubric).
Change any one field and you do not have the same instrument anymore, you have a new one with an
unknown calibration, so a kappa earned against the old triple is void, not suspect.

The fingerprint is a canonical digest (the same machinery the cassette key uses, principle 8), so a
judge swap is attributable: every score can be stamped with the exact instrument that produced it,
and a calibration is only ever compared against scores from the same fingerprint.
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
