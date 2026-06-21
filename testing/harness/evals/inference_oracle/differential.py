"""The differential oracle: grade inference-truth by two independent computations.

Differential / metamorphic testing applied to the oracle problem. The truth is computed twice, once
by a trusted deterministic rules engine and once by the model (the claim under test). If the two
disagree, the answer is wrong — and you reached that verdict WITHOUT a pre-stored label, which is the
only way to grade an answer whose truth you could never tabulate in advance.

This is the inference-truth complement to the lookup oracle. Where `is_correct_vs_truth` checks a
claimed value against a stored column, this checks a claimed value against a DERIVATION over the
facts. The cold open caught a faithful-but-false answer with a lookup; this catches a plausible-but-
wrong DERIVED answer with a difference.
"""
from __future__ import annotations

from dataclasses import dataclass

from evals.inference_oracle.claim import Claim
from evals.inference_oracle.rules import RULES


@dataclass(frozen=True)
class OracleVerdict:
    kind: str
    agree: bool          # did the rules-engine derivation match the claim? (False when not applicable)
    derived: object      # the trusted, independently computed truth (None when not applicable)
    claimed: object      # what the agent asserted
    reason: str
    applicable: bool = True  # False when the derivation does not apply (e.g. over-allowance, uncapped)

    def render(self) -> str:
        flag = "N/A     " if not self.applicable else "AGREE   " if self.agree else "DISAGREE"
        return f"{flag} {self.kind}: derived={self.derived!r} claimed={self.claimed!r} — {self.reason}"


def check(claim: Claim, customer_id: str) -> OracleVerdict:
    """Derive the truth for `claim.kind` and compare it to what the agent asserted.

    Three outcomes, not two: AGREE / DISAGREE when the truth is derivable, and a distinct NOT-
    APPLICABLE when the rule returns ``None`` (the question has no derivable truth for this customer,
    e.g. "are you over your allowance?" on an uncapped plan). Collapsing N/A into DISAGREE would flag
    a correct answer as a contradiction, so it gets its own verdict.
    """
    if claim.kind not in RULES:
        raise KeyError(f"no rule for claim kind {claim.kind!r}; known: {sorted(RULES)}")
    try:
        derived = RULES[claim.kind](customer_id, *claim.args)
    except TypeError as exc:  # wrong number/shape of args for this rule — a malformed claim
        raise TypeError(
            f"rule {claim.kind!r} called with the wrong args {claim.args!r} for {customer_id!r}: {exc}"
        ) from exc
    except KeyError as exc:  # the derivation referenced an unknown id (e.g. a non-existent plan)
        raise KeyError(
            f"rule {claim.kind!r} references unknown id {exc} (customer {customer_id!r}, args {claim.args!r})"
        ) from exc
    if derived is None:
        return OracleVerdict(
            kind=claim.kind, agree=False, derived=None, claimed=claim.value,
            reason="the question does not apply to this customer (no derivable truth)",
            applicable=False,
        )
    agree = derived == claim.value
    reason = (
        "the derivation matches the claim"
        if agree
        else "the agent's answer contradicts the facts it had"
    )
    return OracleVerdict(kind=claim.kind, agree=agree, derived=derived, claimed=claim.value, reason=reason)


__all__ = ["OracleVerdict", "check"]
