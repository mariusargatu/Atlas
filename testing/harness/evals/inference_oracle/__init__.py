"""The differential oracle: inference-truth for the harness (a spike).

The shipped oracle is lookup truth, a claimed value vs a stored column. It handles the easy half of
"true" and quietly implies it generalizes. It does not. The expensive failures are inference truth:
derivations over several facts plus policy, with no column to read. This package grades those by
computing the truth two independent ways (a deterministic rules engine and the model's claim) and
flagging disagreement, so an answer whose truth was never stored in advance can still be caught.
"""
from __future__ import annotations

from evals.inference_oracle.claim import Claim
from evals.inference_oracle.differential import OracleVerdict, check
from evals.inference_oracle.rules import RULES

__all__ = ["Claim", "OracleVerdict", "RULES", "check"]
