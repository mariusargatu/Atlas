"""The one risk derivation shared by `Provenance.risk` (`provenance.py`) and `EvalCase.risk`
(`dataset_case.py`).

Both read the same dataset contract case and compute the SAME facet -- the thing `unified_set()`
joins a `Provenance` back to its projected `EvalCase` by isn't `risk` itself, but the two were kept
in lockstep by a byte identical one liner duplicated in both modules. A hand kept duplicate can
silently drift (one module gains a new fallback field, the other doesn't) and then the two would
disagree about the SAME case. One helper, both call it.

A leaf module on purpose: neither `provenance.py` nor `dataset_case.py` imports the other today, and
this must stay importable from both without creating a cycle.
"""
from __future__ import annotations

from collections.abc import Mapping


def risk_of(case: Mapping[str, object]) -> str:
    """A dataset contract case's risk facet: the first non-null of `adversarial_class`,
    `failure_class`, `intent`, defaulting to the empty string when none is present."""
    return str(case.get("adversarial_class") or case.get("failure_class") or case.get("intent") or "")


__all__ = ["risk_of"]
