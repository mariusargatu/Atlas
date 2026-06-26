"""One provenance vocabulary over both golden record types.

`GoldenCase` carries `source`/`author_role`/`tier`/`category`; a dataset contract case carries
`origin`/`split`/`source_trace_id`. They name the same three concepts differently, so this maps both
onto one shape rather than making either record type move. Canonical vocabulary is the dataset
schema's own `origin` enum.

    canonical      dataset `origin`   GoldenCase.source   tier
    synthetic      synthetic          generated           gold
    authored       authored           hand_written        gold
    promoted       promoted           production          silver
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from evals.evalkit.risk import risk_of

ORIGINS = ("synthetic", "authored", "promoted")
TIERS = ("gold", "silver")

#: canonical origin -> the `GoldenCase.source` literal that means the same thing
_ORIGIN_TO_SOURCE = {"synthetic": "generated", "authored": "hand_written", "promoted": "production"}
_SOURCE_TO_ORIGIN = {v: k for k, v in _ORIGIN_TO_SOURCE.items()}

#: A case is gold unless it came from unratified production traffic.
_ORIGIN_TO_TIER = {"synthetic": "gold", "authored": "gold", "promoted": "silver"}


@dataclass(frozen=True)
class Provenance:
    """The facets the unified set is sliced and gated by."""

    id: str
    origin: str
    source: str
    tier: str
    risk: str


def _from_dataset_case(case: Mapping[str, object]) -> Provenance:
    origin = str(case.get("origin") or "")
    if origin not in ORIGINS:
        raise ValueError(f"{case.get('case_id')}: unknown origin {origin!r}; known: {ORIGINS}")
    return Provenance(
        id=str(case["case_id"]),
        origin=origin,
        source=_ORIGIN_TO_SOURCE[origin],
        tier=_ORIGIN_TO_TIER[origin],
        risk=risk_of(case),
    )


def _from_golden_case(case) -> Provenance:
    origin = _SOURCE_TO_ORIGIN.get(case.source)
    if origin is None:
        raise ValueError(f"{case.id}: unknown origin for source {case.source!r}")
    if case.tier not in TIERS:
        raise ValueError(f"{case.id}: unknown tier {case.tier!r}; known: {TIERS}")
    return Provenance(id=case.id, origin=origin, source=case.source, tier=case.tier, risk=case.risk)


def provenance_of(record) -> Provenance:
    """Normalise either record type. A Mapping is a dataset contract case; anything else is a
    `GoldenCase` (duck typed on `.source`, so a future record type only needs those attributes)."""
    if isinstance(record, Mapping):
        return _from_dataset_case(record)
    return _from_golden_case(record)


__all__ = ["ORIGINS", "TIERS", "Provenance", "provenance_of"]
