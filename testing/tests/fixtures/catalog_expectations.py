"""Independently owned expectations for the live catalog, maintained by the test suite, never
imported from atlas.domain.catalog's own CATALOG data. This module reuses the Plan dataclass
SHAPE only -- its field values are typed here by hand, on purpose, so a test that reads from this
module can actually disagree with production. A test that instead imported CATALOG directly would
agree with production no matter what production said, which would not be testing anything.

When a real catalog value changes, this file must be updated as its own deliberate step. Until it
is, test_catalog_matches_fixture.py fails, which is the intended signal: a real data change should
never pass silently just because nothing was asserting against a stale copy anymore.
"""
from __future__ import annotations

from decimal import Decimal

from atlas.domain.catalog import Plan

EXPECTED_CURRENT_PLAN = Plan(
    id="plan_current_fast",
    name="Fiber 100",
    monthly_price=Decimal("29.99"),
    has_term=False,
    early_termination_fee=Decimal("0.00"),
    data_cap_gb=None,
)

EXPECTED_LEGACY_PLAN = Plan(
    id="plan_legacy_value",
    name="Fiber 100 Legacy",
    monthly_price=Decimal("24.99"),
    has_term=True,
    early_termination_fee=Decimal("150.00"),
    data_cap_gb=500,
)

__all__ = ["EXPECTED_CURRENT_PLAN", "EXPECTED_LEGACY_PLAN"]
