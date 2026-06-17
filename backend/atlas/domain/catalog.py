"""The catalog, what the provider sells. Pure domain. The model never does this arithmetic.

`compute_price` / `check_eligibility` are deterministic catalog logic: the
model proposes a plan, the catalog decides. Money is `Decimal`, never a float.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Plan:
    id: str
    name: str
    monthly_price: Decimal
    has_term: bool                  # is there a minimum term contract?
    early_termination_fee: Decimal  # Decimal("0.00") if none
    data_cap_gb: int | None         # None == uncapped


# One product universe with the registry (`corpus/registry/core.yaml`), enforced by
# testing/tests/test_catalog_matches_registry.py: the current plan is Fiber 100 (term free,
# uncapped, no early-termination fee); the legacy plan is Fiber 100 Legacy (discontinued), which
# carries a 12 month term, the 150.00 early-termination fee, and a hard cap. Daniel
# (accounts.SEED["cust_legacy_term"]) is on the legacy plan; the current Fiber 100 page no longer
# describes his terms. That gap is the cold open: an answer grounded in the current page is false
# for Daniel. The plan IDs are opaque internal handles (never customer facing), so they keep their
# historical `_fast`/`_value` names; the customer-facing plan NAME is what tells the one story.
CATALOG: dict[str, Plan] = {
    "plan_current_fast": Plan(
        "plan_current_fast", "Fiber 100", Decimal("29.99"),
        has_term=False, early_termination_fee=Decimal("0.00"), data_cap_gb=None,
    ),
    "plan_legacy_value": Plan(
        "plan_legacy_value", "Fiber 100 Legacy", Decimal("24.99"),
        has_term=True, early_termination_fee=Decimal("150.00"), data_cap_gb=500,
    ),
}


@dataclass(frozen=True)
class Addon:
    id: str
    name: str
    monthly_price: Decimal


# The add ons the provider offers. An add_addon/remove_addon for anything not here is out of
# bounds, the same "is this a real, offered thing?" rule the plan check applies.
ADDONS: dict[str, Addon] = {
    "static_ip": Addon("static_ip", "Static IP", Decimal("5.00")),
    "sky_sports": Addon("sky_sports", "Sky Sports", Decimal("20.00")),
    "line_rental": Addon("line_rental", "Line Rental", Decimal("12.00")),
}


def get_plan(plan_id: str) -> Plan:
    return CATALOG[plan_id]


def compute_price(plan_id: str) -> Decimal:
    return CATALOG[plan_id].monthly_price


def check_eligibility(plan_id: str) -> bool:
    """Discontinued plans cannot be newly taken, but current plans can."""
    return plan_id in CATALOG and "legacy" not in CATALOG[plan_id].name.lower()
