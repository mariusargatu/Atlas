"""The catalog, what the provider sells. Pure domain; the model never does this arithmetic.

`compute_price` / `check_eligibility` are deterministic catalog logic (principle 11): the
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


# The current offer is term free and uncapped. The legacy plan (discontinued) is what the
# help pages no longer describe. It carries a 12 month term and a hard cap. This gap is
# the cold open: an answer grounded in the current page is false for a legacy customer.
CATALOG: dict[str, Plan] = {
    "plan_current_fast": Plan(
        "plan_current_fast", "Fast (current)", Decimal("35.00"),
        has_term=False, early_termination_fee=Decimal("0.00"), data_cap_gb=None,
    ),
    "plan_legacy_value": Plan(
        "plan_legacy_value", "Value (legacy, discontinued)", Decimal("39.00"),
        has_term=True, early_termination_fee=Decimal("120.00"), data_cap_gb=500,
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
    """Discontinued plans cannot be newly taken; current plans can."""
    return plan_id in CATALOG and "legacy" not in CATALOG[plan_id].name.lower()
