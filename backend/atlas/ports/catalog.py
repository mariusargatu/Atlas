"""The catalog port, what the provider sells (the oracle's right half).

``check_eligibility``/``compute_price`` are deterministic catalog logic, never model arithmetic:
the model proposes a plan, the catalog decides eligibility and the price.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from atlas.domain.catalog import Plan


class CatalogReader(Protocol):
    def get_plan(self, plan_id: str) -> Plan: ...
    def list_plans(self) -> list[Plan]: ...
    def compute_price(self, plan_id: str) -> Decimal: ...
    def check_eligibility(self, plan_id: str) -> bool: ...
