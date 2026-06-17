"""The oracle, the source of ground truth a test consults to decide whether an answer was
correct. It is the account joined to the catalog, never a frozen value (principle 9).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from atlas.domain.accounts import get_account
from atlas.domain.catalog import get_plan


@dataclass(frozen=True)
class Truth:
    has_contract: bool
    early_termination_fee: Decimal
    has_data_cap: bool


def truth_for(customer_id: str) -> Truth:
    """The true facts for this customer, from account + catalog, what claims are judged against."""
    plan = get_plan(get_account(customer_id).plan_id)
    return Truth(
        has_contract=plan.has_term,
        early_termination_fee=plan.early_termination_fee,
        has_data_cap=plan.data_cap_gb is not None,
    )
