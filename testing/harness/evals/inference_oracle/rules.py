"""The rules engine: inference-truth derived from facts, not read from a column.

The shipped oracle (`domain/metrics.is_correct_vs_truth`) is LOOKUP truth: it checks a claimed value
against a stored field (`has_contract`, `has_data_cap`). That handles the easy half of "true". The
expensive failures are INFERENCE truth: "am I over my allowance this month?", "what does switching
plan cost me?". These are derivations over several facts plus policy, with no column to read. A flat
key value oracle cannot grade those without becoming a reasoner.

So this engine DERIVES the answer deterministically, in code, from the same account + catalog facts
the agent had. It is one independent computation of the truth. The differential oracle compares it
against the model's claim (the second computation) and flags disagreement, which is how you grade an
answer whose truth you never stored in advance.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Optional

from atlas.domain.accounts import get_account
from atlas.domain.catalog import compute_price, get_plan


def over_allowance(customer_id: str) -> Optional[bool]:
    """Is the customer over their data allowance this month? None when the plan is uncapped.

    A DERIVATION (usage vs cap), not a stored flag. `has_data_cap` says a cap exists. This says
    whether it was breached, which is the question a customer actually asks and the lookup oracle
    cannot express.
    """
    usage = get_account(customer_id).usage
    if usage.data_cap_gb is None:
        return None
    return usage.gigabytes_used > Decimal(usage.data_cap_gb)


def remaining_allowance_gb(customer_id: str) -> Optional[Decimal]:
    """Allowance left this month (negative when over). None when uncapped."""
    usage = get_account(customer_id).usage
    if usage.data_cap_gb is None:
        return None
    return Decimal(usage.data_cap_gb) - usage.gigabytes_used


def monthly_cost_change(customer_id: str, new_plan_id: str) -> Decimal:
    """The monthly price delta of switching to `new_plan_id` (negative is a saving). Derived from the
    catalog, the arithmetic the model must never do itself (principle 11)."""
    current = get_plan(get_account(customer_id).plan_id).monthly_price
    return compute_price(new_plan_id) - current


# The named derivations the differential oracle can check a claim against. Each takes a customer_id
# and any extra args the claim carries (e.g. the target plan for a cost change).
RULES = {
    "over_allowance": over_allowance,
    "remaining_allowance_gb": remaining_allowance_gb,
    "monthly_cost_change": monthly_cost_change,
}


__all__ = ["RULES", "monthly_cost_change", "over_allowance", "remaining_allowance_gb"]
