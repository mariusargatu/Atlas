"""Runnable demo of the differential oracle: catch inference-truth the lookup oracle can't see.

`task oracle` runs this. Daniel (cust_legacy_term) has used 512 GB against a 500 GB cap, over his
allowance. That fact is a DERIVATION (usage vs cap), not a stored column. The shipped lookup oracle
can confirm a cap *exists* and stops there. It has no way to ask whether he is *over* it. The
differential oracle derives the answer from the facts and compares it to the model's claim, so a
plausible "you're fine" answer that contradicts the arithmetic gets caught with no label stored in advance.

Pure domain reads, zero egress, no model call.
"""
from __future__ import annotations

from decimal import Decimal

from atlas.domain import accounts
from atlas.domain.metrics import Answer, is_correct_vs_truth

from evals.inference_oracle.claim import Claim
from evals.inference_oracle.differential import check
from evals.inference_oracle.rules import over_allowance, remaining_allowance_gb

_CUSTOMER = "cust_legacy_term"  # Daniel: 512 GB used against a 500 GB cap


def main() -> None:
    accounts.reset_state()
    usage = accounts.get_usage(_CUSTOMER)
    remaining = remaining_allowance_gb(_CUSTOMER)
    if remaining is None:
        cap_note = "uncapped plan — no allowance to exceed"
    elif remaining < 0:
        cap_note = f"over by {-remaining:.1f} GB"
    else:
        cap_note = f"{remaining:.1f} GB remaining"
    print(f"facts: {usage.gigabytes_used} GB used against a {usage.data_cap_gb} GB cap ({cap_note})\n")

    # The model's answer claims Daniel is within his allowance: over_allowance == False.
    print('agent answer: "Good news, you are comfortably within your data allowance this month."')
    print("structured claim: over_allowance = False\n")

    # 1) The lookup oracle: it can only check a stored column. "Has a cap?" yes, and it stops.
    lookup_ok = is_correct_vs_truth(Answer(text="...", claims={"has_data_cap": True}), _CUSTOMER)
    print(f"  lookup oracle  (is_correct_vs_truth, has_data_cap={lookup_ok}): "
          "looks fine — it can confirm a cap EXISTS, but cannot ask whether he is OVER it")

    # 2) The differential oracle: derive the truth, compare to the claim.
    verdict = check(Claim("over_allowance", False), _CUSTOMER)
    print(f"  differential   {verdict.render()}\n")

    # A second inference truth question: what does switching to the current Fast plan cost?
    print('agent answer: "Switching to Fast would cost you about £4 more a month."')
    cost_verdict = check(Claim("monthly_cost_change", Decimal("4.00"), args=("plan_current_fast",)), _CUSTOMER)
    print(f"  differential   {cost_verdict.render()}")
    print("  (Fast is £35, the legacy plan is £39, so switching SAVES £4 — the claim has the sign backwards.)\n")

    print("Lookup-truth grades a column. Inference-truth grades a derivation. Most expensive failures")
    print("live in the second, and a flat key-value oracle waves them through.")


if __name__ == "__main__":
    main()
