"""The differential oracle (a spike): grade inference truth the lookup oracle cannot see.

`is_correct_vs_truth` is lookup truth, a claimed value vs a stored column. It cannot grade a
DERIVATION like "am I over my allowance?" (usage vs cap) or "what does switching cost?" (catalog
arithmetic). The differential oracle computes the truth independently in a rules engine and flags any
disagreement with the model's claim, catching plausible but wrong answers with no label stored in advance.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from atlas.domain.metrics import Answer, is_correct_vs_truth

from evals.inference_oracle.claim import Claim
from evals.inference_oracle.differential import check
from evals.inference_oracle.rules import monthly_cost_change, over_allowance, remaining_allowance_gb
from testing.tests.fixtures.catalog_expectations import EXPECTED_CURRENT_PLAN, EXPECTED_LEGACY_PLAN

_DANIEL = "cust_legacy_term"   # 512 GB used against the legacy plan's data cap (see EXPECTED_LEGACY_PLAN)
_SARAH = "cust_current"        # uncapped, on the current plan (see EXPECTED_CURRENT_PLAN)


# ---- the rules engine derives, it does not look up ----

def test_over_allowance_is_derived_from_usage_vs_cap():
    assert over_allowance(_DANIEL) is True       # 512 > the legacy plan's cap
    assert over_allowance(_SARAH) is None        # uncapped -> the question does not apply


def test_remaining_allowance_can_go_negative():
    assert remaining_allowance_gb(_DANIEL) == Decimal(EXPECTED_LEGACY_PLAN.data_cap_gb) - Decimal("512.0")
    assert remaining_allowance_gb(_SARAH) is None


def test_monthly_cost_change_is_a_signed_catalog_delta():
    # Current plan price minus legacy plan price, read from the fixture, never hand computed.
    assert monthly_cost_change(_DANIEL, "plan_current_fast") == (
        EXPECTED_CURRENT_PLAN.monthly_price - EXPECTED_LEGACY_PLAN.monthly_price
    )


# ---- the differential oracle agrees / disagrees ----

def test_disagrees_when_the_claim_contradicts_the_derivation():
    verdict = check(Claim("over_allowance", False), _DANIEL)   # agent says "within allowance"
    assert not verdict.agree and verdict.derived is True and verdict.claimed is False


def test_agrees_when_the_claim_matches_the_derivation():
    verdict = check(Claim("over_allowance", True), _DANIEL)
    assert verdict.agree


def test_catches_a_money_claim_with_the_sign_backwards():
    derived_delta = EXPECTED_CURRENT_PLAN.monthly_price - EXPECTED_LEGACY_PLAN.monthly_price
    claimed_costs_more = check(Claim("monthly_cost_change", -derived_delta, args=("plan_current_fast",)), _DANIEL)
    assert not claimed_costs_more.agree and claimed_costs_more.derived == derived_delta
    claimed_saves = check(Claim("monthly_cost_change", derived_delta, args=("plan_current_fast",)), _DANIEL)
    assert claimed_saves.agree


def test_unknown_claim_kind_is_a_hard_error_not_a_silent_pass():
    with pytest.raises(KeyError):
        check(Claim("will_it_rain", True), _DANIEL)


def test_uncapped_customer_is_not_applicable_not_a_false_disagreement():
    # Sarah is uncapped, so over_allowance derives None. A defensible "you are not over your
    # allowance" claim must NOT be graded as a contradiction. It is a distinct N/A verdict.
    verdict = check(Claim("over_allowance", False), _SARAH)
    assert verdict.applicable is False
    assert verdict.agree is False                 # not a match...
    assert "does not apply" in verdict.reason     # ...but reported as N/A, not a contradiction
    assert "N/A" in verdict.render()


def test_misshaped_claim_args_raise_a_clear_error_not_an_opaque_crash():
    with pytest.raises(TypeError):                # monthly_cost_change needs a plan arg
        check(Claim("monthly_cost_change", Decimal("4.00")), _DANIEL)
    with pytest.raises(KeyError):                 # ...and that plan must exist
        check(Claim("monthly_cost_change", Decimal("4.00"), args=("plan_nonexistent",)), _DANIEL)


def test_render_speaks_agree_and_disagree():
    assert "DISAGREE" in check(Claim("over_allowance", False), _DANIEL).render()
    assert "AGREE" in check(Claim("over_allowance", True), _DANIEL).render()


# ---- the gap: the lookup oracle is blind to the over-allowance question ----

def test_lookup_oracle_cannot_grade_the_over_allowance_claim():
    # The lookup oracle only knows has_contract / has_data_cap. An answer that claims a cap exists is
    # "correct" to it (one does), so it waves Daniel through, while he is 12 GB over that very cap.
    looks_fine = is_correct_vs_truth(Answer(text="...", claims={"has_data_cap": True}), _DANIEL)
    assert looks_fine is True
    # The differential oracle, asked the question lookup cannot express, catches it.
    assert not check(Claim("over_allowance", False), _DANIEL).agree
