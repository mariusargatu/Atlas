"""The differential oracle (a spike): grade inference-truth the lookup oracle cannot see.

`is_correct_vs_truth` is lookup-truth, a claimed value vs a stored column. It cannot grade a
DERIVATION like "am I over my allowance?" (usage vs cap) or "what does switching cost?" (catalog
arithmetic). The differential oracle computes the truth independently in a rules engine and flags any
disagreement with the model's claim, catching plausible-but-wrong answers with no pre-stored label.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from atlas.domain.metrics import Answer, is_correct_vs_truth

from evals.inference_oracle.claim import Claim
from evals.inference_oracle.differential import check
from evals.inference_oracle.rules import monthly_cost_change, over_allowance, remaining_allowance_gb

_DANIEL = "cust_legacy_term"   # 512 GB used, 500 GB cap, on the £39 legacy plan
_SARAH = "cust_current"        # uncapped, on the £35 current plan


# ---- the rules engine derives, it does not look up ----

def test_over_allowance_is_derived_from_usage_vs_cap():
    assert over_allowance(_DANIEL) is True       # 512 > 500
    assert over_allowance(_SARAH) is None        # uncapped -> the question does not apply


def test_remaining_allowance_can_go_negative():
    assert remaining_allowance_gb(_DANIEL) == Decimal("-12.0")
    assert remaining_allowance_gb(_SARAH) is None


def test_monthly_cost_change_is_a_signed_catalog_delta():
    # £35 current Fast minus £39 legacy = a £4 saving (negative).
    assert monthly_cost_change(_DANIEL, "plan_current_fast") == Decimal("-4.00")


# ---- the differential oracle agrees / disagrees ----

def test_disagrees_when_the_claim_contradicts_the_derivation():
    verdict = check(Claim("over_allowance", False), _DANIEL)   # agent says "within allowance"
    assert not verdict.agree and verdict.derived is True and verdict.claimed is False


def test_agrees_when_the_claim_matches_the_derivation():
    verdict = check(Claim("over_allowance", True), _DANIEL)
    assert verdict.agree


def test_catches_a_money_claim_with_the_sign_backwards():
    claimed_costs_more = check(Claim("monthly_cost_change", Decimal("4.00"), args=("plan_current_fast",)), _DANIEL)
    assert not claimed_costs_more.agree and claimed_costs_more.derived == Decimal("-4.00")
    claimed_saves = check(Claim("monthly_cost_change", Decimal("-4.00"), args=("plan_current_fast",)), _DANIEL)
    assert claimed_saves.agree


def test_unknown_claim_kind_is_a_hard_error_not_a_silent_pass():
    with pytest.raises(KeyError):
        check(Claim("will_it_rain", True), _DANIEL)


def test_uncapped_customer_is_not_applicable_not_a_false_disagreement():
    # Sarah is uncapped, so over_allowance derives None. A defensible "you are not over your
    # allowance" claim must NOT be graded as a contradiction; it is a distinct N/A verdict.
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
    # "correct" to it (one does), so it waves Daniel through — while he is 12 GB over that very cap.
    looks_fine = is_correct_vs_truth(Answer(text="...", claims={"has_data_cap": True}), _DANIEL)
    assert looks_fine is True
    # The differential oracle, asked the question lookup cannot express, catches it.
    assert not check(Claim("over_allowance", False), _DANIEL).agree
