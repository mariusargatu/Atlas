import pytest

from atlas.domain.accounts import HARDSHIP_REASON_CATEGORIES, cancellation_fee_outcome


def test_hardship_reason_categories_are_bereavement_job_loss_serious_illness():
    assert HARDSHIP_REASON_CATEGORIES == frozenset({"bereavement", "job_loss", "serious_illness"})


def test_no_term_plan_has_no_fee_regardless_of_reason():
    # cust_current (Sarah) is on plan_current_fast, which carries no early-termination fee
    assert cancellation_fee_outcome("cust_current", "none") == "none"
    assert cancellation_fee_outcome("cust_current", "bereavement") == "none"


@pytest.mark.parametrize("reason", ["bereavement", "job_loss", "serious_illness"])
def test_termed_plan_with_hardship_reason_is_waived_pending_verification(reason):
    # cust_legacy_term (Daniel) is on plan_legacy_value, which carries a real fee (see EXPECTED_LEGACY_PLAN)
    assert cancellation_fee_outcome("cust_legacy_term", reason) == "waived_pending_verification"


def test_termed_plan_with_no_hardship_reason_is_standard():
    assert cancellation_fee_outcome("cust_legacy_term", "none") == "standard"


def test_unknown_reason_category_raises():
    with pytest.raises(ValueError, match="unknown reason_category"):
        cancellation_fee_outcome("cust_legacy_term", "made_up_reason")
