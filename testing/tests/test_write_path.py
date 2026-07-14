"""The write path: typed confirmation + checkpoint boundary idempotency.

The highest consequence surface. A timed out write retried with the same key applies once.
"""
from __future__ import annotations

import pytest

from determinism.sources import IdFactory

from atlas.domain import accounts
from atlas.domain.accounts import apply_write
from atlas.domain.actions import ActionsBackend
from atlas.domain.confirmation import ConfirmationError, PendingAction, execute_if_confirmed

from testing.tests.fixtures.catalog_expectations import EXPECTED_CURRENT_PLAN


def _pending(key: str = "idem-1") -> PendingAction:
    return PendingAction("change_plan", {"plan_id": "plan_current_fast"}, key, "cust_current")


def _wt_backend() -> ActionsBackend:
    """A write through backend, confirmed writes mutate the account store, like the real app."""
    return ActionsBackend(IdFactory("ref"), writer=apply_write)


def _confirm(backend, tool, args, customer_id, key):
    return execute_if_confirmed(PendingAction(tool, args, key, customer_id), "CONFIRM", backend)


def test_typed_confirmation_is_required():
    backend = ActionsBackend(IdFactory("ref"))
    with pytest.raises(ConfirmationError):
        execute_if_confirmed(_pending(), "yes", backend)  # a bare yes is not enough


def test_retry_with_same_key_applies_exactly_once():
    backend = ActionsBackend(IdFactory("ref"))
    pending = _pending()
    first = execute_if_confirmed(pending, "CONFIRM", backend)
    retry = execute_if_confirmed(pending, "CONFIRM", backend)  # timed out retry, same key
    assert first.applied and not retry.applied
    assert first.reference == retry.reference
    assert backend.change_count("cust_current") == 1


def test_a_distinct_action_applies_again():
    backend = ActionsBackend(IdFactory("ref"))
    execute_if_confirmed(_pending("idem-1"), "CONFIRM", backend)
    execute_if_confirmed(_pending("idem-2"), "CONFIRM", backend)
    assert backend.change_count("cust_current") == 2


# ---- write through: a confirmed write changes account state a later read sees ----

def test_change_plan_writes_through_and_reprices_the_bill():
    backend = _wt_backend()
    # Daniel starts on the legacy plan (term, capped). Move him to the current plan
    assert accounts.get_account("cust_legacy_term").plan_id == "plan_legacy_value"
    _confirm(backend, "change_plan", {"plan_id": "plan_current_fast"}, "cust_legacy_term", "k1")
    acct = accounts.get_account("cust_legacy_term")
    assert acct.plan_id == "plan_current_fast"
    assert accounts.get_bill("cust_legacy_term").amount == EXPECTED_CURRENT_PLAN.monthly_price  # re priced
    assert accounts.get_usage("cust_legacy_term").data_cap_gb is None                               # now uncapped


def test_add_then_remove_addon_writes_through():
    backend = _wt_backend()
    _confirm(backend, "add_addon", {"addon_id": "sky_sports"}, "cust_current", "k1")
    assert "sky_sports" in accounts.get_account("cust_current").addons
    _confirm(backend, "remove_addon", {"addon_id": "static_ip"}, "cust_current", "k2")  # seeded addon
    assert "static_ip" not in accounts.get_account("cust_current").addons


def test_open_ticket_writes_through_as_a_new_open_ticket():
    backend = _wt_backend()
    assert accounts.list_tickets("cust_current") == ()  # Sarah's only seeded ticket is closed
    _confirm(backend, "open_ticket", {"subject": "Router replacement"}, "cust_current", "k1")
    tickets = accounts.list_tickets("cust_current")
    assert [t.subject for t in tickets] == ["Router replacement"]
    assert tickets[0].status == "open"


def test_book_engineer_writes_through_to_bookings():
    backend = _wt_backend()
    _confirm(backend, "book_engineer", {"slot": "2026-07-01T09:00"}, "cust_current", "k1")
    assert accounts.get_account("cust_current").bookings == ("2026-07-01T09:00",)


def test_reset_modem_is_audited_without_changing_account_state():
    backend = _wt_backend()
    before = accounts.get_account("cust_current")
    _confirm(backend, "reset_modem", {}, "cust_current", "k1")
    assert accounts.get_account("cust_current") == before          # no persistent field changes
    assert backend.applied("cust_current", "reset_modem")          # but the audit log records it


def test_audit_query_filters_by_tool_not_just_customer():
    backend = _wt_backend()
    _confirm(backend, "change_plan", {"plan_id": "plan_current_fast"}, "cust_current", "k1")
    _confirm(backend, "open_ticket", {"subject": "Router replacement"}, "cust_current", "k2")
    # the audit query is scoped to the tool, not just the customer, two tools, one of each
    assert backend.change_count("cust_current") == 1                       # not 2
    assert len(backend.applied("cust_current", "open_ticket")) == 1
    assert len(backend.applied("cust_current")) == 2                       # both, when tool is unfiltered


def test_idempotency_key_gates_the_write_through_not_just_the_audit():
    backend = _wt_backend()
    # a timed out open_ticket retried with the SAME key must create exactly one ticket
    _confirm(backend, "open_ticket", {"subject": "Router replacement"}, "cust_current", "same-key")
    retry = _confirm(backend, "open_ticket", {"subject": "Router replacement"}, "cust_current", "same-key")
    assert retry.applied is False
    assert len(accounts.list_tickets("cust_current")) == 1  # the writer ran once, not twice


def test_cancel_service_is_audited_without_changing_account_state():
    backend = _wt_backend()
    before = accounts.get_account("cust_legacy_term")
    result = _confirm(backend, "cancel_service", {"reason_category": "bereavement"}, "cust_legacy_term", "idem-cancel-1")
    after = accounts.get_account("cust_legacy_term")
    assert result.applied
    assert after == before  # audit-only: no account field changed
    assert backend.applied("cust_legacy_term", "cancel_service")
