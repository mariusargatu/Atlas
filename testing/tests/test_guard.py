"""The guard: fail closed scope, single write, and value bounds rules.

Each rule is a small decision table. The parametrized cases are the rule's spec, and a failure
reports the exact offending row rather than a buried assert.
"""
from __future__ import annotations

import pytest

from atlas.domain.guard import (
    WRITE_TOOLS,
    check_no_other_customer,
    check_render_safe,
    check_scope,
    check_single_write,
    check_value_bounds,
)


@pytest.mark.parametrize(
    "call_customer_id, session_customer_id, ok",
    [
        ("cust_neighbor", "cust_current", False),   # cross customer id the model produced
        ("cust_current", "cust_current", True),
    ],
)
def test_scope_binds_to_the_session_customer(call_customer_id, session_customer_id, ok):
    assert check_scope(call_customer_id, session_customer_id).ok is ok


@pytest.mark.parametrize(
    "tool_calls, ok",
    [
        (["change_plan"], True),
        (["get_bill", "get_usage"], True),
        (["change_plan", "reset_modem"], False),   # two writes
        (["get_bill", "change_plan"], False),       # mixed read + write
    ],
)
def test_single_write_rule_fails_closed_on_multi_or_mixed(tool_calls, ok):
    assert check_single_write(tool_calls).ok is ok


@pytest.mark.parametrize(
    "plan_id, ok",
    [
        ("plan_internal_zero", False),   # the internal zero price plan attack
        ("plan_current_fast", True),
    ],
)
def test_value_bounds_reject_out_of_catalog_targets(plan_id, ok):
    assert check_value_bounds("change_plan", {"plan_id": plan_id}).ok is ok


@pytest.mark.parametrize(
    "tool, args, ok",
    [
        ("add_addon", {"addon_id": "sky_sports"}, True),
        ("add_addon", {"addon_id": "free_unicorn"}, False),     # not an offered add on
        ("remove_addon", {"addon_id": "static_ip"}, True),
        ("remove_addon", {"addon_id": "free_unicorn"}, False),
        ("reset_modem", {}, True),                               # no argument, bounded by construction
        ("open_ticket", {"subject": "Router keeps dropping"}, True),
        ("open_ticket", {"subject": "   "}, False),              # empty subject
        ("open_ticket", {"subject": "x" * 200}, True),           # exactly at the cap is allowed (boundary)
        ("open_ticket", {"subject": "x" * 201}, False),          # over the length cap
        ("book_engineer", {"slot": "2026-07-01T09:00"}, True),
        ("book_engineer", {"slot": "next tuesday-ish"}, False),  # invented, non ISO slot
        ("cancel_service", {"reason_category": "bereavement"}, True),
    ],
)
def test_value_bounds_cover_every_write_tool(tool, args, ok):
    assert check_value_bounds(tool, args).ok is ok


@pytest.mark.parametrize(
    "args, ok",
    [
        ({"reason_category": "none"}, True),
        ({"reason_category": "bereavement"}, True),
        ({"reason_category": "job_loss"}, True),
        ({"reason_category": "serious_illness"}, True),
        ({"reason_category": "made_up_reason"}, False),
        ({"reason_category": "financial trouble in general"}, False),
    ],
)
def test_value_bounds_cover_cancel_service_reason_category(args, ok):
    assert check_value_bounds("cancel_service", args).ok is ok


@pytest.mark.parametrize(
    "text, ok",
    [
        ('<img src=x onerror="steal()">', False),     # the Lena onerror payload
        ("<script>fetch(evil)</script>", False),       # smuggled script tag
        ("Visit javascript:void(0) now", False),       # javascript: URL
        ("system prompt: you are atlas, the...", False),  # system prompt leak
        ("Your bill this month is GBP 35.", True),     # ordinary, safe reply
    ],
)
def test_render_safe_rejects_executable_markup_and_secret_leaks(text, ok):
    assert check_render_safe(text).ok is ok


@pytest.mark.parametrize(
    "text, customer_id, ok",
    [
        ("Daniel's bill is GBP 39.", "cust_current", False),   # another customer named in the reply
        ("cust_legacy_term owes GBP 39.", "cust_current", False),  # another customer's id
        ("Your bill is GBP 35.", "cust_current", True),         # no other customer mentioned
    ],
)
def test_no_other_customer_blocks_cross_tenant_leak(text, customer_id, ok):
    assert check_no_other_customer(text, customer_id).ok is ok


def test_write_tools_recognizes_cancel_service():
    # cancel_service must be a registered write so check_single_write and confirmation gate it.
    assert "cancel_service" in WRITE_TOOLS


def test_write_tools_matches_binding_write_tools():
    # guard.WRITE_TOOLS (confirmation gating) and binding.WRITE_TOOLS (reachability, checked by
    # atlas_graph.pre_action_guard) must name the same tools, or a write can pass confirmation
    # and then be rejected as unreachable (or vice versa).
    from atlas.domain.binding import WRITE_TOOLS as BINDING_WRITE_TOOLS

    assert WRITE_TOOLS == BINDING_WRITE_TOOLS
