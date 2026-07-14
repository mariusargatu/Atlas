"""The four MCP servers exist with the right tools, and identity is never in a tool schema."""
from __future__ import annotations

import json
import re

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.mcp_servers.account_server import build_account_server
from atlas.mcp_servers.actions_server import build_actions_server
from atlas.mcp_servers.catalog_server import build_catalog_server
from atlas.mcp_servers.knowledge_server import build_knowledge_server

from testing.tests.fixtures.catalog_expectations import EXPECTED_CURRENT_PLAN, EXPECTED_LEGACY_PLAN


@pytest.mark.asyncio
async def test_catalog_server_returns_plan_details():
    server = build_catalog_server()
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        result = await client.call_tool("get_plan", {"plan_id": "plan_legacy_value"})
        payload = json.loads(result.content[0].text)
        assert payload["has_term"] is EXPECTED_LEGACY_PLAN.has_term
        assert payload["early_termination_fee"] == f"D:{EXPECTED_LEGACY_PLAN.early_termination_fee.normalize():f}"  # money is value normalized, canonical


@pytest.mark.asyncio
async def test_catalog_server_computes_price_and_eligibility_in_code():
    server = build_catalog_server()
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"compute_price", "check_eligibility"} <= names  # the catalog decides, not the model
        price = json.loads((await client.call_tool("compute_price", {"plan_id": "plan_current_fast"})).content[0].text)
        assert price["monthly_price"] == f"D:{EXPECTED_CURRENT_PLAN.monthly_price.normalize():f}"  # money is value normalized, canonical
        legacy = json.loads((await client.call_tool("check_eligibility", {"plan_id": "plan_legacy_value"})).content[0].text)
        assert legacy["eligible"] is False  # discontinued plans cannot be newly taken
        current = json.loads((await client.call_tool("check_eligibility", {"plan_id": "plan_current_fast"})).content[0].text)
        assert current["eligible"] is True   # a current plan CAN be taken (the positive case)


@pytest.mark.asyncio
async def test_account_server_exposes_the_four_reads_without_identity_in_schema():
    server = build_account_server("cust_legacy_term")
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        tools = (await client.list_tools()).tools
        names = {t.name for t in tools}
        assert {"get_usage", "get_bill", "get_equipment", "list_tickets"} <= names
        for tool in tools:  # identity is bound from the session, never a tool arg
            assert "customer_id" not in ((tool.inputSchema or {}).get("properties") or {})


@pytest.mark.asyncio
async def test_account_reads_are_scoped_to_the_session_customer():
    # Daniel (legacy, over cap, open ticket, owes 39) vs Sarah (current, paid, no open ticket)
    async with create_connected_server_and_client_session(build_account_server("cust_legacy_term")._mcp_server) as daniel:
        usage = json.loads((await daniel.call_tool("get_usage", {})).content[0].text)
        bill = json.loads((await daniel.call_tool("get_bill", {})).content[0].text)
        equip = json.loads((await daniel.call_tool("get_equipment", {})).content[0].text)
        tickets = json.loads((await daniel.call_tool("list_tickets", {})).content[0].text)
        assert usage["gigabytes_used"] == "D:512" and usage["data_cap_gb"] == EXPECTED_LEGACY_PLAN.data_cap_gb  # over the cap
        assert bill["amount"] == f"D:{EXPECTED_LEGACY_PLAN.monthly_price.normalize():f}" and bill["paid"] is False
        assert equip[0]["serial"] == "VB-7777"
        assert [t["ticket_id"] for t in tickets] == ["tk-2002"]  # the one OPEN ticket, closed ones filtered

    async with create_connected_server_and_client_session(build_account_server("cust_current")._mcp_server) as sarah:
        bill = json.loads((await sarah.call_tool("get_bill", {})).content[0].text)
        tickets = json.loads((await sarah.call_tool("list_tickets", {})).content[0].text)
        assert bill["amount"] == f"D:{EXPECTED_CURRENT_PLAN.monthly_price.normalize():f}"  # a different customer, a different bill: scope holds
        assert tickets == []                # her only ticket is closed, so no open tickets


@pytest.mark.asyncio
async def test_actions_server_has_write_tools_without_identity_in_schema():
    server = build_actions_server("cust_current")
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"change_plan", "reset_modem"} <= names
        for tool in (await client.list_tools()).tools:
            props = (tool.inputSchema or {}).get("properties") or {}
            assert "customer_id" not in props  # identity is bound from the session, not a tool arg


@pytest.mark.asyncio
async def test_cancel_service_tool_includes_computed_fee_outcome():
    server = build_actions_server("cust_legacy_term")
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        result = await client.call_tool("cancel_service", {"reason_category": "bereavement"})
        payload = json.loads(result.content[0].text)
        assert payload["action"] == "cancel_service"
        assert payload["reason_category"] == "bereavement"
        assert payload["fee_outcome"] == "waived_pending_verification"
        assert payload["customer"] == "cust_legacy_term"
        assert payload["status"] == "proposed"


@pytest.mark.asyncio
async def test_actions_server_has_cancel_service_without_identity_in_schema():
    server = build_actions_server("cust_current")
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        tools = (await client.list_tools()).tools
        names = {t.name for t in tools}
        assert "cancel_service" in names
        cancel = next(t for t in tools if t.name == "cancel_service")
        assert "customer_id" not in ((cancel.inputSchema or {}).get("properties") or {})


# --- D11: additionalProperties false + the identity ban generalized past the literal customer_id ----
#
# The two tests above (and test_mcp_transport.py's own) predate this generalization: each hardcodes
# the literal string "customer_id" on ONE server. That misses a hypothetical account_id, user_id, or
# session_customer_id parameter on a future tool, and never runs against catalog_server/knowledge_server
# at all (correctly, today -- neither carries identity -- but nothing WOULD catch it if one someday
# did). The tests below replace the per literal check with a name PATTERN, run across all four
# servers, every tool, every parameter.

# Session identity concepts (SP4 digest, "MCP hardening inventory"): identity is bound at MCP server
# construction from the caller's session/bearer token (ADR-027), never accepted as a tool argument.
# A parameter is banned when one of these concepts appears anywhere in its name as a whole
# underscore delimited token (start or "_" before it, "_" or end after it), whichever server,
# spelling, or SUFFIX it turns up under: account_id, user_id, session_customer_id, customer_account_id,
# but ALSO account_ref, user_ref, customer_number, account_uuid, user_token, customer_uuid (SP4 task
# 5 fix round 1: an earlier version of this pattern anchored the concept token to the END of the
# name (`_id` suffix or bare), so a concept word followed by anything OTHER than "_id" evaded it
# entirely -- not only the literal "customer_id" the narrower, now superseded checks above looked
# for, but any *_id specific anchor at all).
_IDENTITY_CONCEPTS = ("customer", "account", "user")
_IDENTITY_ID_PATTERN = re.compile(rf"(?:^|_)(?:{'|'.join(_IDENTITY_CONCEPTS)})(?:_|$)")

# Explicit allowlist for *_id parameters that are public, non session scoped data, never a signed in
# party, so the pattern above must not (and, checked below, does not) ban them. New legitimate *_id
# parameters are added here consciously, with a one line rationale: the CI signal D11 asks for is
# that a genuinely identity shaped name can never be silenced by adding it here instead of fixing the
# tool, so every entry needs an argument for why it is NOT identity, not just a name that happens to
# clear the pattern today.
_NON_IDENTITY_ID_ALLOWLIST = {
    "plan_id": "a public catalog plan identifier (catalog_server, actions_server.change_plan); "
    "plans are public data, not a signed in party",
    "addon_id": "a public catalog add on identifier (actions_server.add_addon/remove_addon); add "
    "ons are public catalog data, not a signed in party",
    "doc_id": "a public knowledge corpus document identifier; reserved for a future "
    "knowledge_server filter argument (no tool accepts it today)",
}


def _is_identity_shaped(param_name: str) -> bool:
    if param_name in _NON_IDENTITY_ID_ALLOWLIST:
        return False
    return bool(_IDENTITY_ID_PATTERN.search(param_name))


def test_identity_ban_pattern_generalizes_past_the_literal_customer_id_string():
    # the generalization's whole point: names the SP4 digest called out as uncaught by the old,
    # per server, hardcoded "customer_id" checks.
    for name in ("customer_id", "account_id", "user_id", "session_customer_id", "customer_account_id"):
        assert _is_identity_shaped(name), f"{name} should be banned by the generalized pattern"
    # SP4 task 5 fix round 1 (reviewer finding): the FIRST pattern only matched the concept token at
    # the END of the name (`_id` suffix or bare), so anything shaped `concept_<other suffix>` evaded
    # it entirely. These six names are real shaped identity params that name something other than
    # "_id" after the concept word; none of them may evade the ban.
    for name in (
        "account_ref", "user_ref", "customer_number", "account_uuid", "user_token", "customer_uuid",
    ):
        assert _is_identity_shaped(name), f"{name} should be banned by the generalized pattern"
    for name in ("plan_id", "addon_id", "doc_id", "ticket_id", "slot", "query"):
        assert not _is_identity_shaped(name), f"{name} should not be banned"


def _all_server_builders() -> dict[str, object]:
    # a template identity: schemas never depend on the ACTUAL customer id (account_server and
    # actions_server bind identity at connect, but every tool's advertised SHAPE is the same
    # regardless of which customer connected), so any fixed string exercises the real schema.
    return {
        "account": lambda: build_account_server("cust_schema_template"),
        "actions": lambda: build_actions_server("cust_schema_template"),
        "catalog": build_catalog_server,
        "knowledge": lambda: build_knowledge_server(InMemoryRetriever()),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("server_name", sorted(_all_server_builders()))
async def test_no_tool_schema_on_any_server_exposes_an_identity_shaped_parameter(server_name):
    server = _all_server_builders()[server_name]()
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        tools = (await client.list_tools()).tools
        for tool in tools:
            props = (tool.inputSchema or {}).get("properties") or {}
            leaked = [p for p in props if _is_identity_shaped(p)]
            assert not leaked, f"{server_name}.{tool.name} exposes identity shaped parameter(s): {leaked}"


@pytest.mark.asyncio
@pytest.mark.parametrize("server_name", sorted(_all_server_builders()))
async def test_every_tool_schema_on_any_server_sets_additional_properties_false(server_name):
    server = _all_server_builders()[server_name]()
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        tools = (await client.list_tools()).tools
        assert tools, f"{server_name} exposed no tools"
        for tool in tools:
            schema = tool.inputSchema or {}
            assert schema.get("additionalProperties") is False, (
                f"{server_name}.{tool.name} schema does not set additionalProperties: false"
            )


# --- SP4 task 5: the aggregated bind_tools surface (atlas.mcp_servers.tool_surface) --------------


def test_mcp_tool_surface_carries_every_tool_from_every_server_bind_tools_ready():
    from atlas.mcp_servers.tool_surface import mcp_tool_surface

    surface = mcp_tool_surface()
    expected_names = {
        "get_account_summary", "get_usage", "get_bill", "get_equipment", "list_tickets",
        "change_plan", "add_addon", "remove_addon", "reset_modem", "open_ticket", "book_engineer",
        "cancel_service",
        "list_plans", "get_plan", "compute_price", "check_eligibility",
        "search_knowledge",
    }
    assert set(surface) == expected_names
    for name, spec in surface.items():
        assert spec["name"] == name
        assert isinstance(spec["description"], str) and spec["description"]  # every tool is documented
        assert spec["parameters"].get("additionalProperties") is False  # the SAME hardened schema
        assert not _is_identity_shaped_any(spec["parameters"].get("properties") or {})


def _is_identity_shaped_any(properties: dict) -> bool:
    return any(_is_identity_shaped(p) for p in properties)
