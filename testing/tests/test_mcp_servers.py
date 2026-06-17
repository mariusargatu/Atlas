"""The four MCP servers exist with the right tools, and identity is never in a tool schema."""
from __future__ import annotations

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from atlas.mcp_servers.account_server import build_account_server
from atlas.mcp_servers.actions_server import build_actions_server
from atlas.mcp_servers.catalog_server import build_catalog_server


@pytest.mark.asyncio
async def test_catalog_server_returns_plan_details():
    server = build_catalog_server()
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        result = await client.call_tool("get_plan", {"plan_id": "plan_legacy_value"})
        payload = json.loads(result.content[0].text)
        assert payload["has_term"] is True
        assert payload["early_termination_fee"] == "D:120"  # money is value normalized, canonical


@pytest.mark.asyncio
async def test_catalog_server_computes_price_and_eligibility_in_code():
    server = build_catalog_server()
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"compute_price", "check_eligibility"} <= names  # the catalog decides, not the model
        price = json.loads((await client.call_tool("compute_price", {"plan_id": "plan_current_fast"})).content[0].text)
        assert price["monthly_price"] == "D:35"  # money is value normalized, canonical
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
        for tool in tools:  # identity is bound from the session, never a tool arg (principle 1)
            assert "customer_id" not in ((tool.inputSchema or {}).get("properties") or {})


@pytest.mark.asyncio
async def test_account_reads_are_scoped_to_the_session_customer():
    # Daniel (legacy, over cap, open ticket, owes 39) vs Sarah (current, paid, no open ticket)
    async with create_connected_server_and_client_session(build_account_server("cust_legacy_term")._mcp_server) as daniel:
        usage = json.loads((await daniel.call_tool("get_usage", {})).content[0].text)
        bill = json.loads((await daniel.call_tool("get_bill", {})).content[0].text)
        equip = json.loads((await daniel.call_tool("get_equipment", {})).content[0].text)
        tickets = json.loads((await daniel.call_tool("list_tickets", {})).content[0].text)
        assert usage["gigabytes_used"] == "D:512" and usage["data_cap_gb"] == 500  # over the cap
        assert bill["amount"] == "D:39" and bill["paid"] is False
        assert equip[0]["serial"] == "VB-7777"
        assert [t["ticket_id"] for t in tickets] == ["tk-2002"]  # the one OPEN ticket, closed ones filtered

    async with create_connected_server_and_client_session(build_account_server("cust_current")._mcp_server) as sarah:
        bill = json.loads((await sarah.call_tool("get_bill", {})).content[0].text)
        tickets = json.loads((await sarah.call_tool("list_tickets", {})).content[0].text)
        assert bill["amount"] == "D:35"  # a different customer, a different bill: scope holds
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
