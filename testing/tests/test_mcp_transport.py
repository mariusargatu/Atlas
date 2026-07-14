"""The in memory MCP transport: a tool call over the real MCP protocol, in process,
hermetic. Proves identity is out of band (no `customer_id` in the schema) and the result
serializes canonically.
"""
from __future__ import annotations

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from atlas.mcp_servers.account_server import build_account_server


@pytest.mark.asyncio
async def test_account_tool_over_in_memory_transport_reads_session_identity():
    server = build_account_server("cust_legacy_term")  # bound from the session at connect
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        listed = await client.list_tools()
        tool = next(t for t in listed.tools if t.name == "get_account_summary")
        props = (tool.inputSchema or {}).get("properties") or {}
        assert "customer_id" not in props  # identity is never a tool argument
        # D11, over the real protocol round trip (not just server construction): every advertised
        # schema sets additionalProperties false. The general, cross server proof (name pattern +
        # allowlist) lives in test_mcp_servers.py; this is the same assertion over THIS transport.
        assert (tool.inputSchema or {}).get("additionalProperties") is False

        result = await client.call_tool("get_account_summary", {})
        payload = json.loads(result.content[0].text)
        assert payload["customer"] == "Daniel"
        assert payload["has_contract"] is True  # the legacy plan's truth, delivered over MCP
