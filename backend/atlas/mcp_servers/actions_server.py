"""The actions MCP server (the write surface). Bound to the signed in customer at connect, so
`customer_id` is never a tool argument. The write tools only *propose*. The confirmation
interrupt in the graph is what executes them. That gate is upstream, not in the tool.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from determinism.canonical import serialize_tool_result


def build_actions_server(customer_id: str) -> FastMCP:
    mcp = FastMCP("atlas-actions")

    @mcp.tool()
    def change_plan(plan_id: str) -> str:
        """Propose changing the signed in customer's plan. Requires confirmation upstream."""
        return serialize_tool_result(
            {"action": "change_plan", "plan_id": plan_id, "customer": customer_id, "status": "proposed"}
        )

    @mcp.tool()
    def add_addon(addon_id: str) -> str:
        """Propose adding an add on to the signed in customer's service."""
        return serialize_tool_result(
            {"action": "add_addon", "addon_id": addon_id, "customer": customer_id, "status": "proposed"}
        )

    @mcp.tool()
    def remove_addon(addon_id: str) -> str:
        """Propose removing an add on from the signed in customer's service."""
        return serialize_tool_result(
            {"action": "remove_addon", "addon_id": addon_id, "customer": customer_id, "status": "proposed"}
        )

    @mcp.tool()
    def reset_modem() -> str:
        """Propose a modem reset for the signed in customer."""
        return serialize_tool_result({"action": "reset_modem", "customer": customer_id, "status": "proposed"})

    @mcp.tool()
    def open_ticket(subject: str) -> str:
        """Propose opening a support ticket for the signed in customer."""
        return serialize_tool_result(
            {"action": "open_ticket", "subject": subject, "customer": customer_id, "status": "proposed"}
        )

    @mcp.tool()
    def book_engineer(slot: str) -> str:
        """Propose booking an engineer visit for the signed in customer."""
        return serialize_tool_result(
            {"action": "book_engineer", "slot": slot, "customer": customer_id, "status": "proposed"}
        )

    return mcp
