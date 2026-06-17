"""The account MCP server (read only). Identity is bound from the session at connect time,
never exposed as a tool argument (principle 1 / ADR-027). In CI it is reached over the SDK's
in memory transport (hermetic, no subprocess); in dev/prod over Streamable HTTP with the
bearer token. The tool result is canonically serialized so the bytes are stable.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from canonical import serialize_tool_result

from atlas.domain import accounts
from atlas.domain.accounts import get_account
from atlas.domain.catalog import get_plan


def build_account_server(customer_id: str) -> FastMCP:
    """A per session server bound to the signed in customer. The model never sees the id; every
    read below is scoped to `customer_id` from the session, so no tool can fetch another account.
    """
    mcp = FastMCP("atlas-account")

    @mcp.tool()
    def get_account_summary() -> str:
        """The signed in customer's account summary."""
        account = get_account(customer_id)
        plan = get_plan(account.plan_id)
        return serialize_tool_result(
            {"customer": account.name, "plan": plan.name, "has_contract": plan.has_term}
        )

    @mcp.tool()
    def get_usage() -> str:
        """The signed in customer's data usage this billing month."""
        u = accounts.get_usage(customer_id)
        return serialize_tool_result(
            {"period": u.period, "gigabytes_used": u.gigabytes_used, "data_cap_gb": u.data_cap_gb}
        )

    @mcp.tool()
    def get_bill() -> str:
        """The signed in customer's current bill."""
        b = accounts.get_bill(customer_id)
        return serialize_tool_result(
            {"period": b.period, "amount": b.amount, "due_date": b.due_date, "paid": b.paid}
        )

    @mcp.tool()
    def get_equipment() -> str:
        """The equipment on file for the signed in customer."""
        return serialize_tool_result(
            [{"kind": e.kind, "model": e.model, "serial": e.serial} for e in accounts.get_equipment(customer_id)]
        )

    @mcp.tool()
    def list_tickets() -> str:
        """The signed in customer's open support tickets."""
        return serialize_tool_result(
            [{"ticket_id": t.ticket_id, "subject": t.subject, "status": t.status} for t in accounts.list_tickets(customer_id)]
        )

    return mcp
