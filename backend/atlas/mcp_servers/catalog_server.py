"""The catalog MCP server (read only, customer independent). `plan_id` is public catalog data,
not identity, so it is a legitimate tool argument. Results are canonically serialized.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from determinism.canonical import serialize_tool_result

from atlas.domain import catalog


def build_catalog_server() -> FastMCP:
    mcp = FastMCP("atlas-catalog")

    @mcp.tool()
    def list_plans() -> str:
        """The plans the provider currently sells."""
        return serialize_tool_result([{"id": p.id, "name": p.name} for p in catalog.CATALOG.values()])

    @mcp.tool()
    def get_plan(plan_id: str) -> str:
        """Plan details by id."""
        p = catalog.get_plan(plan_id)
        return serialize_tool_result(
            {"id": p.id, "name": p.name, "has_term": p.has_term, "early_termination_fee": p.early_termination_fee}
        )

    @mcp.tool()
    def compute_price(plan_id: str) -> str:
        """The monthly price for a plan. The catalog decides it in code, the model never does the arithmetic."""
        return serialize_tool_result({"plan_id": plan_id, "monthly_price": catalog.compute_price(plan_id)})

    @mcp.tool()
    def check_eligibility(plan_id: str) -> str:
        """Whether a plan can be newly taken: discontinued (legacy) plans cannot."""
        return serialize_tool_result({"plan_id": plan_id, "eligible": catalog.check_eligibility(plan_id)})

    return mcp
