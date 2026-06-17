"""Schema hardening shared by every MCP server (D11): `additionalProperties: false` set explicitly
on every advertised tool schema.

FastMCP's own `@mcp.tool()` decorator builds each tool's `inputSchema` via its own Pydantic
introspection (`mcp.server.fastmcp.utilities.func_metadata.func_metadata`) and never opts into
Pydantic's `extra="forbid"` config, so nothing in the generated schema stops a client from sending
an unexpected extra property; the decorator itself exposes no keyword for one either (confirmed by
reading `FastMCP.tool()`'s full signature: name, title, description, annotations, icons, meta,
structured_output, nothing schema shaped). This module is the fix at the one seam that DOES reach
the generated schema after the fact: `Tool.parameters`, the plain dict FastMCP stores per
registered tool (`mcp.server.fastmcp.tools.base.Tool`), set once, immediately after every
`build_*_server()` call, before any client ever lists tools.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def harden_tool_schemas(mcp: FastMCP) -> FastMCP:
    """Set `additionalProperties: false` on every tool `mcp` has registered so far, in place, and
    return the SAME instance, so a `build_*_server` function can end with
    `return harden_tool_schemas(mcp)`. Idempotent: a schema that already sets the key (none do
    today, but a future tool might via an explicit `model_config`) is left untouched rather than
    overwritten. `Tool.parameters` is a plain dict field on a pydantic model, not itself frozen, so
    reassigning it (a new dict, never a mutated one) is the only mutation this module performs, and
    it happens once, at server construction, never per request."""
    for tool in mcp._tool_manager.list_tools():
        if "additionalProperties" not in tool.parameters:
            tool.parameters = {**tool.parameters, "additionalProperties": False}
    return mcp
