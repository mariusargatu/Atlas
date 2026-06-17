"""The full MCP tool surface as LangChain `bind_tools` compatible dicts (SP4 task 5).

One static structure, built once (at graph construction, never per turn) from every server's OWN
advertised, hardened schema (`hardening.harden_tool_schemas`, additionalProperties false included)
so bind_tools sees exactly what `contracts/mcp_snapshots/*.json` pins, never a hand kept second
copy. Building it needs no live session identity: every tool's SCHEMA is identity independent even
on the two per session servers, account and actions (only CALLING the tool needs an identity, not
describing it), so a fixed template id stands in for the real one.

Deliberately synchronous, no `asyncio.run`: `FastMCP.list_tools()` is declared `async def` but never
actually awaits anything (it only wraps `self._tool_manager.list_tools()`, a plain synchronous call,
into `MCPTool` objects), so this reads the SAME underlying `Tool` records
(`mcp.server.fastmcp.tools.base.Tool`, name/description/parameters) directly off the tool manager,
skipping the pointless coroutine wrapper. This matters here specifically: a caller from INSIDE an
already running event loop (a test, or a future async server startup path) cannot call
`asyncio.run()` at all (`RuntimeError: asyncio.run() cannot be called from a running event loop`);
staying synchronous sidesteps that entirely rather than requiring every caller to know whether it is
itself inside a loop.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.mcp_servers.account_server import build_account_server
from atlas.mcp_servers.actions_server import build_actions_server
from atlas.mcp_servers.catalog_server import build_catalog_server
from atlas.mcp_servers.knowledge_server import build_knowledge_server

# Any fixed string works (see the module docstring); this one is never a real customer id, so a
# reader who greps for it never mistakes it for live account data.
_SCHEMA_TEMPLATE_CUSTOMER_ID = "cust_schema_template"


def _to_binding_tool(mcp: FastMCP) -> list[dict]:
    """The plain `{"name", "description", "parameters"}` shape `BaseChatModel.bind_tools` accepts
    from every provider integration (`langchain_core.utils.function_calling.convert_to_openai_tool`
    recognizes it directly, and each provider's own `bind_tools` converts it again to its native format
    from there), so this stays provider agnostic: nothing here imports `langchain_anthropic` /
    `langchain_openai` / `langchain_ollama`, none of which are installed in the hermetic lane."""
    return [
        {"name": tool.name, "description": tool.description or "", "parameters": tool.parameters}
        for tool in mcp._tool_manager.list_tools()
    ]


def mcp_tool_surface(retriever=None) -> dict[str, dict]:
    """Every tool from every server (knowledge, account, catalog, actions), keyed by name, as a
    bind_tools compatible dict. `retriever` defaults to `InMemoryRetriever()`: knowledge_server's
    OWN schema never depends on which retriever backs it (only `search_knowledge`'s `query: str`
    parameter is advertised either way), so the default is just for callers with no retriever of
    their own (a script, a test); a caller that already has one (the graph) should pass it, so this
    never constructs a second one needlessly."""
    servers = (
        build_account_server(_SCHEMA_TEMPLATE_CUSTOMER_ID),
        build_actions_server(_SCHEMA_TEMPLATE_CUSTOMER_ID),
        build_catalog_server(),
        build_knowledge_server(retriever or InMemoryRetriever()),
    )
    out: dict[str, dict] = {}
    for mcp in servers:
        for spec in _to_binding_tool(mcp):
            out[spec["name"]] = spec
    return out
