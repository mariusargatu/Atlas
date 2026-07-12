"""The full tool calling turn: agent -> MCP tool -> agent -> answer (Spike A, assembled).

The agent (gateway, replayed) decides to read the account. The tools node calls the account
MCP server over the in memory transport and appends the **canonically serialized** result. The
agent runs again over the new messages and answers. The whole chain replays byte stable
precisely because the tool result is canonical, so the second agent cassette key is stable.
Identity (`customer_id`) lives in a non model `session` channel, never in a tool argument.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from mcp.shared.memory import create_connected_server_and_client_session

from atlas.mcp_servers.account_server import build_account_server


class TurnState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session: dict  # {customer_id}, non model channel, identity from the session, never a tool argument


async def _mcp_call(customer_id: str, tool_name: str, args: dict) -> str:
    """Call an account tool over the in memory MCP transport and return the canonical text."""
    server = build_account_server(customer_id)
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        result = await client.call_tool(tool_name, args or {})
        return result.content[0].text


def build_turn_graph(model: BaseChatModel):
    async def agent(state: TurnState) -> dict:
        result = await model._agenerate(list(state["messages"]))
        return {"messages": [result.generations[0].message]}

    async def tools(state: TurnState) -> dict:
        last = state["messages"][-1]
        customer_id = state["session"]["customer_id"]
        out: list[BaseMessage] = []
        for tc in last.tool_calls:
            text = await _mcp_call(customer_id, tc["name"], tc.get("args", {}))
            out.append(ToolMessage(content=text, tool_call_id=tc["id"], name=tc["name"]))
        return {"messages": out}

    def route(state: TurnState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    g = StateGraph(TurnState)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "end": END})
    g.add_edge("tools", "agent")
    return g.compile()
