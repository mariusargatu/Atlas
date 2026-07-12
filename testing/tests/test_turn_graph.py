"""Spike A, complete: a full tool calling turn replays byte stable.

agent (decides to read the account) -> account MCP tool (in memory) -> agent (answers from the
account). The second agent cassette is keyed by messages that include the tool result, so this
only replays if the tool result is canonically stable, the byte stability chain the reviewers
flagged, proven end to end.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from determinism.canonical import serialize_tool_result
from replay.gateway import GatewayChatModel

from atlas.orchestration.turn_graph import build_turn_graph

_TOOLCALL = [{"name": "get_account_summary", "args": {}, "id": "call-1"}]
_FINAL = "Your account is on a legacy plan with a 12-month term, so leaving early carries a fee."


@pytest.mark.asyncio
async def test_full_tool_calling_turn_replays_byte_stable(tmp_path, seed_cassette):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    user = HumanMessage("Is my plan contract-free?")

    # cassette 1: the agent decides to read the account
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": _TOOLCALL})

    # the canonical tool result the MCP server produces for Daniel (legacy plan)
    tool_text = serialize_tool_result(
        {"customer": "Daniel", "plan": "Value (legacy, discontinued)", "has_contract": True}
    )
    ai_toolcall = AIMessage(content="", tool_calls=_TOOLCALL)
    tool_msg = ToolMessage(content=tool_text, tool_call_id="call-1", name="get_account_summary")

    # cassette 2: after the tool result, the agent answers truthfully from the account
    seed_cassette(tmp_path, [user, ai_toolcall, tool_msg], {"content": _FINAL, "tool_calls": []})

    graph = build_turn_graph(gw)
    start = {"messages": [user], "session": {"customer_id": "cust_legacy_term"}}

    out1 = await graph.ainvoke(dict(start))
    out2 = await graph.ainvoke(dict(start))

    assert out1["messages"][-1].content == _FINAL
    assert out1["messages"][-1].content == out2["messages"][-1].content  # byte stable across runs
    assert any(isinstance(m, ToolMessage) for m in out1["messages"])     # really went agent->tool->agent
