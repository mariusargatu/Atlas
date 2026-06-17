"""Spike A (model node): the gateway replays deterministically inside a LangGraph graph.

Hermetic: the cassette stands in for the model, no network. Proves a graph run is
reproducible and that the recorded answer flows through the graph's state.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from atlas.orchestration.spike_graph import build_graph
from gateway import GatewayChatModel

_ANSWER = "Your account is on a legacy plan with a 12-month term."


@pytest.mark.asyncio
async def test_gateway_replays_deterministically_inside_langgraph(tmp_path, seed_cassette):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    messages = [HumanMessage("Is my plan contract-free?")]
    seed_cassette(tmp_path, messages, {"content": _ANSWER, "tool_calls": []})
    graph = build_graph(gw)

    out1 = await graph.ainvoke({"messages": list(messages)})
    out2 = await graph.ainvoke({"messages": list(messages)})

    assert out1["messages"][-1].content == _ANSWER
    assert out1["messages"][-1].content == out2["messages"][-1].content
