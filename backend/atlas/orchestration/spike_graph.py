"""A minimal LangGraph spike: prove the gateway replays deterministically inside a graph.

The smallest graph that puts the record/replay model node inside LangGraph, isolating the
gateway x LangGraph question the reviewers flagged as central. The deterministic
checkpointer and the MCP tools come next; this proves the model node alone is pinned.
"""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class SpikeState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(model: BaseChatModel):
    """A graph of exactly one nondeterministic node (the gateway backed agent)."""

    async def agent(state: SpikeState) -> dict:
        result = await model._agenerate(list(state["messages"]))
        return {"messages": [result.generations[0].message]}

    g = StateGraph(SpikeState)
    g.add_node("agent", agent)
    g.add_edge(START, "agent")
    g.add_edge("agent", END)
    return g.compile()
