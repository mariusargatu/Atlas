"""P4: the confirmation interrupt through a real LangGraph graph + the deterministic
checkpointer. Asserts on content only (never an engine minted checkpoint id).
"""
from __future__ import annotations

import pytest
from langgraph.types import Command

from checkpointer import new_checkpointer
from determinism import IdFactory

from atlas.domain.actions import ActionsBackend
from atlas.orchestration.confirm_graph import build_confirm_graph


def _graph(backend):
    return build_confirm_graph(
        backend, IdFactory("idem"), "cust_current", "plan_current_fast", new_checkpointer()
    )


@pytest.mark.asyncio
async def test_pauses_at_the_gate_then_executes_on_typed_confirm():
    backend = ActionsBackend(IdFactory("ref"))
    graph = _graph(backend)
    cfg = {"configurable": {"thread_id": "t1"}}

    first = await graph.ainvoke({"messages": []}, cfg)
    assert "__interrupt__" in first  # paused at the confirmation gate, nothing applied yet
    assert backend.change_count("cust_current") == 0

    out = await graph.ainvoke(Command(resume="CONFIRM"), cfg)
    assert out["result"]["applied"] is True
    assert backend.change_count("cust_current") == 1


@pytest.mark.asyncio
async def test_a_bare_yes_is_rejected_and_nothing_applies():
    backend = ActionsBackend(IdFactory("ref"))
    graph = _graph(backend)
    cfg = {"configurable": {"thread_id": "t2"}}

    await graph.ainvoke({"messages": []}, cfg)
    out = await graph.ainvoke(Command(resume="yes"), cfg)
    assert "error" in out["result"]
    assert backend.change_count("cust_current") == 0
