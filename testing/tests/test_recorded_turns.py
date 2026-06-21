"""The cold open with a REAL model in the loop, recorded once against Ollama, replayed hermetically.

Unlike the crafted guard unit fixtures (test_atlas_graph), this turn's assistant answer is genuine
qwen2.5:7b output, grounded in the term free plan page. It is TRUE for a current customer and FALSE
for the legacy customer, and the oracle/render guard tells them apart. Replays byte stable with no
Ollama, no network (the cassette under testing/harness/cassettes/atlas/ is committed). Re record with
`task record-atlas`.
"""
from __future__ import annotations

import pytest

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from replay.gateway import GatewayChatModel
from recording.record_atlas_cassettes import OUT, cold_open_messages

from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph

_RECORDED_MODEL = "ollama:qwen2.5:7b"  # must match the committed cassette's request model_id


def _graph():
    gateway = GatewayChatModel(model_id=_RECORDED_MODEL, cassette_dir=OUT, mode="replay")
    return build_atlas_graph(gateway, IdFactory("idem"), ActionsBackend(IdFactory("ref")), new_checkpointer())


@pytest.mark.asyncio
async def test_recorded_cold_open_held_for_legacy_customer():
    out = await _graph().ainvoke(
        {"messages": cold_open_messages(), "session": {"customer_id": "cust_legacy_term"}},
        {"configurable": {"thread_id": "replay-legacy"}},
    )
    # the real model said "contract-free, no fee", false for Daniel, held at the render guard
    assert out["final_response"].startswith("[safe handoff]")


@pytest.mark.asyncio
async def test_recorded_answer_renders_for_current_customer():
    out = await _graph().ainvoke(
        {"messages": cold_open_messages(), "session": {"customer_id": "cust_current"}},
        {"configurable": {"thread_id": "replay-current"}},
    )
    # same answer, true for Sarah → it renders (the cold open's point: grounded ≠ true)
    assert "contract-free" in out["final_response"]
