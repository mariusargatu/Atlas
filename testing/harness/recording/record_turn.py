"""Prove the record → replay loop against a live provider (default: local Ollama).

1. RECORD: wrap the chosen provider in the gateway, run one real turn, save a cassette.
2. REPLAY: rebuild the gateway in replay mode (no provider), run the same turn, assert byte equal.

This is the loop the whole suite rests on: a live model (Ollama here) is captured once, and every CI run
afterwards replays the cassette with zero egress. Ollama is not byte stable live, which is exactly
why we replay. Run with `task record-demo` (needs a running Ollama daemon + a pulled chat model).
"""
from __future__ import annotations

import asyncio
import tempfile

from langchain_core.messages import HumanMessage

from replay.gateway import GatewayChatModel
from replay.providers import build_chat_model, provider_tag

from atlas.orchestration.spike_graph import build_graph


async def main() -> None:
    cassettes = tempfile.mkdtemp(prefix="record-demo-")
    tag = provider_tag()
    prompt = [HumanMessage("In one sentence: what is a broadband data cap?")]

    # 1. RECORD: live provider through the gateway
    rec = GatewayChatModel(model_id=tag, cassette_dir=cassettes, mode="record", inner=build_chat_model())
    recorded = await build_graph(rec).ainvoke({"messages": prompt})
    text = recorded["messages"][-1].content
    print(f"[record] {tag} → {text[:120]!r}")

    # 2. REPLAY: no provider, cassette only
    rep = GatewayChatModel(model_id=tag, cassette_dir=cassettes, mode="replay")
    replayed = await build_graph(rep).ainvoke({"messages": prompt})

    assert replayed["messages"][-1].content == text, "replay diverged from the recorded cassette"
    print("[replay] byte-identical to the recording ✓  (this is what CI runs — zero egress)")


if __name__ == "__main__":
    asyncio.run(main())
