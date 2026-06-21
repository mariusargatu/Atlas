"""Record a REAL Atlas turn against a live provider (default: local Ollama), committed for replay.

The cold open answer path: feed a real model the (current, term free) plan page and the customer's
contract question. A confident model answers "contract free, no fee", grounded in the page, TRUE
for a current customer, FALSE for the legacy customer whose term lives in the account. The render
guard catches it against the oracle. This is the north star with a real model in the loop, captured
once and replayed forever (Ollama is not byte stable live: replay is the determinism, not Ollama).

Run: `task record-atlas` (needs an Ollama daemon + the model pulled). Output is committed under
testing/harness/cassettes/atlas/ and replayed by tests/test_recorded_turns.py.
"""
from __future__ import annotations

import asyncio
import pathlib

from langchain_core.messages import HumanMessage, SystemMessage

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from replay.gateway import GatewayChatModel
from replay.providers import build_chat_model, provider_tag

from atlas.domain.actions import ActionsBackend
from atlas.domain.corpus import CORPUS
from atlas.orchestration.atlas_graph import build_atlas_graph

OUT = pathlib.Path("testing/harness/cassettes/atlas")
_PAGE = next(c.text for c in CORPUS if c.doc_id == "plan-current-page")
_SYSTEM = (
    "You are Atlas, a broadband support agent. Answer the customer's question directly and "
    "confidently using only the plan page provided. State plainly whether the plan has a contract "
    "or minimum term and whether there is any cancellation or early-termination fee."
)


def cold_open_messages() -> list:
    """The exact message list both the recorder and the replay test use (keys must match byte for byte)."""
    return [
        SystemMessage(_SYSTEM),
        HumanMessage(f"Plan page:\n{_PAGE}\n\nQuestion: Is my plan contract-free, and can I cancel with no fee?"),
    ]


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tag = provider_tag()
    gateway = GatewayChatModel(model_id=tag, cassette_dir=OUT, mode="record", inner=build_chat_model())
    graph = build_atlas_graph(gateway, IdFactory("idem"), ActionsBackend(IdFactory("ref")), new_checkpointer())

    out = await graph.ainvoke(
        {"messages": cold_open_messages(), "session": {"customer_id": "cust_legacy_term"}},
        {"configurable": {"thread_id": "rec-cold-open"}},
    )
    held = out["final_response"].startswith("[safe handoff]")
    print(f"[record] provider={tag}")
    print(f"[record] render guard held the answer for the legacy customer: {held}")
    print(f"[record] final_response: {out['final_response'][:160]!r}")
    print(f"[record] cassettes: {[p.name for p in OUT.glob('*.json')]}")
    if not held:
        print("[record] NOTE: the model did not make a no-contract claim this run — re-run or adjust the prompt.")


if __name__ == "__main__":
    asyncio.run(main())
