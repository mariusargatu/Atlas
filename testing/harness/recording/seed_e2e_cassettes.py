"""Seed the deterministic cassettes the Playwright E2E lane replays.

Two prompts drive the two flows the E2E asserts: the cold open (held at the render guard) and a
plan change (typed CONFIRM gate). Run with `task seed-e2e`; the output is committed so CI needs no
recording step. Writes through the same `Cassette` schema and `FileCassetteStore` the gateway reads,
so the keys match at replay and the on disk shape can never drift from what the gateway expects.
"""
from __future__ import annotations

import pathlib

from langchain_core.messages import HumanMessage

from replay.cassette import Cassette, build_request
from replay.cassette_store import FileCassetteStore

OUT = pathlib.Path("testing/harness/cassettes/e2e")
_MODEL_ID = "claude-test"

_CASES = [
    # cold open: a grounded but false answer; the render guard holds it for the legacy customer
    (
        [HumanMessage("Is my plan contract-free?")],
        {"content": "Your plan is contract-free, no fee, cancel any time.", "tool_calls": []},
    ),
    # write: the agent proposes change_plan; the confirmation interrupt pauses for a typed CONFIRM
    (
        [HumanMessage("Switch me to the fast plan")],
        {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]},
    ),
]


def main() -> None:
    store = FileCassetteStore(OUT)
    for messages, response in _CASES:
        cassette = Cassette(model_id=_MODEL_ID, request=build_request(_MODEL_ID, messages), response=response)
        store.save(cassette)
        print(f"wrote {OUT / f'{cassette.key}.json'}")


if __name__ == "__main__":
    main()
