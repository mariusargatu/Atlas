"""Seed the deterministic cassettes the Playwright E2E lane (and the SP4 task 2 persistence live
test) replay.

Two prompts drive the two flows the E2E asserts: the cold open (held at the render guard) and a
plan change (typed CONFIRM gate). A third, multi turn case backs
`testing/tests/test_persistence_live.py`: it proves a Postgres backed checkpoint survives a backend
container restart, which needs TWO turns on the SAME thread, both replayable. Run with `task
seed-e2e`. The output is committed so CI (and the backend Docker image, which COPYs this whole
directory) needs no recording step. Writes through the same `Cassette` schema and
`FileCassetteStore` the gateway reads, so the keys match at replay and the on disk shape can never
drift from what the gateway expects.
"""
from __future__ import annotations

import pathlib

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from replay.cassette import Cassette, build_request
from replay.cassette_store import FileCassetteStore

OUT = pathlib.Path("testing/harness/cassettes/e2e")
_MODEL_ID = "claude-test"

_CASES = [
    # cold open: a grounded but false answer, held by the render guard for the legacy customer
    (
        [HumanMessage("Is my plan contract-free?")],
        {"content": "Your plan is contract-free, no fee, cancel any time.", "tool_calls": []},
    ),
    # write: the agent proposes change_plan, and the confirmation interrupt pauses for a typed CONFIRM
    (
        [HumanMessage("Switch me to the fast plan")],
        {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]},
    ),
]

# SP4 task 2's persistence live test (cust_current, thread "persistence-restart-check"): two plain
# turns, no tool calls, so no guard trips and the render step ships the model's text verbatim. Each
# entry is (question, answer); the SECOND turn's cassette key must cover the WHOLE accumulated
# history (round 1's question + round 1's answer + round 2's question, per replay/cassette.py's
# build_request), which `_seed_persistence_turns` below reconstructs the same way the graph's own
# checkpointer would: this is the point of the test -- a 200 on round 2 is only possible if the
# checkpointer handed the running graph that exact accumulated history back after the restart.
_PERSISTENCE_TURNS = [
    ("What is your name?", "I'm Atlas, your broadband support assistant."),
    ("Can you say that again please?", "Sure, I'm Atlas, your broadband support assistant, glad to help again."),
]


def _seed_persistence_turns(store: FileCassetteStore) -> None:
    history: list[BaseMessage] = []
    for question, answer in _PERSISTENCE_TURNS:
        history = [*history, HumanMessage(question)]
        response = {"content": answer, "tool_calls": []}
        cassette = Cassette(model_id=_MODEL_ID, request=build_request(_MODEL_ID, history), response=response)
        store.save(cassette)
        print(f"wrote {OUT / f'{cassette.key}.json'}")
        history = [*history, AIMessage(answer)]


def main() -> None:
    store = FileCassetteStore(OUT)
    for messages, response in _CASES:
        cassette = Cassette(model_id=_MODEL_ID, request=build_request(_MODEL_ID, messages), response=response)
        store.save(cassette)
        print(f"wrote {OUT / f'{cassette.key}.json'}")
    _seed_persistence_turns(store)


if __name__ == "__main__":
    main()
