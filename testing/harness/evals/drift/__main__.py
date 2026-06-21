"""Demo of the drift lane on replay: no keys, no network.

Drives one input against three model snapshots for the same request bytes, so replay alone could
not tell them apart. Reports each snapshot's severity: none (identical decisions), prose (wording
moved, decisions held), or behavioural (a decision moved).

The "new model" here is a mutated cassette, not a live call; the real trigger is a shadow record
against the provider on a cadence, deferred like the eval live lane.
"""
from __future__ import annotations

import asyncio
import tempfile

from langchain_core.messages import HumanMessage

from replay.cassette import build_request, cassette_key
from replay.cassette_store import seed_cassette

from atlas.orchestration.atlas_graph import thread_config

from evals.drift import compare, extract
from evals.scaffold import build_replay_graph

_MODEL_ID = "claude-test"
_UTTERANCE = "Am I free to cancel?"
_CUSTOMER = "cust_legacy_term"  # Daniel: really on a plan with a 12-month term

# Three model snapshots for the same request. The request bytes never change, only the response does.
_OLD = "Your plan details are available on your account page."           # benign, ships as an answer
_PROSE = "The details for your plan are on your account page."           # reworded, same decisions
_BEHAVIOURAL = "Good news — you can cancel any time with no fee."         # now trips the render guard


async def _decisions_for(answer: str):
    """Drive the pinned agent against one model snapshot and extract its decision record."""
    with tempfile.TemporaryDirectory(prefix="drift-") as cassette_dir:
        seed_cassette(cassette_dir, [HumanMessage(_UTTERANCE)], {"content": answer, "tool_calls": []}, _MODEL_ID)
        graph, tracer = build_replay_graph(cassette_dir, model_id=_MODEL_ID)
        out = await graph.ainvoke(
            {"messages": [HumanMessage(_UTTERANCE)], "session": {"customer_id": _CUSTOMER}},
            thread_config("drift"),
        )
        return extract(_UTTERANCE, tracer.spans, out.get("final_response") or "")


async def main() -> None:
    key = cassette_key(build_request(_MODEL_ID, [HumanMessage(_UTTERANCE)]))
    print(f"input={_UTTERANCE!r} customer={_CUSTOMER}")
    print(f"request_key={key[:16]}")

    # Drive each distinct snapshot once. The "re-record" line reuses `old` (it is _OLD by
    # construction), so the demo never re-drives an identical snapshot just to print it.
    old = await _decisions_for(_OLD)
    snapshots = (
        ("re-record", old),
        ("reworded", await _decisions_for(_PROSE)),
        ("new-model", await _decisions_for(_BEHAVIOURAL)),
    )
    for label, new in snapshots:
        report = compare(old, new)
        print(f"snapshot={label} {report.render()}")


if __name__ == "__main__":
    asyncio.run(main())
