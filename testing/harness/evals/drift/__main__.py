"""Runnable demo of the drift lane on the REPLAY lane: zero keys, zero egress.

`task drift` runs this. It takes ONE input the customer could send and shows three model snapshots
for it — the request bytes identical every time, so REPLAY would happily return any of them and the
suite would stay green. The drift lane re-runs the pinned agent on each snapshot and diffs the
DECISIONS against the committed one:

  none         the new snapshot decides exactly as the old one  (green that is actually green)
  prose        the wording moved, the decisions held            (low signal)
  behavioural  the new snapshot trips a different guard / changes the outcome  (the silent move)

The "new model" here is a deliberately mutated cassette, not a live call. The real trigger is a
shadow RECORD against the provider on a cadence; that needs keys + the `record` group and is
deferred, exactly like the eval LIVE lane.
"""
from __future__ import annotations

import asyncio
import tempfile

from langchain_core.messages import HumanMessage

from replay.cassette import build_request, cassette_key
from replay.cassette_store import seed_cassette

from evals.drift import compare, extract
from evals.scaffold import build_replay_graph

_MODEL_ID = "claude-test"
_UTTERANCE = "Am I free to cancel?"
_CUSTOMER = "cust_legacy_term"  # Daniel: really on a plan with a 12-month term

# Three model snapshots for the SAME request. The request bytes never change; only the response does.
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
            {"configurable": {"thread_id": "drift"}},
        )
        return extract(_UTTERANCE, tracer.spans, out.get("final_response") or "")


async def main() -> None:
    key = cassette_key(build_request(_MODEL_ID, [HumanMessage(_UTTERANCE)]))
    print(f'input: "{_UTTERANCE}"  (customer {_CUSTOMER})')
    print(f"request key: {key[:16]}…  — identical for every snapshot below; REPLAY can't tell them apart\n")

    # Drive each distinct snapshot once. The "re-record (same)" line reuses `old` (it is _OLD by
    # construction), so the demo never re-drives an identical snapshot just to print it.
    old = await _decisions_for(_OLD)
    snapshots = (
        ("re-record (same)", old),
        ("reworded", await _decisions_for(_PROSE)),
        ("new model", await _decisions_for(_BEHAVIOURAL)),
    )
    for label, new in snapshots:
        report = compare(old, new)
        print(f"  [{label:16}] {report.render()}")

    print("\nThe request never changed. The cassette would replay green. The decision diff is the only")
    print("thing that would have told you the model moved.")


if __name__ == "__main__":
    asyncio.run(main())
