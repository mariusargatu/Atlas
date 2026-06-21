"""The drift lane: catch response drift behind a stable request key.

Replay pins a proxy of the model and never re-checks it. When the live model moves but the request
bytes stay identical, the suite stays green on a stale cassette. The drift lane re-runs the pinned
agent on a new snapshot and diffs the decisions (intent, tools, guards, outcome), separating
behavioural drift from mere prose drift. Tested hermetically by mutating a cassette, no live call.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from determinism.canonical import canonical, digest
from tracing import Span

from evals.drift.compare import compare
from evals.drift.record import DecisionRecord, extract
from evals.scaffold import build_replay_graph

_MODEL_ID = "claude-test"


async def _drive_decisions(cassette_dir, utterance, customer_id):
    """Drive one turn on replay and extract its decision record (the drift lane's read of a snapshot)."""
    graph, tracer = build_replay_graph(cassette_dir, model_id=_MODEL_ID)
    out = await graph.ainvoke(
        {"messages": [HumanMessage(utterance)], "session": {"customer_id": customer_id}},
        {"configurable": {"thread_id": "t"}},
    )
    return extract(utterance, tracer.spans, out.get("final_response") or "")


def _record(intent="action", tools=(), guards=(), outcome="answer", claim="x"):
    return DecisionRecord(intent=intent, tools=tuple(tools), guards=tuple(guards),
                          outcome=outcome, claim_digest=digest(claim))


# ---- extraction from a real driven turn ----

@pytest.mark.asyncio
async def test_extract_reads_decisions_from_a_driven_turn(tmp_path, seed_cassette):
    # A benign answer ships: no tools, the render guard passes, the outcome is an answer.
    seed_cassette(tmp_path, [HumanMessage("Am I free to cancel?")],
                  {"content": "Your plan details are on your account page.", "tool_calls": []}, _MODEL_ID)
    rec = await _drive_decisions(tmp_path, "Am I free to cancel?", "cust_legacy_term")
    assert rec.intent == "action"                       # "cancel" is an action cue, deterministic
    assert rec.tools == ()
    assert ("pre_render_guard", True) in rec.guards
    assert rec.outcome == "answer"


# ---- compare: behavioural vs prose vs none ----

def test_compare_identical_records_is_no_drift():
    rec = _record(guards=(("pre_render_guard", True),))
    report = compare(rec, rec)
    assert report.severity() == "none" and report.changed_decisions == ()


def test_compare_prose_only_change_is_prose_drift():
    old = _record(claim="your plan details are on your account page")
    new = _record(claim="the details for your plan are on your account page")
    report = compare(old, new)
    assert report.severity() == "prose" and report.changed_decisions == () and report.prose_changed


def test_compare_changed_outcome_and_guard_is_behavioural_drift():
    old = _record(guards=(("pre_render_guard", True),), outcome="answer")
    new = _record(guards=(("pre_render_guard", False),), outcome="handoff")
    report = compare(old, new)
    assert report.severity() == "behavioural"
    assert set(report.changed_decisions) == {"guards", "outcome"}


def test_compare_changed_tools_is_behavioural_drift():
    old = _record(tools=())
    new = _record(tools=(("get_account_summary", None),))
    report = compare(old, new)
    assert report.severity() == "behavioural" and report.changed_decisions == ("tools",)


def test_same_write_tool_with_different_args_is_behavioural_drift():
    # change_plan to a different valid plan is a different decision, not a reword. A name-only
    # trajectory reads these as identical; carrying the write args catches the move.
    old = _record(tools=(("change_plan", canonical({"plan_id": "plan_current_fast"})),))
    new = _record(tools=(("change_plan", canonical({"plan_id": "plan_legacy_value"})),))
    report = compare(old, new)
    assert report.severity() == "behavioural" and report.changed_decisions == ("tools",)


def test_same_write_tool_with_equal_args_is_not_drift():
    # Value-equal (canonicalized) write args must not read as a change, or every rerun is false drift.
    old = _record(tools=(("change_plan", canonical({"plan_id": "plan_current_fast"})),))
    new = _record(tools=(("change_plan", canonical({"plan_id": "plan_current_fast"})),))
    assert compare(old, new).severity() == "none"


def test_render_speaks_each_severity():
    none = compare(_record(), _record())
    assert none.render() == "drift=none"
    prose = compare(_record(claim="a"), _record(claim="b"))
    assert prose.render() == "drift=prose"
    behavioural = compare(_record(outcome="answer"), _record(outcome="handoff"))
    line = behavioural.render()
    assert line.startswith("drift=behavioural") and "outcome=" in line and "handoff" in line


def test_extract_reads_outcome_from_spans_not_prose():
    # The outcome comes from the span tree (an applied execute_action), never from the prose. A
    # benign answer whose text merely mentions "your reference is" is not a write.
    applied = [Span(seq=1, name="execute_action", kind="node", parent=None,
                    attributes={"applied": True, "reference": "ref-000001"})]
    assert extract("Change my plan", applied, "Done. Your reference is ref-000001.").outcome == "write-applied"

    # Same prose, but no execute_action span -> it is an answer, not a write (the false positive the
    # prose parsing approach would have made).
    benign = extract("Where's my ref?", [], "Your reference is printed on your latest bill.")
    assert benign.outcome == "answer"

    # A guard that failed closed (no execute_action) is a handoff, read structurally.
    refused = [Span(seq=1, name="pre_render_guard", kind="guard", parent=None, attributes={"ok": False})]
    assert extract("Am I free to cancel?", refused, "anything").outcome == "handoff"

    # An execute_action that did not apply (a refused confirmation) is also a handoff, not a write.
    not_applied = [Span(seq=1, name="execute_action", kind="node", parent=None, attributes={"applied": False})]
    assert extract("Change my plan", not_applied, "anything").outcome == "handoff"


# ---- end to end: a mutated cassette, identical request, behavioural drift caught ----

@pytest.mark.asyncio
async def test_behavioural_drift_caught_though_request_is_identical(tmp_path, seed_cassette):
    utterance = "Am I free to cancel?"
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    # Same request bytes. The only difference is the response (the proxy that silently drifted).
    seed_cassette(old_dir, [HumanMessage(utterance)],
                  {"content": "Your plan details are on your account page.", "tool_calls": []}, _MODEL_ID)
    seed_cassette(new_dir, [HumanMessage(utterance)],
                  {"content": "Good news — you can cancel any time with no fee.", "tool_calls": []}, _MODEL_ID)

    old = await _drive_decisions(old_dir, utterance, "cust_legacy_term")
    new = await _drive_decisions(new_dir, utterance, "cust_legacy_term")
    report = compare(old, new)

    assert report.severity() == "behavioural"           # the new model trips the render guard
    assert "outcome" in report.changed_decisions and new.outcome == "handoff"
