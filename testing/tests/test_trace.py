"""The trace core is assertable, the target every later part reads from (principle 4).

A turn produces a deterministic span tree: tool spans hang under the agent span that requested
them, and guard verdicts are first class spans (our domain logic, annotated explicitly). This is
what trajectory (P4), simulation (P7), security (P8), and production (P9) assert against.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from checkpointer import new_checkpointer
from determinism import IdFactory
from gateway import GatewayChatModel
from tracing import InMemoryTracer

from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph

_FALSE_ANSWER = "Your plan is contract-free, no fee, cancel any time."


def _graph(cassette_dir, backend, tracer):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=cassette_dir, mode="replay")
    return build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer(), tracer=tracer)


def test_span_attributes_are_read_only_and_annotate_replaces():
    """A span's attributes are a read only view; annotate produces a NEW span, never mutates."""
    from tracing import InMemoryTracer, NullTracer

    t = InMemoryTracer()
    seq = t.open("pre_render_guard", "guard", ok=True)
    with pytest.raises(TypeError):
        t.spans[0].attributes["ok"] = False  # MappingProxyType: cannot mutate in place
    t.annotate(seq, reason="held")
    span = t.spans[0]
    assert span.attributes["ok"] is True and span.attributes["reason"] == "held"  # merged by replace
    assert NullTracer().annotate(seq, x=1) is None  # the no op adapter is a safe stand in


@pytest.mark.asyncio
async def test_read_path_emits_a_tool_span_under_the_agent(tmp_path, seed_cassette):
    tracer = InMemoryTracer()
    user = HumanMessage("What plan am I on?")
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [{"name": "get_account_summary", "args": {}, "id": "r1"}]})
    # second hop: agent answers from the tool result. The tool text must be the REAL account server
    # output (else the second cassette key misses), so derive it from the domain, not by hand.
    from langchain_core.messages import AIMessage, ToolMessage
    from canonical import serialize_tool_result
    from atlas.domain.accounts import get_account
    from atlas.domain.catalog import get_plan

    acct = get_account("cust_legacy_term")
    plan = get_plan(acct.plan_id)
    tool_text = serialize_tool_result({"customer": acct.name, "plan": plan.name, "has_contract": plan.has_term})
    ai = AIMessage(content="", tool_calls=[{"name": "get_account_summary", "args": {}, "id": "r1"}])
    tool_msg = ToolMessage(content=tool_text, tool_call_id="r1", name="get_account_summary")
    seed_cassette(tmp_path, [user, ai, tool_msg], {"content": "You are on the Value plan.", "tool_calls": []})

    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")), tracer)
    await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_legacy_term"}},
        {"configurable": {"thread_id": "tr1"}},
    )

    assert tracer.tool_order() == ["get_account_summary"]
    tool_span = tracer.of_kind("tool")[0]
    agent_spans = tracer.of_kind("llm")
    assert tool_span.parent in {s.seq for s in agent_spans}  # tool hangs under an agent span


@pytest.mark.asyncio
async def test_render_guard_verdict_is_a_span_on_the_cold_open(tmp_path, seed_cassette):
    tracer = InMemoryTracer()
    user = HumanMessage("Is my plan contract-free?")
    seed_cassette(tmp_path, [user], {"content": _FALSE_ANSWER, "tool_calls": []})
    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")), tracer)
    await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_legacy_term"}},
        {"configurable": {"thread_id": "tr2"}},
    )

    verdicts = tracer.guard_verdicts()
    render = [v for v in verdicts if v.name == "pre_render_guard"]
    assert render and render[0].attributes["ok"] is False  # the false answer was held, and the trace proves it


@pytest.mark.asyncio
async def test_spans_are_ordered_by_sequence_not_clock(tmp_path, seed_cassette):
    tracer = InMemoryTracer()
    user = HumanMessage("Is my plan contract-free?")
    seed_cassette(tmp_path, [user], {"content": _FALSE_ANSWER, "tool_calls": []})
    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")), tracer)
    await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_current"}},
        {"configurable": {"thread_id": "tr3"}},
    )

    seqs = [s.seq for s in tracer.spans]
    assert seqs == sorted(seqs)  # monotonic, deterministic order, never time keyed
