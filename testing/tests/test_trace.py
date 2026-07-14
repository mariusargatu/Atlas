"""The trace core is assertable, the target every later part reads from.

A turn produces a deterministic span tree: tool spans hang under the agent span that requested
them, and guard verdicts are first class spans (our domain logic, annotated explicitly). This is
what trajectory, simulation, security, and production assertions read from.
"""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from replay.gateway import GatewayChatModel
from tracing import InMemoryTracer, Span, retrieved_doc_ids

from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph

_FALSE_ANSWER = "Your plan is contract-free, no fee, cancel any time."


def _graph(cassette_dir, backend, tracer):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=cassette_dir, mode="replay")
    return build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer(), tracer=tracer)


def test_span_attributes_are_read_only_and_annotate_replaces():
    """A span's attributes are a read only view. Annotate produces a NEW span, never mutates."""
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
    from determinism.canonical import serialize_tool_result
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


# ---- retrieved_doc_ids: the sole producer of GradeContext.retrieved_doc_ids ----
#
# Built directly from `Span` objects, no graph run needed: this decoder reads only `span.name` and
# `span.attributes["result"]`, the exact shape the real `search_knowledge` tool span carries, so a
# hand built span is byte equivalent to a recorded one for this function's purposes. Before this
# module, nothing ever called `retrieved_doc_ids` outside of `runner.run_case`; a rename of the span
# name or the `result` attribute would have turned `RetrievalIdsRecalledGrader` into an unconditional
# FAIL with no test going red.


def _search_span(seq: int, result: object, *, name: str = "search_knowledge") -> Span:
    return Span(seq=seq, name=name, kind="tool", parent=None, attributes={"result": result})


def test_retrieved_doc_ids_reads_the_bare_passages_array_payload():
    """The happy path payload shape: a bare JSON array of passages, no wrapper object."""
    payload = json.dumps([{"chunk_id": "a"}, {"chunk_id": "b"}])
    assert retrieved_doc_ids((_search_span(0, payload),)) == ("a", "b")


def test_retrieved_doc_ids_reads_the_degraded_wrapper_payload():
    """The degradation ladder wraps the same passages array in {atlas_degraded, degradation_mode,
    passages}; the decoder must reach into it, not just the bare array shape."""
    payload = json.dumps(
        {"atlas_degraded": True, "degradation_mode": "reranker_down", "passages": [{"chunk_id": "x"}, {"chunk_id": "y"}]}
    )
    assert retrieved_doc_ids((_search_span(0, payload),)) == ("x", "y")


def test_retrieved_doc_ids_dedupes_first_seen_order_preserved():
    """Two search_knowledge spans (e.g. a re-query mid turn) whose passages overlap: the union is
    de-duplicated, and the surviving order is first-seen, never sorted or last-seen."""
    first = json.dumps([{"chunk_id": "a"}, {"chunk_id": "b"}])
    second = json.dumps([{"chunk_id": "b"}, {"chunk_id": "c"}, {"chunk_id": "a"}])
    spans = (_search_span(0, first), _search_span(1, second))
    assert retrieved_doc_ids(spans) == ("a", "b", "c")


def test_retrieved_doc_ids_skips_a_non_search_knowledge_span():
    """A tool span from any other tool (e.g. the account tool) must never contribute ids, even if its
    `result` happens to parse as the same shape."""
    payload = json.dumps([{"chunk_id": "a"}])
    assert retrieved_doc_ids((_search_span(0, payload, name="get_account_summary"),)) == ()


def test_retrieved_doc_ids_skips_a_non_string_result():
    """`result` recorded as a non-string (e.g. a dict, if a future caller stops serializing to JSON
    text) must be silently skipped, never raise."""
    assert retrieved_doc_ids((_search_span(0, {"passages": [{"chunk_id": "a"}]}),)) == ()


def test_retrieved_doc_ids_skips_a_malformed_non_json_result():
    """A `result` string that fails to parse as JSON at all must be silently skipped, never raise."""
    assert retrieved_doc_ids((_search_span(0, "not json at all {"),)) == ()


def test_retrieved_doc_ids_skips_when_passages_is_not_a_list():
    """A wrapper payload whose `passages` key is present but not a list (a malformed upstream
    payload) must be silently skipped, never raise."""
    payload = json.dumps({"passages": "not-a-list"})
    assert retrieved_doc_ids((_search_span(0, payload),)) == ()


def test_retrieved_doc_ids_is_empty_with_no_search_knowledge_span_at_all():
    """No tool span present at all (e.g. a turn answered without retrieval) is a defined empty
    result, not a crash -- the same convention the decoder's own docstring names."""
    assert retrieved_doc_ids(()) == ()
