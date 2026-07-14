"""The unified Atlas graph: answer / read / act paths end to end.

The answer path proves the runtime cold open catch: the same grounded but false answer is held
at the render guard for a legacy customer and rendered for a current one. The act path proves
the guard + confirmation interrupt + idempotent execution.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from determinism.canonical import serialize_tool_result
from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from replay.gateway import GatewayChatModel

from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph

_FALSE_ANSWER = "Your plan is contract-free, no fee, cancel any time."


def _graph(cassette_dir, backend):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=cassette_dir, mode="replay")
    return build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer())


@pytest.mark.asyncio
async def test_answer_path_cold_open_caught_at_render_for_legacy_customer(tmp_path, seed_cassette):
    user = HumanMessage("Is my plan contract-free?")
    seed_cassette(tmp_path, [user], {"content": _FALSE_ANSWER, "tool_calls": []})
    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")))
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_legacy_term"}},
        {"configurable": {"thread_id": "a1"}},
    )
    assert out["final_response"].startswith("[safe handoff]")  # the false answer is held, not rendered


@pytest.mark.asyncio
async def test_answer_path_same_answer_renders_for_current_customer(tmp_path, seed_cassette):
    user = HumanMessage("Is my plan contract-free?")
    seed_cassette(tmp_path, [user], {"content": _FALSE_ANSWER, "tool_calls": []})
    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")))
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_current"}},
        {"configurable": {"thread_id": "a2"}},
    )
    assert out["final_response"] == _FALSE_ANSWER  # true for Sarah, so it renders


@pytest.mark.asyncio
async def test_act_path_confirms_then_executes_once(tmp_path, seed_cassette):
    backend = ActionsBackend(IdFactory("ref"))
    user = HumanMessage("Switch me to the fast plan")
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]})
    graph = _graph(tmp_path, backend)
    cfg = {"configurable": {"thread_id": "act1"}}
    first = await graph.ainvoke({"messages": [user], "session": {"customer_id": "cust_current"}}, cfg)
    assert "__interrupt__" in first  # paused at the confirmation gate
    out = await graph.ainvoke(Command(resume="CONFIRM"), cfg)
    assert "Done" in out["final_response"]
    assert backend.change_count("cust_current") == 1


@pytest.mark.asyncio
async def test_multi_turn_keys_cache_and_traces_per_turn_not_per_conversation(tmp_path, seed_cassette):
    """Two turns on ONE thread (state persists via the checkpointer): each turn gets its own trace
    span keyed on ITS question, and each answer is cached under the question that produced it, not
    the conversation's first. A later thread asking the second question is served the right answer."""
    from tracing import InMemoryTracer

    from atlas.orchestration.atlas_graph import build_atlas_graph

    tracer = InMemoryTracer()
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    graph = build_atlas_graph(gw, IdFactory("idem"), ActionsBackend(IdFactory("ref")), new_checkpointer(), tracer=tracer)

    q1, a1 = HumanMessage("What are your opening hours?"), "We're open 9 to 5."
    q2, a2 = HumanMessage("What is a data cap?"), "A data cap is a monthly limit."
    seed_cassette(tmp_path, [q1], {"content": a1, "tool_calls": []})
    seed_cassette(tmp_path, [q1, AIMessage(a1), q2], {"content": a2, "tool_calls": []})  # turn 2 sees turn 1's history

    sess = {"customer_id": "cust_current"}
    r1 = await graph.ainvoke({"messages": [q1], "session": sess}, {"configurable": {"thread_id": "cust_current::c"}})
    r2 = await graph.ainvoke({"messages": [q2], "session": sess}, {"configurable": {"thread_id": "cust_current::c"}})
    assert (r1["final_response"], r2["final_response"]) == (a1, a2)

    # per turn trace spans, each carrying its OWN question (not q1 twice)
    turn_inputs = [s.attributes["input"] for s in tracer.spans if s.kind == "turn"]
    assert turn_inputs == [q1.content, q2.content]

    # the SECOND answer was cached under the SECOND question: a fresh thread asking q2 is served a2
    # (cache hit short circuits the model, no cassette needed for this turn)
    r3 = await graph.ainvoke({"messages": [q2], "session": sess}, {"configurable": {"thread_id": "cust_current::other"}})
    assert r3["final_response"] == a2
    assert [s for s in tracer.spans if s.name == "cache" and s.attributes.get("hit")]  # served from cache


@pytest.mark.asyncio
async def test_act_path_writes_through_to_account_state(tmp_path, seed_cassette):
    """End to end: the graph confirms change_plan and the write lands in the account store, so a
    later read returns the new plan, a proper working application, not an audit only stub."""
    from atlas.domain import accounts
    from atlas.domain.accounts import apply_write
    from atlas.orchestration.atlas_graph import build_atlas_graph

    backend = ActionsBackend(IdFactory("ref"), writer=apply_write)
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer())

    user = HumanMessage("Switch me to the fast plan")
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]})
    cfg = {"configurable": {"thread_id": "wt1"}}
    assert accounts.get_account("cust_legacy_term").plan_id == "plan_legacy_value"  # before
    await graph.ainvoke({"messages": [user], "session": {"customer_id": "cust_legacy_term"}}, cfg)
    await graph.ainvoke(Command(resume="CONFIRM"), cfg)
    assert accounts.get_account("cust_legacy_term").plan_id == "plan_current_fast"  # after: read sees the write
    assert backend.change_count("cust_legacy_term") == 1


@pytest.mark.asyncio
async def test_write_then_read_turn_reflects_the_new_plan(tmp_path, seed_cassette):
    """Two turns on one store: turn 1 confirms change_plan, turn 2 reads the account and the answer
    reflects the new plan, read after write through the full graph, not just the backend."""
    from determinism.canonical import serialize_tool_result
    from atlas.domain import accounts
    from atlas.domain.accounts import apply_write
    from atlas.domain.catalog import get_plan
    from atlas.orchestration.atlas_graph import build_atlas_graph

    backend = ActionsBackend(IdFactory("ref"), writer=apply_write)
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer())

    # turn 1, act: Daniel switches to the current plan, then confirms
    act = HumanMessage("Switch me to the fast plan")
    seed_cassette(tmp_path, [act], {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]})
    await graph.ainvoke({"messages": [act], "session": {"customer_id": "cust_legacy_term"}}, {"configurable": {"thread_id": "t-act"}})
    await graph.ainvoke(Command(resume="CONFIRM"), {"configurable": {"thread_id": "t-act"}})

    # turn 2, read: get_account_summary now returns the POST write plan (Daniel is on the current plan)
    read = HumanMessage("What plan am I on now?")
    seed_cassette(tmp_path, [read], {"content": "", "tool_calls": [{"name": "get_account_summary", "args": {}, "id": "r1"}]})
    plan = get_plan("plan_current_fast")
    summary = serialize_tool_result({"customer": accounts.get_account("cust_legacy_term").name, "plan": plan.name, "has_contract": plan.has_term})
    ai = AIMessage(content="", tool_calls=[{"name": "get_account_summary", "args": {}, "id": "r1"}])
    tool_msg = ToolMessage(content=summary, tool_call_id="r1", name="get_account_summary")
    seed_cassette(tmp_path, [read, ai, tool_msg], {"content": "You are now on the Fast (current) plan.", "tool_calls": []})

    out = await graph.ainvoke({"messages": [read], "session": {"customer_id": "cust_legacy_term"}}, {"configurable": {"thread_id": "t-read"}})
    assert out["final_response"] == "You are now on the Fast (current) plan."  # the read turn sees the write


@pytest.mark.asyncio
async def test_write_tool_is_unreachable_on_a_non_action_turn(tmp_path, seed_cassette):
    """Least agency at runtime: an injected document makes the model emit reset_modem on a
    troubleshooting turn. The write is unreachable (not merely guarded), nothing executes."""
    backend = ActionsBackend(IdFactory("ref"))
    user = HumanMessage("My wifi is down, can you help?")
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [{"name": "reset_modem", "args": {}, "id": "x1"}]})
    graph = _graph(tmp_path, backend)
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_current", "intent": "troubleshooting"}},
        {"configurable": {"thread_id": "bind1"}},
    )
    assert out["final_response"].startswith("[safe handoff]")
    assert not backend.applied("cust_current")  # the tool was never reached, let alone executed


@pytest.mark.asyncio
async def test_read_tool_is_unreachable_on_a_policy_turn(tmp_path, seed_cassette):
    """A policy question turn binds knowledge + catalog only, so an account read is unreachable."""
    user = HumanMessage("What are your opening hours?")
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [{"name": "get_account_summary", "args": {}, "id": "r9"}]})
    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")))
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_current", "intent": "policy_question"}},
        {"configurable": {"thread_id": "bind2"}},
    )
    assert out["final_response"].startswith("[safe handoff]")  # account read not bound to a policy turn


@pytest.mark.asyncio
async def test_act_path_materializes_the_proposal_through_the_actions_server(tmp_path, seed_cassette):
    """The write proposal is produced by the customer scoped actions MCP server (not fabricated):
    the interrupt payload carries the server's 'proposed' record."""
    user = HumanMessage("Switch me to the fast plan")
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]})
    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")))
    first = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_current"}},
        {"configurable": {"thread_id": "prop1"}},
    )
    assert "__interrupt__" in first
    proposal = first["pending"]["proposal"]
    assert '"status":"proposed"' in proposal and '"customer":"cust_current"' in proposal


@pytest.mark.asyncio
async def test_act_path_value_bounds_fail_closed(tmp_path, seed_cassette):
    backend = ActionsBackend(IdFactory("ref"))
    # an ACTION phrasing so routing reaches the value-bounds gate; a non-action phrasing would be
    # stopped earlier by binding (change_plan unreachable), testing the wrong guard.
    user = HumanMessage("Change my plan to the internal zero plan")
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_internal_zero"}, "id": "c2"}]})
    graph = _graph(tmp_path, backend)
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_current"}},
        {"configurable": {"thread_id": "act2"}},
    )
    assert out["final_response"].startswith("[safe handoff]")
    assert "not a real, offered plan" in out["final_response"]  # the VALUE-BOUNDS rejection, not binding
    assert backend.change_count("cust_current") == 0  # never executed


@pytest.mark.asyncio
async def test_act_path_bogus_addon_fails_closed(tmp_path, seed_cassette):
    backend = ActionsBackend(IdFactory("ref"))
    user = HumanMessage("Add the free unicorn add-on")
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [
        {"name": "add_addon", "args": {"addon_id": "free_unicorn"}, "id": "a1"},
    ]})
    graph = _graph(tmp_path, backend)
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_current"}},
        {"configurable": {"thread_id": "ba1"}},
    )
    assert out["final_response"].startswith("[safe handoff]")  # not a real add on, never proposed
    assert not backend.applied("cust_current", "add_addon")


@pytest.mark.asyncio
async def test_read_path_answers_from_the_account(tmp_path, seed_cassette):
    user = HumanMessage("What plan am I on?")
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [{"name": "get_account_summary", "args": {}, "id": "r1"}]})
    tool_text = serialize_tool_result(
        {"customer": "Daniel", "plan": "Value (legacy, discontinued)", "has_contract": True}
    )
    ai = AIMessage(content="", tool_calls=[{"name": "get_account_summary", "args": {}, "id": "r1"}])
    tool_msg = ToolMessage(content=tool_text, tool_call_id="r1", name="get_account_summary")
    seed_cassette(tmp_path, [user, ai, tool_msg], {"content": "You are on the Value (legacy) plan.", "tool_calls": []})
    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")))
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_legacy_term"}},
        {"configurable": {"thread_id": "r1t"}},
    )
    assert out["final_response"] == "You are on the Value (legacy) plan."


@pytest.mark.asyncio
async def test_read_path_answers_the_bill_from_the_account(tmp_path, seed_cassette):
    """A get_bill read routes through tools_read to the session scoped account server and the agent
    answers from it, the four account reads flow through the graph, not just the summary."""
    from determinism.canonical import serialize_tool_result
    from atlas.domain import accounts

    user = HumanMessage("What's my bill this month?")
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [{"name": "get_bill", "args": {}, "id": "b1"}]})
    b = accounts.get_bill("cust_legacy_term")
    tool_text = serialize_tool_result({"period": b.period, "amount": b.amount, "due_date": b.due_date, "paid": b.paid})
    ai = AIMessage(content="", tool_calls=[{"name": "get_bill", "args": {}, "id": "b1"}])
    tool_msg = ToolMessage(content=tool_text, tool_call_id="b1", name="get_bill")
    seed_cassette(tmp_path, [user, ai, tool_msg], {"content": "Your current bill is GBP 39.00, due 2026-06-28.", "tool_calls": []})

    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")))
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_legacy_term"}},
        {"configurable": {"thread_id": "bill1"}},
    )
    assert out["final_response"] == "Your current bill is GBP 39.00, due 2026-06-28."


@pytest.mark.asyncio
async def test_answer_path_retrieves_then_grounded_answer_caught_at_render(tmp_path, seed_cassette):
    """The agent searches the help docs, grounds its answer in the retrieved page, and the
    render guard holds it for the legacy customer: RAG actually in the loop."""
    from atlas.adapters.inmemory_retriever import InMemoryRetriever

    query = "plan contract term cancel fee"
    user = HumanMessage("Is my plan contract-free?")
    toolcall = [{"name": "search_knowledge", "args": {"query": query}, "id": "k1"}]
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": toolcall})

    chunks = InMemoryRetriever().search(query)
    passages = serialize_tool_result([{"doc_id": c.doc_id, "text": c.text} for c in chunks])
    ai = AIMessage(content="", tool_calls=toolcall)
    tool_msg = ToolMessage(content=passages, tool_call_id="k1", name="search_knowledge")
    seed_cassette(tmp_path, [user, ai, tool_msg], {"content": _FALSE_ANSWER, "tool_calls": []})

    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")))
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_legacy_term"}},
        {"configurable": {"thread_id": "rag1"}},
    )
    assert out["final_response"].startswith("[safe handoff]")  # grounded in the page, false for Daniel


@pytest.mark.asyncio
async def test_act_path_single_write_rule_fails_closed_on_a_batch(tmp_path, seed_cassette):
    backend = ActionsBackend(IdFactory("ref"))
    user = HumanMessage("Switch my plan and reset my modem")
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [
        {"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"},
        {"name": "reset_modem", "args": {}, "id": "c2"},
    ]})
    graph = _graph(tmp_path, backend)
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_current"}},
        {"configurable": {"thread_id": "sw1"}},
    )
    assert out["final_response"].startswith("[safe handoff]")
    assert backend.change_count("cust_current") == 0  # neither write executed


@pytest.mark.asyncio
async def test_act_path_rejects_a_model_supplied_customer_id(tmp_path, seed_cassette):
    backend = ActionsBackend(IdFactory("ref"))
    user = HumanMessage("Switch me to the fast plan")
    # the model tries to steer whose account is changed by smuggling a customer_id into the args
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [
        {"name": "change_plan", "args": {"plan_id": "plan_current_fast", "customer_id": "cust_neighbor"}, "id": "c1"},
    ]})
    graph = _graph(tmp_path, backend)
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_current"}},
        {"configurable": {"thread_id": "sc1"}},
    )
    assert out["final_response"].startswith("[safe handoff]")
    assert backend.change_count("cust_neighbor") == 0  # the model does not get to pick the customer


@pytest.mark.asyncio
async def test_render_guard_holds_unsafe_markup(tmp_path, seed_cassette):
    user = HumanMessage("Show me my plan")
    seed_cassette(tmp_path, [user], {"content": '<img src=x onerror="steal()">', "tool_calls": []})
    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")))
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_current"}},
        {"configurable": {"thread_id": "rs1"}},
    )
    assert out["final_response"].startswith("[safe handoff]")  # the Lena payload never reaches the browser


@pytest.mark.asyncio
async def test_render_guard_holds_another_customers_data(tmp_path, seed_cassette):
    user = HumanMessage("What's my bill?")
    seed_cassette(tmp_path, [user], {"content": "Daniel's bill this month is GBP 39.", "tool_calls": []})
    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")))
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_current"}},
        {"configurable": {"thread_id": "oc1"}},
    )
    assert out["final_response"].startswith("[safe handoff]")  # cross tenant disclosure caught at the door


@pytest.mark.asyncio
async def test_repeated_question_is_served_from_the_cache(tmp_path, seed_cassette):
    """The second identical turn is answered from the per customer cache, not the model, proven
    by a `cache` span with `hit=True` on the trace."""
    from tracing import InMemoryTracer

    from atlas.orchestration.atlas_graph import build_atlas_graph

    tracer = InMemoryTracer()
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    graph = build_atlas_graph(gw, IdFactory("idem"), ActionsBackend(IdFactory("ref")), new_checkpointer(), tracer=tracer)

    user = HumanMessage("What are your opening hours?")
    seed_cassette(tmp_path, [user], {"content": "We're open 9 to 5.", "tool_calls": []})
    state = {"messages": [user], "session": {"customer_id": "cust_current"}}
    first = await graph.ainvoke(state, {"configurable": {"thread_id": "ca1"}})
    assert first["final_response"] == "We're open 9 to 5."
    second = await graph.ainvoke(state, {"configurable": {"thread_id": "ca2"}})
    assert second["final_response"] == "We're open 9 to 5."

    cache_hits = [s for s in tracer.spans if s.name == "cache" and s.attributes.get("hit")]
    assert cache_hits  # the repeat turn short circuited the model


# ---- Finding 03: cross-customer cache isolation is enforced by the GRAPH, not just the class ----

@pytest.mark.asyncio
async def test_cache_isolates_two_customers_on_the_same_account_question(tmp_path, seed_cassette):
    """Two customers ask the byte-identical account question through ONE shared cache. The answer is
    customer specific (a bill), so B must be served HER OWN figure, never A's cached one. This is the
    regression the leaky NaiveCache fails (it keys on the question alone) and the render guard would
    NOT catch, because a bare amount names no other customer. Goes red under NaiveCache()."""
    from atlas.domain import accounts
    from atlas.orchestration.atlas_graph import build_atlas_graph

    q = HumanMessage("What's my bill this month?")
    # the first-turn request carries no customer id, so it is one shared cassette: the model calls get_bill.
    seed_cassette(tmp_path, [q], {"content": "", "tool_calls": [{"name": "get_bill", "args": {}, "id": "b1"}]})
    ai = AIMessage(content="", tool_calls=[{"name": "get_bill", "args": {}, "id": "b1"}])

    def seed_answer(customer_id, answer):
        b = accounts.get_bill(customer_id)
        tool_text = serialize_tool_result({"period": b.period, "amount": b.amount, "due_date": b.due_date, "paid": b.paid})
        tool_msg = ToolMessage(content=tool_text, tool_call_id="b1", name="get_bill")
        seed_cassette(tmp_path, [q, ai, tool_msg], {"content": answer, "tool_calls": []})

    seed_answer("cust_legacy_term", "Your bill is GBP 39.00.")   # Daniel
    seed_answer("cust_current", "Your bill is GBP 35.00.")       # Sarah

    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    graph = build_atlas_graph(gw, IdFactory("idem"), ActionsBackend(IdFactory("ref")), new_checkpointer())

    a = await graph.ainvoke({"messages": [q], "session": {"customer_id": "cust_legacy_term"}}, {"configurable": {"thread_id": "tA"}})
    b = await graph.ainvoke({"messages": [q], "session": {"customer_id": "cust_current"}}, {"configurable": {"thread_id": "tB"}})
    assert a["final_response"] == "Your bill is GBP 39.00."
    assert b["final_response"] == "Your bill is GBP 35.00."   # NOT served Daniel's cached 39.00


# ---- Finding 06: a confirmed write invalidates the cache, so a repeat read is not served stale ----

@pytest.mark.asyncio
async def test_repeated_read_after_a_confirmed_write_is_not_served_stale(tmp_path, seed_cassette):
    """A customer reads a figure (cached), changes their own plan (which re-prices the bill), then asks
    the SAME question again. The confirmed write invalidates their cache, so the repeat read reflects
    the new state instead of the pre-write figure short-circuited from the cache."""
    from atlas.domain import accounts
    from atlas.domain.accounts import apply_write
    from atlas.domain.catalog import compute_price
    from atlas.orchestration.atlas_graph import build_atlas_graph

    backend = ActionsBackend(IdFactory("ref"), writer=apply_write)
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer())

    read = HumanMessage("What's my bill this month?")
    ai_r = AIMessage(content="", tool_calls=[{"name": "get_bill", "args": {}, "id": "b1"}])
    seed_cassette(tmp_path, [read], {"content": "", "tool_calls": [{"name": "get_bill", "args": {}, "id": "b1"}]})

    def seed_bill_answer(amount, answer):
        orig = accounts.get_bill("cust_legacy_term")
        tool_text = serialize_tool_result({"period": orig.period, "amount": amount, "due_date": orig.due_date, "paid": orig.paid})
        tool_msg = ToolMessage(content=tool_text, tool_call_id="b1", name="get_bill")
        seed_cassette(tmp_path, [read, ai_r, tool_msg], {"content": answer, "tool_calls": []})

    from decimal import Decimal
    seed_bill_answer(Decimal("39.00"), "Your bill is GBP 39.00.")                       # turn 1: pre-write
    seed_bill_answer(compute_price("plan_current_fast"), "Your bill is GBP 35.00.")     # turn 3: post-write, re-priced

    # turn 1: read the bill (cached under the customer key)
    t1 = await graph.ainvoke({"messages": [read], "session": {"customer_id": "cust_legacy_term"}}, {"configurable": {"thread_id": "r1"}})
    assert t1["final_response"] == "Your bill is GBP 39.00."

    # turn 2: change the plan and confirm (re-prices the bill, and invalidates the customer's cache)
    act = HumanMessage("Switch me to the fast plan")
    seed_cassette(tmp_path, [act], {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]})
    await graph.ainvoke({"messages": [act], "session": {"customer_id": "cust_legacy_term"}}, {"configurable": {"thread_id": "w1"}})
    await graph.ainvoke(Command(resume="CONFIRM"), {"configurable": {"thread_id": "w1"}})

    # turn 3: the SAME read question on a fresh thread. Without invalidation it would be served the stale
    # GBP 39.00 from the cache; with it, the cache misses and the read reflects the re-priced bill.
    t3 = await graph.ainvoke({"messages": [read], "session": {"customer_id": "cust_legacy_term"}}, {"configurable": {"thread_id": "r3"}})
    assert t3["final_response"] == "Your bill is GBP 35.00."


# ---- Finding 07: refusal paths run model-controlled fragments through the output escaper ----

@pytest.mark.asyncio
async def test_value_bounds_refusal_does_not_reflect_injected_markup(tmp_path, seed_cassette):
    """A rejected write argument is model-controlled; the refusal must pass the same output escaper as
    the render path, so an injected <img ...> plan id never reaches the reply verbatim."""
    payload = "<img src=x onerror=alert(1)>"
    user = HumanMessage("Change my plan to the fast one")
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": payload}, "id": "c1"}]})
    graph = _graph(tmp_path, ActionsBackend(IdFactory("ref")))
    out = await graph.ainvoke(
        {"messages": [user], "session": {"customer_id": "cust_current"}},
        {"configurable": {"thread_id": "xss1"}},
    )
    assert out["final_response"].startswith("[safe handoff]")
    assert "<img" not in out["final_response"] and "onerror" not in out["final_response"]


# ---- Finding 08: an account read on a thread makes a later knowledge-only turn non-shareable ----

@pytest.mark.asyncio
async def test_knowledge_turn_after_an_account_read_is_not_shared_generically(tmp_path, seed_cassette):
    """The 'safe to share' signal is sticky at thread scope. Turn 1 reads the account (account_seen),
    turn 2 is knowledge-only: it could restate account data from the thread history, so its answer is
    keyed per-customer, NOT under the shared generic key another customer would hit. Without the sticky
    flag turn 2 would be marked generic (used_knowledge and not used_account THIS turn) and leak."""
    from atlas.adapters.inmemory_retriever import InMemoryRetriever
    from atlas.domain import accounts
    from atlas.domain.cache import PerCustomerCache
    from atlas.orchestration.atlas_graph import build_atlas_graph

    cache = PerCustomerCache()  # held so we can inspect the KEY the answer was stored under
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    graph = build_atlas_graph(gw, IdFactory("idem"), ActionsBackend(IdFactory("ref")), new_checkpointer(), cache=cache)

    # turn 1: an account read -> used_account, so account_seen becomes True and stays True
    u1 = HumanMessage("What's my bill this month?")
    ai1 = AIMessage(content="", tool_calls=[{"name": "get_bill", "args": {}, "id": "b1"}])
    seed_cassette(tmp_path, [u1], {"content": "", "tool_calls": [{"name": "get_bill", "args": {}, "id": "b1"}]})
    b = accounts.get_bill("cust_legacy_term")
    toolmsg1 = ToolMessage(content=serialize_tool_result({"period": b.period, "amount": b.amount, "due_date": b.due_date, "paid": b.paid}), tool_call_id="b1", name="get_bill")
    ans1 = "Your bill is GBP 39.00."
    seed_cassette(tmp_path, [u1, ai1, toolmsg1], {"content": ans1, "tool_calls": []})

    # turn 2 (same thread): a knowledge-only question that restates nothing new, still keyed per-customer
    u2 = HumanMessage("What is a data cap?")
    ai_ans1 = AIMessage(content=ans1)  # turn 1's answer, now in the thread history (see test_multi_turn_...)
    query = "data cap"
    ai2 = AIMessage(content="", tool_calls=[{"name": "search_knowledge", "args": {"query": query}, "id": "k1"}])
    seed_cassette(tmp_path, [u1, ai1, toolmsg1, ai_ans1, u2], {"content": "", "tool_calls": [{"name": "search_knowledge", "args": {"query": query}, "id": "k1"}]})
    passages = serialize_tool_result([{"doc_id": c.doc_id, "text": c.text} for c in InMemoryRetriever().search(query)])
    toolmsg2 = ToolMessage(content=passages, tool_call_id="k1", name="search_knowledge")
    ans2 = "A data cap is a monthly usage limit."
    seed_cassette(tmp_path, [u1, ai1, toolmsg1, ai_ans1, u2, ai2, toolmsg2], {"content": ans2, "tool_calls": []})

    cfg = {"configurable": {"thread_id": "sticky"}}
    await graph.ainvoke({"messages": [u1], "session": {"customer_id": "cust_legacy_term"}}, cfg)
    r2 = await graph.ainvoke({"messages": [u2], "session": {"customer_id": "cust_legacy_term"}}, cfg)
    assert r2["final_response"] == ans2

    # the knowledge answer is stored PER CUSTOMER, never under the shared generic key another customer hits
    assert cache.get("cust_legacy_term", "What is a data cap?", generic=False) == ans2
    assert cache.get("cust_neighbor", "What is a data cap?", generic=True) is None
