"""The unified Atlas graph, the runtime the architecture article draws.

agent decides; routing splits answer / read / act. The act path goes through the fail closed
`pre_action_guard` and the confirmation `interrupt()`. The answer path goes through
`pre_render_guard`, which is the LAST place a fee claim that contradicts the account is caught
and held (the cold open's runtime catch). Identity lives in the non model `session` channel.
Reads run over the account MCP server (in memory transport); the write executes via the actions
backend behind the confirmation gate.
"""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from mcp.shared.memory import create_connected_server_and_client_session

from atlas.domain import guard as guardrules
from atlas.domain.actions import ActionsBackend
from atlas.domain.binding import KNOWLEDGE_TOOLS, classify_intent, is_reachable
from atlas.domain.cache import PerCustomerCache
from atlas.domain.confirmation import ConfirmationError, PendingAction, execute_if_confirmed
from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.mcp_servers.account_server import build_account_server
from atlas.mcp_servers.actions_server import build_actions_server
from atlas.mcp_servers.knowledge_server import build_knowledge_server
from tracing import NullTracer

# The outcome sentinels the runtime ships and the eval/drift lanes read back. ONE definition: a
# reword here updates every grader that keys off it, instead of silently failing those graders open
# (a safety eval that stops matching "[safe handoff]" would quietly certify the refusals it exists
# to catch). The drift lane reads the outcome from the span tree, not these strings.
HANDOFF_PREFIX = "[safe handoff]"
WRITE_CONFIRMATION = "Your reference is"


class AtlasState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session: dict  # {customer_id}, non model channel
    pending: Optional[dict]
    result: Optional[dict]
    final_response: Optional[str]
    trace_root: Optional[int]    # seq of THIS turn's span (the input) everything hangs under
    trace_parent: Optional[int]  # seq of the agent span the current tools hang under
    turn_question: Optional[str]    # the current turn's user question, the cache key for this turn
    intent: Optional[str]           # per turn intent, binds which tools are reachable (least agency)
    used_account: Optional[bool]    # turn read this customer's account → answer is customer specific
    used_knowledge: Optional[bool]  # turn used the generic help corpus


def _text_of(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


async def _account_call(customer_id: str, tool_name: str, args: dict) -> str:
    server = build_account_server(customer_id)
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        res = await client.call_tool(tool_name, args or {})
        return res.content[0].text


async def _knowledge_call(retriever, query: str) -> str:
    server = build_knowledge_server(retriever)
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        res = await client.call_tool("search_knowledge", {"query": query})
        return res.content[0].text


async def _actions_call(customer_id: str, tool: str, args: dict) -> str:
    """Materialize the write proposal through the customer scoped actions MCP server (the real
    write surface). Identity is bound at connect, so customer_id is never a tool argument."""
    server = build_actions_server(customer_id)
    call_args = {k: v for k, v in (args or {}).items() if k != "customer_id"}
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        res = await client.call_tool(tool, call_args)
        return res.content[0].text


def build_atlas_graph(model: BaseChatModel, ids, backend: ActionsBackend, checkpointer, retriever=None, tracer=None, cache=None):
    retriever = retriever or InMemoryRetriever()
    tracer = tracer or NullTracer()  # observation is opt in; the runtime is always instrumented
    cache = cache or PerCustomerCache()  # per customer keying, a generic answer is shared, a bill is not

    async def agent(state: AtlasState) -> dict:
        messages = state["messages"]
        # A fresh user turn is the only hop whose last message is a HumanMessage; a tool loop
        # re entry ends in a ToolMessage. Under the checkpointer, state persists across turns on a
        # thread, so the turn span / cache lookup / per turn flags must reset HERE, per turn, keying
        # off the CURRENT question, never the conversation's first one.
        fresh_turn = bool(messages) and isinstance(messages[-1], HumanMessage)
        extra: dict = {}
        if fresh_turn:
            question = _text_of(messages[-1])
            # intent is set per turn (caller may pin it on the session; else classified) and binds
            # which tools are reachable for the whole turn, least agency, decided before the model acts.
            intent = state["session"].get("intent") or classify_intent(question)
            # record the BOUND intent on the turn span, so a grader/drift lane reads what the runtime
            # actually bound (and could see it move), not a re-derivation from the raw utterance.
            root = tracer.open("turn", "turn", input=question, intent=intent)
            extra = {"trace_root": root, "turn_question": question, "intent": intent,
                     "used_account": False, "used_knowledge": False}
            for generic in (False, True):  # customer specific key first, then the shared generic key
                hit = cache.get(state["session"]["customer_id"], question, generic=generic)
                if hit is not None:
                    tracer.open("cache", "node", root, hit=True, generic=generic)
                    # the cached reply still passes through the render guard before it ships
                    return {"messages": [AIMessage(content=hit)], **extra}
        else:
            root = state.get("trace_root")
        result = await model._agenerate(list(messages))
        seq = tracer.open("agent", "llm", root, model=getattr(model, "model_id", "model"))
        return {"messages": [result.generations[0].message], "trace_parent": seq, **extra}

    async def tools_read(state: AtlasState) -> dict:
        last = state["messages"][-1]
        cid = state["session"]["customer_id"]
        intent = state.get("intent") or "troubleshooting"
        # least agency: a tool not bound to this turn's intent is unreachable, not merely guarded
        unreachable = [tc["name"] for tc in last.tool_calls if not is_reachable(intent, tc["name"])]
        if unreachable:
            tracer.open("bind_guard", "guard", state.get("trace_root"), ok=False, intent=intent, tools=unreachable)
            return {"final_response": f"{HANDOFF_PREFIX} {unreachable[0]} is not available on a {intent} turn"}
        parent = state.get("trace_parent")
        used_knowledge = state.get("used_knowledge") or False
        used_account = state.get("used_account") or False
        out: list[BaseMessage] = []
        for tc in last.tool_calls:
            if tc["name"] in KNOWLEDGE_TOOLS:
                text = await _knowledge_call(retriever, tc.get("args", {}).get("query", ""))
                used_knowledge = True
            else:
                text = await _account_call(cid, tc["name"], tc.get("args", {}))
                used_account = True  # answer now depends on this customer, never shared as generic
            tracer.open(tc["name"], "tool", parent, args=tc.get("args", {}), result=text)
            out.append(ToolMessage(content=text, tool_call_id=tc["id"], name=tc["name"]))
        return {"messages": out, "used_knowledge": used_knowledge, "used_account": used_account}

    async def pre_action_guard(state: AtlasState) -> dict:
        last = state["messages"][-1]
        cid = state["session"]["customer_id"]
        intent = state.get("intent") or "troubleshooting"
        names = [c["name"] for c in last.tool_calls]
        # least agency first: a write tool not bound to this turn's intent is unreachable (the
        # injected document "reset this customer's modem" on a troubleshooting turn).
        root = state.get("trace_root")
        unreachable = [n for n in names if not is_reachable(intent, n)]
        if unreachable:
            tracer.open("pre_action_guard", "guard", root, ok=False, reason=f"{unreachable[0]} unreachable on a {intent} turn", tool=unreachable[0])
            return {"final_response": f"{HANDOFF_PREFIX} {unreachable[0]} is not available on a {intent} turn"}
        single = guardrules.check_single_write(names)
        if not single.ok:  # a multi or mixed read+write batch fails closed before anything runs
            tracer.open("pre_action_guard", "guard", root, ok=False, reason=single.reason)
            return {"final_response": f"{HANDOFF_PREFIX} {single.reason}"}
        tc = last.tool_calls[0]
        args = tc.get("args", {})
        # an id the model tried to put in the args is rejected unless it matches the session
        scope = guardrules.check_scope(args.get("customer_id", cid), cid)
        if not scope.ok:
            tracer.open("pre_action_guard", "guard", root, ok=False, reason=scope.reason, tool=tc["name"])
            return {"final_response": f"{HANDOFF_PREFIX} {scope.reason}"}
        bounds = guardrules.check_value_bounds(tc["name"], args)
        tracer.open("pre_action_guard", "guard", root, ok=bounds.ok, reason=bounds.reason, tool=tc["name"])
        if not bounds.ok:
            return {"final_response": f"{HANDOFF_PREFIX} {bounds.reason}"}
        # materialize the proposal through the customer scoped actions MCP server (the write surface)
        proposal = await _actions_call(cid, tc["name"], args)
        tracer.open(tc["name"], "tool", state.get("trace_root"), args=args, proposal=proposal)
        return {
            "pending": {
                "tool": tc["name"],
                "args": args,
                "idempotency_key": ids.next(),  # bound before the interrupt checkpoint
                "customer_id": cid,
                "proposal": proposal,
            }
        }

    def confirm(state: AtlasState) -> dict:
        if state.get("final_response"):  # guard already failed closed
            return {}
        typed = interrupt({"proposal": state["pending"]})
        p = state["pending"]
        try:
            pending = PendingAction(tool=p["tool"], args=p["args"], idempotency_key=p["idempotency_key"], customer_id=p["customer_id"])
            res = execute_if_confirmed(pending, typed, backend)
            tracer.open("execute_action", "node", state.get("trace_root"), applied=res.applied, reference=res.reference)
            return {
                "result": {"reference": res.reference, "applied": res.applied},
                "final_response": f"Done. {WRITE_CONFIRMATION} {res.reference}.",
            }
        except ConfirmationError as exc:
            tracer.open("execute_action", "node", state.get("trace_root"), applied=False, reason=str(exc))
            return {"final_response": f"{HANDOFF_PREFIX} {exc}"}

    def pre_render_guard(state: AtlasState) -> dict:
        text = getattr(state["messages"][-1], "content", "") or ""
        cid = state["session"]["customer_id"]
        root = state.get("trace_root")
        # fail closed, in order: unsafe markup / secret leak, other customer data, then grounded vs true
        for verdict in (
            guardrules.check_render_safe(text),
            guardrules.check_no_other_customer(text, cid),
            guardrules.check_render_truth(text, cid),
        ):
            if not verdict.ok:
                tracer.open("pre_render_guard", "guard", root, ok=False, reason=verdict.reason)
                return {"final_response": f"{HANDOFF_PREFIX} {verdict.reason}; let me get a person."}
        tracer.open("pre_render_guard", "guard", root, ok=True, reason="")
        # safe to ship → memoize under THIS turn's question. Only a knowledge only, account free
        # answer is shared as generic; anything that touched the account is keyed per customer.
        question = state.get("turn_question")
        if question is not None:
            generic = bool(state.get("used_knowledge")) and not state.get("used_account")
            cache.put(cid, question, text, generic=generic)
        tracer.open("render", "node", root, output=text)
        return {"final_response": text}

    def route_after_agent(state: AtlasState) -> str:
        calls = getattr(state["messages"][-1], "tool_calls", None)
        if not calls:
            return "render"
        if any(c["name"] in guardrules.WRITE_TOOLS for c in calls):
            return "act"
        return "read"

    def route_after_read(state: AtlasState) -> str:
        # a binding block sets final_response and ends; otherwise loop back for the model to answer
        return "blocked" if state.get("final_response") else "loop"

    g = StateGraph(AtlasState)
    g.add_node("agent", agent)
    g.add_node("tools_read", tools_read)
    g.add_node("pre_action_guard", pre_action_guard)
    g.add_node("confirm", confirm)
    g.add_node("pre_render_guard", pre_render_guard)
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"render": "pre_render_guard", "read": "tools_read", "act": "pre_action_guard"},
    )
    g.add_conditional_edges("tools_read", route_after_read, {"blocked": END, "loop": "agent"})
    g.add_edge("pre_action_guard", "confirm")
    g.add_edge("confirm", END)
    g.add_edge("pre_render_guard", END)
    return g.compile(checkpointer=checkpointer)
