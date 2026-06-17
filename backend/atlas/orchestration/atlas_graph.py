"""The unified Atlas graph: the agent decides, and routing splits answer / read / act.

Writes pass the fail closed `pre_action_guard` and the confirmation `interrupt()`; answers pass
`pre_render_guard`, whose truth check is a cue-based heuristic (`guard.check_render_truth`), not
structured claim extraction. Identity lives in the non model `session` channel, never a tool
argument. The read loop and the write both call the ONE graded budget function
(`atlas.domain.budget.check_budget`) over the turn's running tool-call sequence, before executing,
so the runtime never ships a turn that check would fail; the recursion limit is a langgraph backstop
sized from the same budget, not its implicit default.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Annotated, Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from langgraph.types import interrupt
from mcp.shared.memory import create_connected_server_and_client_session

from atlas.domain import guard as guardrules
from atlas.domain.actions import ActionsBackend
from atlas.domain.binding import CATALOG_TOOLS, KNOWLEDGE_TOOLS, bound_tools, classify_intent, is_reachable
from atlas.domain.budget import DEFAULT_BUDGET, DEFAULT_RETRIEVAL_TOOLS, RECURSION_LIMIT, check_budget
from atlas.domain.cache import PerCustomerCache
from atlas.domain.confirmation import ConfirmationError, PendingAction, execute_if_confirmed
from atlas.domain.degradation import DEGRADATION_MODE_NONE, DEGRADED_RESULT_KEY, escalate
from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.adapters.pgvector_retriever import PgvectorRetriever
from atlas.adapters.resilience import CircuitBreaker, ProviderError, RetryPolicy, call_with_resilience_async
from atlas.mcp_servers.account_server import build_account_server
from atlas.mcp_servers.actions_server import build_actions_server
from atlas.mcp_servers.catalog_server import build_catalog_server
from atlas.mcp_servers.knowledge_server import build_knowledge_server
from determinism.canonical import serialize_tool_result
from tracing import NullTracer

# The outcome sentinels the runtime ships and the eval/drift lanes read back. ONE definition: a
# reword here updates every grader that keys off it, instead of silently failing those graders open
# (a safety eval that stops matching "[safe handoff]" would quietly certify the refusals it exists
# to catch). The drift lane reads the outcome from the span tree, not these strings.
HANDOFF_PREFIX = "[safe handoff]"
WRITE_CONFIRMATION = "Your reference is"

# fee_outcome -> the sentence the confirmation message states for it. Deliberately a lookup, not a
# per-tool branch: any future proposal field that should reach the confirmation text is a new key
# here, not a new tool-name check in confirm() itself. Only cancel_service produces fee_outcome
# today (no third field to generalize over yet), so this stays narrow rather than a general
# "render any proposal field" framework nothing else needs.
_FEE_OUTCOME_DETAIL = {
    "waived_pending_verification": " Your early-termination fee has been waived, pending verification.",
    "standard": " The standard early-termination fee applies.",
}


def _confirmation_detail(proposal: str) -> str:
    """Tool-specific detail to fold into the generic 'Done, reference' confirmation, read from the
    proposal the MCP tool itself already returned pre-confirmation -- never recomputed here, so
    this can never state something different from what the customer already saw proposed. Found
    live-validating cancel_service (SP: the confirmation said only 'Done, your reference is X',
    never mentioning the fee outcome the backend had already correctly computed): the generic
    template threw the information away instead of using it. Returns "" for a proposal with no
    known extra field, which is every write tool except cancel_service today, so their confirmation
    text is unchanged."""
    try:
        data = json.loads(proposal)
    except (json.JSONDecodeError, TypeError):
        return ""
    return _FEE_OUTCOME_DETAIL.get(data.get("fee_outcome"), "")

# The degradation ladder's terminal rung (SP4 task 4): the honest refusal message, fixed rather
# than model composed, since there is nothing grounded left to compose an answer from once the
# ladder is exhausted. Routed through `_safe_handoff` (below) like every other guard failure, so a
# safety eval keyed off `HANDOFF_PREFIX` still catches it.
_REFUSAL_MESSAGE = "I do not have a reliable answer for that right now; let me get a person."


def thread_config(thread_id: str) -> dict:
    """The ONE invoke config for the graph: the thread id plus the recursion limit tied to the call
    budget (`atlas.domain.budget.RECURSION_LIMIT`), not langgraph's implicit 25. Every caller that
    drives the graph, the product edge AND the eval/drift/simulation/trajectory lanes, goes through
    here, so the graded lanes exercise the graph under the EXACT superstep ceiling production runs
    (finding 2): a bound that only the product edge honoured would be an untested production setting.
    The thread id is namespaced by the caller (the product edge prefixes the customer id); this helper
    does not namespace, so an eval lane's plain thread id is passed through unchanged.
    """
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


# ATLAS_RETRIEVER maps 1:1 to which adapter backs the knowledge port, no translation, mirroring
# server.py's ATLAS_MODE convention (`_resolve_mode`). "inmemory" (the default, including when the
# env var is unset) is the hermetic CI adapter every test and `task test` implicitly relies on;
# "pgvector" (D36 tier 2, SP3 task 7) is the real hybrid adapter over Postgres + TEI, wired by the
# served app (`server.py`) so `docker compose up` gives real local retrieval out of the box.
_KNOWN_RETRIEVERS = ("inmemory", "pgvector")


def select_retriever(kind: str | None = None):
    """Adapter selection for the knowledge port (D36 tier 2). `kind` defaults to `ATLAS_RETRIEVER`
    (unset -> "inmemory"), so every hermetic test and eval lane that never sets the env var keeps
    getting `InMemoryRetriever` untouched -- this function is additive, not a replacement for
    `build_atlas_graph`'s own `retriever=None -> InMemoryRetriever()` fallback, which still serves
    any caller (tests, eval lanes) that constructs the graph directly without going through here.
    `PgvectorRetriever()` reads its OWN env (`ATLAS_PG_DSN`/`ATLAS_TEI_EMBED_URL`/
    `ATLAS_TEI_RERANK_URL`/`ATLAS_INDEX_DIR`), so this function only decides WHICH class to build,
    never how to configure it. A typo'd value fails fast (matching `_resolve_mode`'s discipline)
    rather than silently falling back to the hermetic adapter, which would mask a broken deployment
    as "it works" while quietly serving the toy corpus."""
    kind = kind or os.environ.get("ATLAS_RETRIEVER", "inmemory")
    if kind == "inmemory":
        return InMemoryRetriever()
    if kind == "pgvector":
        return PgvectorRetriever()
    raise RuntimeError(f"unknown ATLAS_RETRIEVER={kind!r}; expected one of {'|'.join(_KNOWN_RETRIEVERS)}")


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
    tools_called: Optional[tuple[str, ...]]  # names of the tool calls executed THIS turn, in order (reset
    #                                          per fresh turn). The SINGLE per-turn tally the graded
    #                                          check_budget reads, so the runtime and the grade share one
    #                                          sequence; a write appends its name here before it executes.
    account_seen: Optional[bool]    # STICKY across the thread: the account was read on some earlier
    #                                 turn, so a later knowledge-only turn may restate it and is not
    #                                 shareable as generic (unlike used_account, this is NOT reset per turn)
    remaining_steps: RemainingSteps  # managed by langgraph: supersteps left under the recursion limit
    degradation_mode: Optional[str]  # SP4 task 4: this turn's worst ladder rung fired so far ("none"
    #                                  default per turn); every transition sets it via escalate() (last
    #                                  rung wins, but only upward -- a lower rung never overwrites a higher one)
    needs_refusal: Optional[bool]    # internal routing signal only (per turn, not part of the ladder's own
    #                                  contract): a transition wants this turn routed to the `refusal` node


def _text_of(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _mcp_result(res) -> tuple[str, str]:
    """The isError backstop, shared by every MCP dispatcher (SP4 task 5 fix round 1: originally
    landed only on `_knowledge_call`; the catalog routing gap this same task fixed elsewhere is the
    exact failure mode this closes everywhere else too). `res.isError` means FastMCP's own generic
    `except Exception` handler already caught and stringified an UNCLASSIFIED failure inside the
    tool (`knowledge_server.py`'s module docstring names the exact swallow); treating that
    stringified text as ordinary tool content would be the fail OPEN a caller must never make.
    Returns `(text, DEGRADATION_MODE_NONE)` on an ordinary result, `("", "refusal")` on isError --
    the SAME two state values `_knowledge_call` already used, reused here rather than inventing a
    second vocabulary for the identical idea."""
    if res.isError:
        return "", "refusal"
    return res.content[0].text, DEGRADATION_MODE_NONE


async def _account_call(customer_id: str, tool_name: str, args: dict) -> tuple[str, str]:
    server = build_account_server(customer_id)
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        res = await client.call_tool(tool_name, args or {})
        return _mcp_result(res)


async def _catalog_call(tool_name: str, args: dict) -> tuple[str, str]:
    """SP4 task 5 fix: `tools_read`'s read branch used to route EVERY non knowledge tool call to
    `_account_call`, so a catalog tool (`list_plans`/`get_plan`/`compute_price`/`check_eligibility`,
    reachable per `domain.binding.CATALOG_TOOLS` on a policy_question/account_read/action turn)
    would 404 as "Unknown tool" on the account server, which never declares them -- a real,
    previously untested gap between what binding.py DECLARES reachable and what the graph actually
    WIRES (see the repo's own CLAUDE.md on this exact failure shape). `catalog_server` carries no
    identity (D8: catalog data is public), so this needs no customer_id at all."""
    server = build_catalog_server()
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        res = await client.call_tool(tool_name, args or {})
        return _mcp_result(res)


async def _knowledge_call(retriever, query: str) -> tuple[str, str]:
    """Search the knowledge server and read back the ladder's own signal (SP4 task 4). Returns
    `(tool_result_text, degradation_mode)`. `mode` is `"none"` on an ordinary successful search --
    the overwhelmingly common hermetic path (`InMemoryRetriever` never raises a typed error and
    never reports a retry), where `tool_result_text` is returned UNPARSED, byte identical to the
    pre ladder behaviour. `"retry"`/`"drop_rerank"`/`"lexical_only"` mean the search server's own
    ladder walk (`knowledge_server.search_knowledge`) recovered; `tool_result_text` is the plain
    passages array the model has always seen, serialized again out of the degraded envelope.
    `"refusal"` means the ladder was exhausted: `tool_result_text` is empty and the caller must not
    treat this as an answerable tool result.

    SP4 task 5 ride along, the isError backstop: `_mcp_result` (shared by every dispatcher as of fix
    round 1, originally landed only here) is checked BEFORE anything else. An UNCLASSIFIED exception
    inside `search_knowledge` (anything the ladder's own try/except in knowledge_server.py does not
    recognize -- a bug, a bare `KeyError`, anything that is not
    `RerankServiceError`/`EmbeddingServiceError`/`RetrievalError`) never reaches this function as a
    typed error: FastMCP's own `call_tool` handler already caught it and stringified it into the
    result text, with `isError=True` (this module's own docstring names that swallow). Treating that
    text as content in that case -- parsing it, or worse, falling through to the plain text happy
    path below -- would compose an answer over the exception's own text: the exact fail OPEN the
    degradation ladder (Task 4) exists to close. Every `isError` result routes straight to refusal
    instead, never inspected further."""
    server = build_knowledge_server(retriever)
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        res = await client.call_tool("search_knowledge", {"query": query})
    raw, mode = _mcp_result(res)
    if mode == "refusal":
        return "", "refusal"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw, DEGRADATION_MODE_NONE
    if not isinstance(parsed, dict) or not parsed.get(DEGRADED_RESULT_KEY):
        return raw, DEGRADATION_MODE_NONE
    mode = parsed.get("degradation_mode", DEGRADATION_MODE_NONE)
    if mode == "refusal":
        return "", "refusal"
    return serialize_tool_result(parsed.get("passages", [])), mode


async def _actions_call(customer_id: str, tool: str, args: dict) -> tuple[str, str]:
    """Materialize the write proposal through the customer scoped actions MCP server (the real
    write surface). Identity is bound at connect, so customer_id is never a tool argument. Returns
    `(text, mode)` like every other dispatcher (SP4 task 5 fix round 1, the isError backstop
    generalized past `_knowledge_call`): mode is `"refusal"` on an isError result, else
    `DEGRADATION_MODE_NONE`; the caller (`pre_action_guard`) fails closed via `_safe_handoff` on
    refusal rather than materializing a `pending` write proposal from stringified error text."""
    server = build_actions_server(customer_id)
    call_args = {k: v for k, v in (args or {}).items() if k != "customer_id"}
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        res = await client.call_tool(tool, call_args)
        return _mcp_result(res)


def _tool_bindable(model: BaseChatModel):
    """The live inner provider a gateway wraps in record/live mode, duck typed off the SAME two
    plain fields `GatewayChatModel` already exposes (`.mode`, `.inner`,
    `testing/harness/replay/gateway.py`) -- never importing that class into
    backend/atlas/orchestration (harness may import backend, never the reverse). Mirrors
    `knowledge_server.py`'s `_retried()` accessor, this codebase's own established "best effort duck
    typed read of an adapter's own carrier" pattern. Returns None in replay/hermetic mode
    (`GatewayChatModel.inner` is None there) AND for every plain `BaseChatModel` test fake that
    carries neither attribute (`getattr` defaults both to None), so the hermetic path never even
    reaches `.bind_tools()`.

    SP4 task 5 fix round 1 (reviewer finding): the mode comparison is EQUALITY, never `str()`.
    `GatewayMode` is `class GatewayMode(str, enum.Enum)`; for this mixin shape, `str(GatewayMode.
    REPLAY)` is `"GatewayMode.REPLAY"` (`Enum.__str__` wins over the `str` mixin for `str()` calls,
    a well known enum gotcha), never equal to the plain string `"replay"`, so a `str(...) ==
    "replay"` comparison never matches a real `GatewayMode` value. Equality does not have this
    problem: `GatewayMode.REPLAY == "replay"` is `True` (the `str` mixin backs `__eq__`), and this
    also still matches a plain string `.mode` (a test fake that never imports `GatewayMode` at
    all), so dropping `str()` entirely fixes both the real enum AND keeps every existing fake
    working unchanged."""
    inner = getattr(model, "inner", None)
    if inner is None:
        return None
    if getattr(model, "mode", "replay") == "replay":
        return None
    return inner


def _bound_tool_specs(mcp_tools: dict[str, dict] | None, intent: str) -> list[dict]:
    """The turn's own least agency, applied to real tool binding (SP4 task 5): only the tools
    `domain.binding.bound_tools(intent)` already declares reachable for THIS intent are ever handed
    to the model, the same reachable set `tools_read`/`pre_action_guard` enforce downstream by
    intercept. `domain/binding.py`'s own docstring named this exact moment: "A dev/prod build would
    also hand the model only the bound tools, so the capability is simply absent" -- this is that
    dev/prod build."""
    if not mcp_tools:
        return []
    reachable = bound_tools(intent)
    return [spec for name, spec in mcp_tools.items() if name in reachable]


async def _generate_message(
    model: BaseChatModel, messages: list[BaseMessage], mcp_tools: dict[str, dict] | None, intent: str,
    *, breaker: CircuitBreaker | None = None, policy: RetryPolicy | None = None,
    provider_key: str = "primary-model",
) -> BaseMessage:
    """SP4 task 5: the ONE seam that decides whether a generation call binds real tools.

    Live/record mode (`model` duck types a live `.inner`, see `_tool_bindable`) with a non empty,
    intent scoped tool surface: bind those tools onto the REAL inner provider model (its own
    `bind_tools`, not the gateway's -- `GatewayChatModel` inherits `BaseChatModel.bind_tools`
    unmodified, which raises `NotImplementedError`; going through `.inner` sidesteps that without
    touching `testing/harness/replay/gateway.py`) and drive it with `.ainvoke`, which returns the
    `AIMessage` directly, real `tool_calls` included when the model chose to call one. Live/record
    with no tools bound this turn falls to `model._agenerate(messages)` instead (the gateway's own
    call, so RECORD mode still persists the cassette) -- the OTHER live generation shape, sibling to
    the bound one above, never the replay path below.

    Replay/hermetic mode, or any caller with no `mcp_tools` at all (`mcp_tools=None` is
    `build_atlas_graph`'s own default, so every existing caller is unaffected): BYTE IDENTICAL to
    before this task -- `model._agenerate(messages)`, no `tools` kwarg, UNWRAPPED by anything below.
    The gateway's cassette key (`replay/cassette.py`'s `build_request`) never sees a `tools` kwarg
    either, so the cassette shape is unchanged, per D19's deferral (recorded cassettes WITH real
    tool_calls are a record time concern, deferred).

    SP4 final fix wave (F2): `breaker`/`policy`, when given, wrap ONLY the live/record call (either
    shape above) in `call_with_resilience_async` -- the SAME classification table the retrieval
    seams use (`resilience.py`), translating a raw provider/SDK exception into `ProviderError`
    before it ever reaches `agent()`'s `except ProviderError`, so a live provider outage finally
    walks the ladder (retry inside this seam, then `provider_fallback`/`refusal` one level up in
    `agent()`) instead of an unhandled exception. Conditioned on `inner is not None`, exactly like
    the tool binding choice above: the replay path is never wrapped, so a `CassetteMiss` (the
    deliberate hard fail on a recording gap) is never reclassified as a retryable provider failure.
    Both default to `None` so a caller that omits them keeps the behaviour from before this fix wave (no caller
    in this codebase does today, but the parameters stay optional rather than required)."""
    inner = _tool_bindable(model)
    specs = _bound_tool_specs(mcp_tools, intent) if inner is not None else []

    async def _live_call() -> BaseMessage:
        if specs:
            bound = inner.bind_tools(specs)
            return await bound.ainvoke(messages)
        result = await model._agenerate(messages)
        return result.generations[0].message

    if inner is not None:
        if breaker is not None and policy is not None:
            return await call_with_resilience_async(
                _live_call, policy=policy, breaker=breaker, provider_key=provider_key,
            )
        return await _live_call()
    result = await model._agenerate(messages)
    return result.generations[0].message


def build_atlas_graph(
    model: BaseChatModel, ids, backend: ActionsBackend, checkpointer,
    retriever=None, tracer=None, cache=None, fallback_model: BaseChatModel | None = None,
    mcp_tools: dict[str, dict] | None = None, generation_clock: Callable[[], float] | None = None,
):
    """`fallback_model` (SP4 task 4, `provider_fallback`): an optional second chat model tried ONCE
    when `model` raises a retryable `ProviderError` (never on a non retryable one, which routes
    straight to refusal). Defaults to None (no fallback configured), so every existing caller is
    unaffected; live wiring (a real second provider) reads `ATLAS_FALLBACK_MODEL`, `server.py`'s
    concern (SP4 final fix wave, F2) -- this parameter only carries the routing logic the hermetic
    lane asserts against fakes.

    `mcp_tools` (SP4 task 5, bind_tools): the full MCP tool surface
    (`atlas.mcp_servers.tool_surface.mcp_tool_surface()`), name -> bind_tools compatible dict.
    Defaults to None, so every existing caller (every hermetic test, and the served app in replay
    mode) is unaffected -- see `_generate_message`. `server.py` is the only caller that passes one,
    and only when `ATLAS_MODE` is not replay.

    `generation_clock` (SP4 final fix wave, F2): the injected callable the generation seam's own
    `CircuitBreaker` reads (`time.monotonic`'s own shape), defaulting to `time.monotonic` only HERE
    -- the same "inject with a live default, never inside the primitive itself" discipline
    `PgvectorRetriever`'s own `clock` parameter documents -- so a hermetic test that needs to walk
    the generation breaker's state machine deterministically can still pass a fake one, the same way
    `test_fault_lane.py` already does for the retrieval seams' breakers."""
    retriever = retriever or InMemoryRetriever()
    tracer = tracer or NullTracer()  # observation is opt in, but the runtime is always instrumented
    cache = cache or PerCustomerCache()  # per customer keying, a generic answer is shared, a bill is not
    # SP4 final fix wave (F2): the generation seam's own resilience wiring, one breaker/policy pair
    # per compiled graph (mirrors PgvectorRetriever's own per adapter instance state, never a module
    # level singleton that would leak breaker state across independently built graphs/tests). Only
    # ever exercised when a call actually reaches the live/record path (`_generate_message`'s own
    # `inner is not None` gate); replay/hermetic calls never touch this.
    generation_breaker = CircuitBreaker(generation_clock or time.monotonic)
    generation_retry_policy = RetryPolicy()

    def _safe_handoff(reason: str) -> dict:
        """A fail-closed handoff whose reason is run through the SAME output escaper the render path
        uses. A refusal interpolates model-controlled fragments (a tool name, a rejected argument via
        `!r`), so the 'escape at the door' rule must hold here too, not only on the model answer: an
        injected `<img ...>` in a plan id never reaches the reply verbatim. If the reason is unsafe,
        it is dropped for a fixed string (the specifics stay in the trace span, not the user text)."""
        safe = guardrules.check_render_safe(reason)
        text = reason if safe.ok else "that request is not available here"
        return {"final_response": f"{HANDOFF_PREFIX} {text}"}

    def _abandon_batch(last: BaseMessage, user_reason: str, why: str) -> dict:
        """Fail closed on a model tool-call batch we refuse to run: hand off with `user_reason`, AND
        answer every pending tool_call with a ToolMessage. An AIMessage whose tool_calls go unanswered
        is malformed history: under the checkpointer it persists and corrupts the NEXT live-mode turn's
        model input (the provider rejects a tool_call with no matching tool result). `why` is the
        internal reason recorded on each ToolMessage (the user-facing text stays in `user_reason`)."""
        not_run = [
            ToolMessage(content=f"not executed: {why}", tool_call_id=tc["id"], name=tc["name"])
            for tc in last.tool_calls
        ]
        return {**_safe_handoff(user_reason), "messages": not_run}

    async def agent(state: AtlasState) -> dict:
        messages = state["messages"]
        # A fresh user turn is the only hop whose last message is a HumanMessage. A tool loop
        # re entry ends in a ToolMessage. Under the checkpointer, state persists across turns on a
        # thread, so the turn span / cache lookup / per turn flags must reset HERE, per turn, keying
        # off the CURRENT question, never the conversation's first one.
        fresh_turn = bool(messages) and isinstance(messages[-1], HumanMessage)
        extra: dict = {}
        if fresh_turn:
            question = _text_of(messages[-1])
            # intent is set per turn (caller may pin it on the session, else classified) and binds
            # which tools are reachable for the whole turn, least agency, decided before the model acts.
            intent = state["session"].get("intent") or classify_intent(question)
            # record the BOUND intent on the turn span, so a grader/drift lane reads what the runtime
            # actually bound (and could see it move), not a fresh derivation from the raw utterance.
            # customer_id (SP8 task 1, ADR-029): from the session/bearer identity, NEVER a tool
            # argument and never the model (CLAUDE.md's own rule) -- trace_translation.py HMACs it
            # into atlas.subject.pseudonym at export, so the raw id never reaches an exported span.
            root = tracer.open(
                "turn", "turn", input=question, intent=intent,
                customer_id=state["session"]["customer_id"],
            )
            # Reset the per-turn terminal channels too: under the checkpointer a prior turn's
            # final_response / pending / result persist on the thread, and a read-path turn routes on
            # `final_response` (route_after_read), so a stale one would end turn 2 with turn 1's answer.
            extra = {"trace_root": root, "turn_question": question, "intent": intent,
                     "used_account": False, "used_knowledge": False,
                     "tools_called": (),
                     "final_response": None, "pending": None, "result": None,
                     "degradation_mode": DEGRADATION_MODE_NONE, "needs_refusal": False}
            for generic in (False, True):  # customer specific key first, then the shared generic key
                hit = cache.get(state["session"]["customer_id"], question, generic=generic)
                if hit is not None:
                    tracer.open("cache", "node", root, hit=True, generic=generic)
                    # the cached reply still passes through the render guard before it ships
                    return {"messages": [AIMessage(content=hit)], **extra}
        else:
            root = state.get("trace_root")
        # SP4 task 5: the SAME intent tools_read/pre_action_guard already fall back to when a loop
        # re entry has none on state -- resolved here once, before generation, so bind_tools (below)
        # and the downstream guards agree on exactly what this turn was allowed to reach.
        turn_intent = extra.get("intent") or state.get("intent") or "troubleshooting"
        # SP4 task 4: a typed, non retryable ProviderError from generation routes to the refusal
        # node (mode "refusal", stamped there); a retryable one falls over to `fallback_model` ONCE
        # if one is configured (mode "provider_fallback"), else it also routes to refusal -- there
        # is nothing else this node can try. No AIMessage is appended on either refusal path (the
        # `messages` key is simply omitted), so no dangling tool_calls / malformed history risk.
        try:
            message = await _generate_message(
                model, list(messages), mcp_tools, turn_intent,
                breaker=generation_breaker, policy=generation_retry_policy, provider_key="primary-model",
            )
        except ProviderError as primary_exc:
            if primary_exc.retryable and fallback_model is not None:
                try:
                    message = await _generate_message(
                        fallback_model, list(messages), mcp_tools, turn_intent,
                        breaker=generation_breaker, policy=generation_retry_policy, provider_key="fallback-model",
                    )
                except Exception as fallback_exc:
                    tracer.open("agent_failure", "guard", root, ok=False,
                                reason=f"primary and fallback generation both failed: {fallback_exc}")
                    return {**extra, "needs_refusal": True}
                # SP4 final fix wave (F1): resolved BEFORE merge, the same pattern `turn_intent`
                # already uses one line above -- a fresh turn's per turn reset lives in `extra`
                # (this function's own local dict), which has not merged into `state` yet at this
                # point in the SAME invocation. Reading `state` first would escalate from the
                # PREVIOUS turn's PERSISTED degradation_mode on this thread (under the checkpointer,
                # a worse prior turn -- refusal, say -- would wrongly stamp a later, healthy
                # fallback turn as "refusal" instead of "provider_fallback"). `extra`'s own reset
                # always wins on a fresh turn; a loop re entry within the SAME turn has no `extra`
                # reset (fresh_turn was False), so it correctly falls through to `state`.
                mode = escalate(
                    extra.get("degradation_mode") or state.get("degradation_mode") or DEGRADATION_MODE_NONE,
                    "provider_fallback",
                )
                seq = tracer.open("agent", "llm", root, model=getattr(fallback_model, "model_id", "fallback"),
                                   degradation_mode=mode)
                return {"messages": [message], "trace_parent": seq,
                        **extra, "degradation_mode": mode}
            tracer.open("agent_failure", "guard", root, ok=False, reason=str(primary_exc),
                        retryable=primary_exc.retryable)
            return {**extra, "needs_refusal": True}
        seq = tracer.open("agent", "llm", root, model=getattr(model, "model_id", "model"))
        return {"messages": [message], "trace_parent": seq, **extra}

    async def tools_read(state: AtlasState) -> dict:
        last = state["messages"][-1]
        cid = state["session"]["customer_id"]
        intent = state.get("intent") or "troubleshooting"
        root = state.get("trace_root")
        # least agency: a tool not bound to this turn's intent is unreachable, not merely guarded
        unreachable = [tc["name"] for tc in last.tool_calls if not is_reachable(intent, tc["name"])]
        if unreachable:
            tracer.open("bind_guard", "guard", root, ok=False, intent=intent, tools=unreachable)
            return _abandon_batch(last, f"{unreachable[0]} is not available on a {intent} turn",
                                  f"{unreachable[0]} unreachable on a {intent} turn")
        # rounds backstop: another read round trip costs 2 supersteps (agent + tools_read), which the
        # call-count budget below cannot see. The recursion limit is sized so a budget-legal turn always
        # fits, so this only bites a loop the budget already refuses; it still ends the turn in the
        # guarded handoff path rather than letting langgraph raise GraphRecursionError mid turn.
        if state.get("remaining_steps", RECURSION_LIMIT) <= 2:
            reason = f"remaining_steps={state.get('remaining_steps')} cannot fit another read round trip"
            tracer.open("budget_guard", "guard", root, ok=False, reason=reason)
            return _abandon_batch(last, "this is taking more steps than a single turn allows", reason)
        # THE per-turn budget: the ONE graded function (atlas.domain.budget.check_budget) over the turn's
        # running tool-call sequence PLUS this batch, before executing. Bounds the whole turn, not one
        # batch, so a small-batch retry storm across rounds (2 calls x 4 rounds spends 8) and an oversized
        # single batch (a 4-search retrieval storm) are BOTH caught here, by the same arithmetic the
        # monitor/trajectory lanes grade. Failing closed before the tool spans open is what keeps the
        # shipped trace within budget, so the grade can never call a turn the runtime shipped a breach.
        called = state.get("tools_called") or ()
        batch_names = tuple(tc["name"] for tc in last.tool_calls)
        report = check_budget(called + batch_names, DEFAULT_BUDGET, retrieval_tools=DEFAULT_RETRIEVAL_TOOLS)
        if not report.ok:
            reason = "; ".join(report.reasons)
            tracer.open("budget_guard", "guard", root, ok=False, reason=reason)
            return _abandon_batch(last, "that needs more tool calls this turn than the budget allows", reason)
        parent = state.get("trace_parent")
        used_knowledge = state.get("used_knowledge") or False
        used_account = state.get("used_account") or False
        degradation_mode = state.get("degradation_mode") or DEGRADATION_MODE_NONE
        out: list[BaseMessage] = []

        def _refusal_mid_batch(index: int, tc: dict, content: str, reason: str) -> dict:
            """SP4 task 5 fix round 1: the isError backstop generalized past knowledge alone, shared
            by every dispatcher's refusal (`_knowledge_call`'s already existing ladder exhausted shape,
            now also `_account_call`/`_catalog_call`). `out` (closed over) holds the REAL
            ToolMessages already produced for calls BEFORE `index` in this batch -- unlike
            `_abandon_batch`, which assumes nothing has run yet -- so only `tc` itself and everything
            AFTER it are marked failed/not executed; a caller (the model) never sees a dangling
            tool_call with no matching ToolMessage."""
            tracer.open("refusal_trigger", "guard", root, ok=False, reason=reason, tool=tc["name"])
            failed = ToolMessage(content=content, tool_call_id=tc["id"], name=tc["name"])
            not_run = [
                ToolMessage(content="not executed: a prior tool call in this batch triggered a refusal",
                            tool_call_id=other["id"], name=other["name"])
                for other in last.tool_calls[index + 1:]
            ]
            return {
                "messages": out + [failed] + not_run,
                "used_knowledge": used_knowledge, "used_account": used_account,
                "account_seen": bool(state.get("account_seen")) or used_account,
                "tools_called": called + batch_names,
                "needs_refusal": True,
            }

        for index, tc in enumerate(last.tool_calls):
            assemble_seq = None
            if tc["name"] in KNOWLEDGE_TOOLS:
                # SP6 task 2: the RAG pipeline's own stage durations (`atlas.stage.embed_ms`/
                # `retrieve_ms`/`rerank_ms`), opened around the ONE MCP round trip this graph can
                # actually see (`_knowledge_call`, which hides the retriever's own embed/retrieve/
                # rerank breakdown behind its boundary -- a disclosed simplification, not a guess;
                # see `trace_translation.py`'s module docstring). All three measure the SAME real
                # elapsed interval; which ones get CLOSED (and therefore exported -- the OTel
                # adapter never exports a span that never closes) depends on `local_mode`: a rung
                # this call skipped never reports a duration for the stage it skipped, matching the
                # `degraded_turn.json` golden example (no `atlas.stage.rerank_ms` on a drop_rerank
                # turn).
                embed_seq = tracer.open("embed", "stage", parent)
                retrieve_seq = tracer.open("retrieve", "stage", parent)
                rerank_seq = tracer.open("rerank", "stage", parent)
                text, local_mode = await _knowledge_call(retriever, tc.get("args", {}).get("query", ""))
                tracer.close(retrieve_seq)  # retrieval is always attempted
                if local_mode != "lexical_only":  # lexical_only is the one rung that skips embed
                    tracer.close(embed_seq)
                if local_mode != "drop_rerank":  # drop_rerank is the one rung that skips rerank
                    tracer.close(rerank_seq)
                used_knowledge = True
                if local_mode == "refusal":
                    # SP4 task 4: the degradation ladder is exhausted for this search -- route to the
                    # refusal node rather than looping the model on an empty/degraded tool result (a
                    # confident sounding answer grounded in nothing). `degradation_mode` is left
                    # untouched here: the refusal node stamps "refusal" itself (last rung wins,
                    # always upward).
                    return _refusal_mid_batch(index, tc, "retrieval unavailable: degradation ladder exhausted",
                                               "knowledge retrieval exhausted the degradation ladder")
                degradation_mode = escalate(degradation_mode, local_mode)
                # "assemble": composing the retrieved passages into this tool call's result, the one
                # RAG stage this graph genuinely CAN measure on its own (unlike embed/retrieve/rerank
                # above), bracketing the shared tail below (tool span open + ToolMessage append).
                assemble_seq = tracer.open("assemble", "stage", parent)
            elif tc["name"] in CATALOG_TOOLS:
                # SP4 task 5 fix: catalog reads used to fall into the `else` branch below and get
                # routed to the ACCOUNT server, which never declares list_plans/get_plan/
                # compute_price/check_eligibility (an "Unknown tool" MCP error, silently treated as
                # if it were a real answer -- see `_catalog_call`'s own docstring). Catalog data is
                # public and customer independent (D8), so this deliberately does NOT set
                # used_account: a catalog only turn stays eligible for the generic cache key, the
                # same as it always implicitly was before this tool was reachable at all.
                text, local_mode = await _catalog_call(tc["name"], tc.get("args", {}))
                if local_mode == "refusal":
                    return _refusal_mid_batch(index, tc, f"{tc['name']} unavailable: unclassified MCP error",
                                               f"{tc['name']} returned an unclassified MCP error")
            else:
                text, local_mode = await _account_call(cid, tc["name"], tc.get("args", {}))
                used_account = True  # answer now depends on this customer, never shared as generic
                if local_mode == "refusal":
                    return _refusal_mid_batch(index, tc, f"{tc['name']} unavailable: unclassified MCP error",
                                               f"{tc['name']} returned an unclassified MCP error")
            tracer.open(tc["name"], "tool", parent, args=tc.get("args", {}), result=text)
            out.append(ToolMessage(content=text, tool_call_id=tc["id"], name=tc["name"]))
            if assemble_seq is not None:
                tracer.close(assemble_seq)
        # account_seen is sticky at thread scope: once True it stays True across turns (never reset in
        # the fresh-turn `extra`), so a later knowledge-only turn restating account data is not shared.
        return {"messages": out, "used_knowledge": used_knowledge, "used_account": used_account,
                "account_seen": bool(state.get("account_seen")) or used_account,
                "tools_called": called + batch_names, "degradation_mode": degradation_mode}

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
            return _safe_handoff(f"{unreachable[0]} is not available on a {intent} turn")
        single = guardrules.check_single_write(names)
        if not single.ok:  # a multi or mixed read+write batch fails closed before anything runs
            tracer.open("pre_action_guard", "guard", root, ok=False, reason=single.reason)
            return _safe_handoff(single.reason)
        tc = last.tool_calls[0]
        args = tc.get("args", {})
        # an id the model tried to put in the args is rejected unless it matches the session
        scope = guardrules.check_scope(args.get("customer_id", cid), cid)
        if not scope.ok:
            tracer.open("pre_action_guard", "guard", root, ok=False, reason=scope.reason, tool=tc["name"])
            return _safe_handoff(scope.reason)
        bounds = guardrules.check_value_bounds(tc["name"], args)
        tracer.open("pre_action_guard", "guard", root, ok=bounds.ok, reason=bounds.reason, tool=tc["name"])
        if not bounds.ok:
            return _safe_handoff(bounds.reason)
        # the write counts against the SAME per-turn budget the reads do: run the one graded check over
        # the turn's sequence with this write appended, before materialising anything. A turn that read
        # up to the budget and then reaches for a write is over budget, and must fail closed here rather
        # than execute, so the repo's own claim (the runtime never ships a turn check_budget would fail)
        # holds on the write surface too, not only in the read loop.
        called = state.get("tools_called") or ()
        report = check_budget(called + (tc["name"],), DEFAULT_BUDGET, retrieval_tools=DEFAULT_RETRIEVAL_TOOLS)
        if not report.ok:
            tracer.open("budget_guard", "guard", root, ok=False, reason="; ".join(report.reasons), tool=tc["name"])
            return _safe_handoff("that needs more tool calls this turn than the budget allows")
        # materialize the proposal through the customer scoped actions MCP server (the write surface)
        proposal, local_mode = await _actions_call(cid, tc["name"], args)
        if local_mode == "refusal":
            # SP4 task 5 fix round 1: the isError backstop generalized to the write surface too.
            # Every OTHER guard failure in this function already fails closed via `_safe_handoff`
            # (single write, scope, value bounds, budget) rather than the read path's dedicated
            # `refusal` node/`needs_refusal` signal, so an unclassified actions server error is
            # routed the SAME way, consistent with every guard check above it, never a `pending`
            # write materialized from stringified error text.
            tracer.open("pre_action_guard", "guard", root, ok=False,
                        reason=f"{tc['name']} returned an unclassified MCP error", tool=tc["name"])
            return _safe_handoff(f"{tc['name']} is not available right now")
        tracer.open(tc["name"], "tool", state.get("trace_root"), args=args, proposal=proposal)
        return {
            # append the write to the turn's tally so the graded invariant and the runtime see the same
            # sequence: the write is spent budget the moment it is proposed, not only if it is confirmed.
            "tools_called": called + (tc["name"],),
            "pending": {
                "tool": tc["name"],
                "args": args,
                "idempotency_key": ids.next(),  # bound before the interrupt checkpoint
                "customer_id": cid,
                "proposal": proposal,
            },
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
            # the account changed: drop this customer's cached answers so a repeat question is
            # recomputed against fresh state instead of served the pre-write figure (read after write).
            cache.invalidate(p["customer_id"])
            detail = _confirmation_detail(p.get("proposal", ""))
            return {
                "result": {"reference": res.reference, "applied": res.applied},
                "final_response": f"Done.{detail} {WRITE_CONFIRMATION} {res.reference}.",
            }
        except ConfirmationError as exc:
            tracer.open("execute_action", "node", state.get("trace_root"), applied=False, reason=str(exc))
            return _safe_handoff(str(exc))

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
                return _safe_handoff(f"{verdict.reason}; let me get a person.")
        tracer.open("pre_render_guard", "guard", root, ok=True, reason="")
        # safe to ship → memoize under THIS turn's question. Only a knowledge only, account free
        # answer is shared as generic. Anything that touched the account is keyed per customer.
        question = state.get("turn_question")
        if question is not None:
            # generic (shareable across customers) only if this turn used knowledge, touched no account
            # THIS turn, AND the thread never read the account: a knowledge-only turn can still restate
            # account data from earlier in the conversation, so the sticky flag decides, not just this turn.
            generic = (bool(state.get("used_knowledge"))
                       and not state.get("used_account")
                       and not state.get("account_seen"))
            cache.put(cid, question, text, generic=generic)
        tracer.open("render", "node", root, output=text)
        return {"final_response": text}

    def refusal(state: AtlasState) -> dict:
        """The ladder's terminal rung (SP4 task 4): a still failing retrieval (`tools_read`) or a
        non retryable/fallback exhausted generation failure (`agent`) both route here rather than
        the model composing an answer from nothing. Produces the fixed honest refusal message
        (never model composed) and stamps `degradation_mode` via the SAME upward only escalation
        every other transition uses, so whatever rung this turn already reached (drop_rerank,
        lexical_only, ...) is preserved if a caller somehow reached here at a mode already at or
        above "refusal" -- which cannot happen today (refusal is the top rung) but keeps this node
        consistent with every other transition rather than a special cased overwrite."""
        root = state.get("trace_root")
        mode = escalate(state.get("degradation_mode") or DEGRADATION_MODE_NONE, "refusal")
        tracer.open("refusal", "node", root, degradation_mode=mode)
        return {**_safe_handoff(_REFUSAL_MESSAGE), "degradation_mode": mode}

    def route_after_agent(state: AtlasState) -> str:
        if state.get("needs_refusal"):
            return "refusal"
        calls = getattr(state["messages"][-1], "tool_calls", None)
        if not calls:
            return "render"
        if any(c["name"] in guardrules.WRITE_TOOLS for c in calls):
            return "act"
        return "read"

    def route_after_read(state: AtlasState) -> str:
        if state.get("needs_refusal"):
            return "refusal"
        # a binding block sets final_response and ends, otherwise loop back for the model to answer
        return "blocked" if state.get("final_response") else "loop"

    def route_after_action(state: AtlasState) -> str:
        # symmetric with route_after_read: a guard (binding, single-write, scope, value-bounds, budget)
        # failed closed and set final_response, so END here rather than spend a `confirm` superstep to
        # do nothing. Only a clean proposal (pending set, no final_response) goes on to the confirmation
        # interrupt. Ending directly also keeps an over-budget write's handoff inside the recursion limit.
        return "blocked" if state.get("final_response") else "confirm"

    g = StateGraph(AtlasState)
    g.add_node("agent", agent)
    g.add_node("tools_read", tools_read)
    g.add_node("pre_action_guard", pre_action_guard)
    g.add_node("confirm", confirm)
    g.add_node("pre_render_guard", pre_render_guard)
    g.add_node("refusal", refusal)
    g.add_edge(START, "agent")
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {"render": "pre_render_guard", "read": "tools_read", "act": "pre_action_guard", "refusal": "refusal"},
    )
    g.add_conditional_edges("tools_read", route_after_read, {"blocked": END, "loop": "agent", "refusal": "refusal"})
    g.add_conditional_edges("pre_action_guard", route_after_action, {"blocked": END, "confirm": "confirm"})
    g.add_edge("confirm", END)
    g.add_edge("pre_render_guard", END)
    g.add_edge("refusal", END)
    return g.compile(checkpointer=checkpointer)
