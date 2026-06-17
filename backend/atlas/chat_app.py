"""The product API the Vite SPA talks to (ADR-028): auth + chat over the Atlas graph.

Thin edge over `build_atlas_graph`, no new agent logic. Identity comes from the validated bearer
token, never the request body. `thread_id` is namespaced server side per customer so one customer
can never resume another's pending interrupt. The graph (with its checkpointer) is a single long
lived object, so a `/chat` interrupt and its `/chat/resume` share confirmation state.

`/chat` stays request/response (every existing test keeps working unchanged). `/chat/stream`
(SP4 task 6) is the streaming counterpart: an `sse-starlette` `EventSourceResponse` driven by
`graph.astream_events`, emitting the frozen vocabulary in `contracts/sse/schema.json`
(`message_start`, `token`, `citation`, `degradation`, `error`, `message_end`). See
`_run_stream`'s own docstring for the event mapping and the safety reasoning behind it.

The model + graph are injected so tests wire a replayed gateway and an in memory backend
(hermetic), exactly like the rest of the suite.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Callable

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from atlas.auth import TokenError, bearer_context, issue_token, validate_token
from atlas.domain.accounts import SEED
from atlas.domain.binding import KNOWLEDGE_TOOLS
from atlas.domain.degradation import DEGRADATION_MODE_NONE
from atlas.logging_setup import bind_trace_context
from atlas.orchestration.atlas_graph import HANDOFF_PREFIX, thread_config
from determinism.sources import IdFactory
from replay.gateway import CassetteMiss
from tracing import NullTracer

# "atlas.chat_app" (not "atlas"): a child of the SAME tree server.py's `configure_logging`
# (`atlas.logging_setup`, SP6 task 4) attaches its one JSON handler to, so a record logged here
# propagates through it exactly like server.py's own `_log`. This module never calls
# `configure_logging` itself (that stays server.py's job, run once at `create_app()`); under a plain
# test import with no handler configured, `logging`'s own lastResort handler still prints ERROR+ to
# stderr, and pytest's `caplog` fixture captures every record regardless of handler configuration.
# It DOES call `bind_trace_context` (below, in `_run_stream`'s own error path) so the resulting JSON
# line carries the SAME trace id the client already saw in `message_start`, as a structured field,
# not only embedded in the message text.
_log = logging.getLogger("atlas.chat_app")

# The turn's guarded final answer, chunked into word sized "token" events (trailing whitespace
# kept on each piece, so a plain join reconstructs the exact original string). `GatewayChatModel`
# has no `_stream`/`_astream` of its own (see `_run_stream`'s docstring), so this is the ONLY
# source of delta shaped granularity this reference system can honestly offer today.
_TOKEN_SPLIT = re.compile(r"\S+\s*")

# The frozen v0.1.0 SSE vocabulary has no dedicated event for "this turn needs a typed CONFIRM"
# (`contracts/sse/schema.json`'s six event types, none shaped for a pending write proposal). A
# streamed turn that ends in `__interrupt__` (see `_thread`/`ainvoke` in `/chat` below) still owes
# the client a terminal event, so it gets this fixed, safe notice as a single token, never the raw
# `pending.proposal` text (which is not guard checked output). Confirming the action stays on
# `/chat/resume`, unchanged by this task ("non streaming mode remains" applies here verbatim).
_PENDING_CONFIRMATION_NOTICE = "This needs your confirmation before it can proceed."

# The client visible text for an UNRECOGNIZED exception (`_error_payload` below): a fixed, safe
# notice, never that exception's own `str(exc)`, which might carry anything (a file path, a
# connection string, an internal detail) the client has no business seeing. The real message
# always reaches the server log first (`_log.exception`, in `_run_stream`'s own except block),
# never only this generic text.
_GENERIC_STREAM_ERROR_MESSAGE = "internal error; see server logs for detail"

_REFRESH_COOKIE = "atlas_refresh"
_ACCESS_TTL = 1800       # 30 min
_REFRESH_TTL = 7 * 24 * 3600

_bearer = HTTPBearer(auto_error=False)


# ---- request models ----
class LoginBody(BaseModel):
    customer_id: str


class ChatBody(BaseModel):
    message: str
    thread_id: str = "default"


class ResumeBody(BaseModel):
    thread_id: str
    confirmation: str


# ---- response models (so the generated TS client is fully typed, ADR-028 single source of truth) ----
class AuthOut(BaseModel):
    access_token: str
    customer_id: str
    name: str  # the customer's display name, so the UI never shows a raw internal id


class PendingOut(BaseModel):
    tool: str
    args: dict
    idempotency_key: str | None = None
    customer_id: str | None = None


class ResultOut(BaseModel):
    reference: str
    applied: bool


class ChatOut(BaseModel):
    type: str
    thread_id: str
    final_response: str | None = None
    pending: PendingOut | None = None
    result: ResultOut | None = None


# ---- SSE streaming helpers (module level: no closure over app state needed) ----


def _frame(payload: dict) -> dict:
    """One `dict` sse-starlette turns into a wire frame: `event:` set to the SAME discriminator
    the JSON `data:` payload itself carries (required on every shape in `contracts/sse/schema.json`),
    so a plain EventSource client can dispatch on either, and the conformance test only ever needs
    to `json.loads` the `data:` line to get an object that validates as is."""
    return {"event": payload["event"], "data": json.dumps(payload)}


def _token_chunks(text: str) -> list[str]:
    return _TOKEN_SPLIT.findall(text) if text else []


def _citations_from_messages(messages, seen: set[str]) -> list[dict]:
    """`tools_read`'s own `ToolMessage`s, filtered to the knowledge tools and parsed back into
    citation events. `seen` is the caller's per stream de dupe set (the same document surfacing
    from two search calls in one turn is not new information). A `ToolMessage` that fails to parse
    as a JSON passages array (the abandoned/refusal shapes, `"not executed: ..."` strings) yields
    nothing, silently -- there is no passage to cite in that case."""
    out: list[dict] = []
    for msg in messages:
        if getattr(msg, "name", None) not in KNOWLEDGE_TOOLS:
            continue
        try:
            passages = json.loads(getattr(msg, "content", "") or "")
        except json.JSONDecodeError:
            continue
        if not isinstance(passages, list):
            continue
        for passage in passages:
            if not isinstance(passage, dict):
                continue
            doc_id = passage.get("doc_id")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            event = {"event": "citation", "doc_id": doc_id}
            entity_ids = passage.get("entity_ids")  # optional per schema; absent at the MCP
            if entity_ids:                          # serialization today (knowledge_server.py never
                event["entity_ids"] = list(entity_ids)  # emits it) -- carried through when it IS present
            out.append(event)
    return out


def _error_payload(exc: Exception) -> dict:
    """`code` names a known failure by its typed shape where one exists (`CassetteMiss`, this
    reference system's own replay boundary), else the exception's class name -- never a raw
    traceback string either way. `message` is scoped by the SAME distinction: `CassetteMiss`'s own
    `str(exc)` is engineered to be user facing already (it names the cassette key and the
    remediation, see `replay/gateway.py`'s `_miss_message`), so a KNOWN type keeps it verbatim; any
    OTHER exception is unrecognized here and never forwards its raw message to the client (it might
    carry a file path, a connection string, or an internal detail with no business reaching a
    caller) -- the client gets `_GENERIC_STREAM_ERROR_MESSAGE` instead, and the real detail is
    logged server side by the caller (`_run_stream`'s `_log.exception`) before this is ever built.
    `recoverable` defaults to `False`: by the time an exception escapes `graph.astream_events`
    every retry the resilience layer owns has already been exhausted upstream (Task 3/4), so there
    is nothing left this endpoint itself could transparently retry."""
    if isinstance(exc, CassetteMiss):
        return {"event": "error", "code": "cassette_miss", "message": str(exc), "recoverable": False}
    return {"event": "error", "code": type(exc).__name__, "message": _GENERIC_STREAM_ERROR_MESSAGE, "recoverable": False}


def make_chat_app(
    clock, graph, *, cors_origins: list[str] | None = None,
    stream_events_fn: Callable[[object, dict, dict], AsyncIterator[dict]] | None = None,
    tracer=None,
) -> FastAPI:
    """`stream_events_fn` (SP4 task 6, test only): replaces `graph.astream_events(state, cfg,
    version="v2", durability="sync")` as the event source `/chat/stream` drives, when given.
    Defaults to `None` (production behaviour, always). `server.py`'s `create_app` never passes this
    -- the ONLY callers that ever will are tests proving the mid stream error + terminal guarantee
    (`test_sse_contract.py`), so the seam is a constructor parameter, never an env var or a config
    value anything deployable could flip.

    `tracer` (SP6 task 2): the SAME `Tracer` instance `server.py` passes to `build_atlas_graph`
    (never a second, independent one -- `atlas.adapters.otel_tracer.OtelTracer` keys its span tree
    by its own per instance seq counter, so two instances would never agree on what a `trace_root`
    seq means). Used ONLY to mark ttft (`atlas.stage.ttft_ms`), measured from this edge's own turn
    start to the first content bearing SSE event it yields to the client (`_run_stream`'s own
    docstring carries the exact landed definition and, fix round 2, why the clock read and the span
    object's creation happen at two different moments); the envelope `trace_id` itself comes from
    the GRAPH's own state (`trace_root`, below), never re derived here. Defaults to `NullTracer()`,
    matching the default everywhere else in this codebase."""
    app = FastAPI(title="Atlas")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    tracer = tracer or NullTracer()
    # trace_id (SP4 task 6 / SP6 task 2): the response envelope trace id IS the graph's own tracer's
    # turn root id -- the SAME value `tracer.open("turn", ...)` already returns as `trace_root`,
    # plumbed out to this edge by reading it straight off the first real node's `on_chain_end`
    # output (every `/chat/stream` call starts a FRESH turn, so the graph's OWN "agent" node always
    # sets `trace_root` in its returned state on the very first superstep -- see `_run_stream`).
    #
    # `message_start` must be the first event on the wire (the frozen SSE contract, every existing
    # test), which is BEFORE the graph has produced any output at all; `_run_stream` below buffers
    # nothing extra to get this -- it simply defers yielding `message_start` until the first REAL
    # node event arrives (queuing nothing: that first event carries no citation/degradation data of
    # its own, see `_run_stream`'s docstring), then falls through to the SAME per event processing
    # every later event already gets.
    #
    # `NullTracer` (the hermetic/default adapter) returns the SAME sentinel (`-1`) on every turn, so
    # a real, UNIQUE id is still required whenever the tracer issues no real one: `trace_ids` is that
    # documented fallback (never the primary source any more), the same deterministic `IdFactory`
    # pattern every other id in this codebase uses (`server.py`'s `IdFactory("ref")`/`IdFactory("idem")`).
    # KNOWN, ACCEPTED GAP: a turn that fails BEFORE the graph's first node produces no real turn
    # span, so its client/log id is this fallback value and joins to no exported trace; nothing at
    # this edge can name a span that never came to exist (SP6 final review, reconfirmed live).
    trace_ids = IdFactory("trace")

    def _resolve_trace_id(trace_root) -> str:
        if isinstance(trace_root, int) and trace_root >= 0:
            return str(trace_root)
        return trace_ids.next()

    def require_scope(scope: str):
        # Bearer as a security scheme (not a header parameter) so the generated client injects the
        # token via middleware, never as a per call argument, keeps identity out of the call site.
        # Validation itself goes through the one `bearer_context` helper the MCP edge also uses,
        # so the two edges cannot drift on how a malformed header is rejected.
        def dependency(request: Request, _creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> dict:
            try:
                ctx = bearer_context(request.headers.get("authorization"), clock.now())
            except TokenError:
                raise HTTPException(status_code=401, detail="missing or invalid bearer token")
            if scope not in ctx["scopes"]:
                raise HTTPException(status_code=403, detail=f"missing scope: {scope}")
            return ctx

        return dependency

    def _thread(customer_id: str, thread_id: str) -> dict:
        # namespace per customer. A client supplied thread_id can never address another's checkpoint.
        # The recursion limit (superstep ceiling tied to the call budget, not langgraph's ~4x looser
        # default of 25) comes from the shared `thread_config` helper, so this product edge and every
        # eval lane drive the graph under the identical bound (finding 2).
        return thread_config(f"{customer_id}::{thread_id}")

    # ---- auth ----
    @app.post("/auth/login", response_model=AuthOut)
    def login(body: LoginBody, response: Response) -> dict:
        """Demo sign in AS a seeded customer (no password). Real auth is out of scope."""
        if body.customer_id not in SEED:
            raise HTTPException(status_code=404, detail="unknown customer")
        now = clock.now()
        # least agency: a session starts read only. A write is acquired on the turn that needs it
        # via /auth/step-up (ADR-027 step up authorization), not handed out at login.
        access = issue_token(body.customer_id, ["read"], now, ttl_seconds=_ACCESS_TTL)
        refresh = issue_token(body.customer_id, ["read"], now, ttl_seconds=_REFRESH_TTL)
        response.set_cookie(
            _REFRESH_COOKIE, refresh, httponly=True, secure=True, samesite="strict", max_age=_REFRESH_TTL
        )
        return {"access_token": access, "customer_id": body.customer_id, "name": SEED[body.customer_id].name}

    @app.post("/auth/step-up", response_model=AuthOut)
    def step_up(ctx: dict = Depends(require_scope("read"))) -> dict:
        """Step up authorization: elevate an authenticated read session to a short lived write token
        for the turn that confirms an action. A production system gates this on re auth (password/MFA).
        The demo elevates on a valid read token. The write scope is never minted at login."""
        cid = ctx["customer_id"]
        access = issue_token(cid, ["read", "write"], clock.now(), ttl_seconds=_ACCESS_TTL)
        return {"access_token": access, "customer_id": cid, "name": SEED[cid].name}

    @app.post("/auth/refresh", response_model=AuthOut)
    def refresh(response: Response, atlas_refresh: str | None = Cookie(default=None)) -> dict:
        token = atlas_refresh
        if not token:
            raise HTTPException(status_code=401, detail="no refresh cookie")
        try:
            ctx = validate_token(token, clock.now())
        except TokenError:
            raise HTTPException(status_code=401, detail="invalid refresh token")
        access = issue_token(ctx["customer_id"], list(ctx["scopes"]), clock.now(), ttl_seconds=_ACCESS_TTL)
        return {"access_token": access, "customer_id": ctx["customer_id"], "name": SEED[ctx["customer_id"]].name}

    @app.exception_handler(GraphRecursionError)
    async def _graph_exhausted(request: Request, exc: GraphRecursionError) -> JSONResponse:
        # ONE home for the recursion backstop across BOTH graph invoking routes (/chat, /chat/resume):
        # the graph hands off before the limit in its read loop, but if the ceiling is hit anyway the
        # client gets the typed [safe handoff] ChatOut, never a raw 500. thread_id echoes the request
        # body (FastAPI caches it once the route has parsed it) so the SPA can still correlate the
        # reply; both request bodies carry `thread_id`, defaulting as ChatBody/ResumeBody would.
        try:
            thread_id = (await request.json()).get("thread_id", "default")
        except Exception:
            thread_id = "default"
        return JSONResponse({
            "type": "final",
            "thread_id": thread_id,
            "final_response": f"{HANDOFF_PREFIX} this request needs more steps than a single turn allows; let me get a person.",
        })

    # ---- chat ----
    @app.post("/chat", response_model=ChatOut)
    async def chat(body: ChatBody, ctx: dict = Depends(require_scope("read"))) -> dict:
        customer_id = ctx["customer_id"]  # from the token claim, NEVER the body
        state = {"messages": [HumanMessage(body.message)], "session": {"customer_id": customer_id}}
        # durability="sync" (SP4 task 2): the checkpoint write completes before this call returns, so
        # a client that has already seen the HTTP response is guaranteed the turn survived a restart
        # of the process backing the Postgres saver. The default ("async") only schedules the write,
        # which would race a restart immediately after response -- exactly the live persistence test.
        result = await graph.ainvoke(state, _thread(customer_id, body.thread_id), durability="sync")
        if "__interrupt__" in result:
            return {"type": "interrupt", "thread_id": body.thread_id, "pending": result.get("pending")}
        return {"type": "final", "thread_id": body.thread_id, "final_response": result["final_response"]}

    async def _run_stream(state: dict, cfg: dict, session_id: str) -> AsyncIterator[dict]:
        """The event mapping (SP4 task 6), driven off `graph.astream_events`'s own node level
        `on_chain_end` events (`ev["name"] == ev["metadata"]["langgraph_node"]` picks out a REAL
        node's own state update, never a conditional edge's routing string or the top level
        `"LangGraph"` chain -- see the task report for how this was derived empirically).

        Citations and degradation are emitted AS SOON as the node that produced them completes
        (`tools_read`): retrieval and the ladder's own rungs are not user facing generated text, so
        there is nothing to guard before surfacing them.

        Token events are the guarded FINAL text ONLY, chunked, emitted after the WHOLE run
        completes -- never a raw delta from the model's own generation. Two independent reasons,
        either one sufficient alone: (1) `GatewayChatModel` (the only model this reference system
        ever calls) has no `_stream`/`_astream` of its own, so `graph.astream_events` never emits an
        `on_chat_model_stream` event for it in the first place (`atlas_graph._generate_message`
        also calls `model._agenerate(...)` directly in replay/hermetic mode, bypassing LangChain's
        callback manager entirely -- confirmed empirically, no event fires at all on that path).
        (2) even where it WOULD be available (the live, tool bound path's `.ainvoke()`), this
        graph's `pre_render_guard`/`refusal` can still veto or REWRITE the model's raw answer
        AFTER generation (the same guard `/chat` already depends on) -- streaming un vetted text
        as it is produced would leak exactly what that guard exists to catch. So: buffer, guard,
        THEN stream the guarded text as word chunks -- the only interpretation of "generation
        deltas" this graph's own safety design can honestly support.

        A turn that ends in `__interrupt__` (a pending write) never gets a `final_response`; see
        `_PENDING_CONFIRMATION_NOTICE`'s own docstring for that gap. `GraphRecursionError` maps to
        `finish_reason="truncated"` (a controlled outcome, no `error` event: nothing partial was
        ever exposed to guard). Every other exception is logged server side in full
        (`_log.exception`, the "atlas" logger tree, the trace id it also stamps into `message_start`
        appended so a reported failure is a direct grep -- SP6 task 2, closing the SP4 error
        correlation carry) THEN maps to the in band `error` event (`_error_payload`, message scoped
        -- see its own docstring) and `message_end(finish_reason="error")` -- the terminal guarantee
        holds on every path, this is the only `except` in this function that does not `return`
        without it. One caveat: this guarantee is a property of the code, not of the transport -- if
        the client itself disconnects, ASGI raises `GeneratorExit` into this generator and none of
        the above runs; the guarantee holds while the connection lives, not past it.

        SP6 task 4: the `_log.exception` call above is now wrapped in `bind_trace_context(trace_id,
        ...)`, so the SAME trace id already reused verbatim from Task 2 (never a second one minted)
        also lands as a structured JSON `trace_id` field, not only substring embedded in the message
        text -- a reported failure is a direct grep against a JSON key, not just a string match.
        `span_id` is always absent here (I1 fix, SP6 final review): `ttft_seq` names the ttft stage
        span `_start()` always opens before this except block runs, but this function NEVER reaches
        the later `tracer.close(ttft_seq)` call on any path that lands here, and a stage span that
        never closes never exports (`SimpleSpanProcessor` only flushes an ended span,
        `OtelTracer`'s own docstring) -- logging `ttft_seq` as `span_id` used to name a span that, on
        a real tracer, genuinely never left the process (the review's own I1 finding, reproduced
        live). `atlas.turn.seq` on the turn root span (stamped by every real tracer now) is the
        supported way to find the turn from `trace_id` alone; there is no second, more specific span
        this edge can honestly point to on this path.

        `trace_id` (SP6 task 2): resolved from the FIRST real node's own output (always "agent" on
        a fresh turn, which every `/chat/stream` call is), the moment it arrives -- `message_start`
        is deferred until then, never sent with a value invented ahead of the graph's own tracer.

        `ttft_mark` / `ttft_seq` (`atlas.stage.ttft_ms`, SP6 task 2 fix rounds 1 and 2): the CLOCK
        READ happens at THIS function's own first executed line below (`tracer.mark()`), before
        `source_fn` has produced a single event -- the true turn start this edge can observe. It must
        NOT be taken only once `trace_root` is known (the first node's `on_chain_end`): for a direct
        answer turn that event fires AFTER the graph's own "agent" node has already made its LLM call
        and returned (`atlas_graph.py` opens "turn" then "agent", an `llm` kind span ended instantly
        with no monotonic wrap of its own, entirely before this edge ever sees an event) -- marking
        there would silently exclude generation and report only post generation guard/render
        overhead, not time to first token (fix round 1's regression,
        `test_ttft_span_measures_from_turn_start_not_after_the_first_graph_event`, in
        `test_trace_id_handoff.py`, pins this down with a scripted clock).

        The SPAN ITSELF, though, is not created until `_start()` runs (fix round 2): opening it here,
        immediately, with no real parent to nest under yet, is what fix round 1 actually did, and it
        was wrong at the OTel level -- an unparented span is not merely "parentless within this
        trace," it starts a brand new, independently random trace_id of its own, so `ttft` shipped as
        a disconnected trace, never findable by grouping spans under its own turn's trace_id (fix
        round 2's regression, `test_ttft_span_shares_its_turns_trace_id_and_still_measures_from_turn_start`,
        pins this down). `_start()` creates the real span once `trace_root` is known, passing
        `start_at=ttft_mark` so its reported duration and (approximate) OTel start time both still
        anchor to the ORIGINAL mark, never to whenever `_start()` itself happens to run
        (`OtelTracer.open`'s own docstring covers the backdating mechanism). It closes right before
        the FIRST token producing branch below, on every path that reaches one -- the earliest point
        this edge actually puts guarded content on the wire.

        Landed definition, also carried in `trace_translation.py`'s module docstring so a reader of
        either file gets the identical answer: wall clock from this generator's own turn start to
        the first content bearing SSE event this edge yields to the client. NOT strict OTel GenAI
        "model first token" -- this graph fully buffers, guards, and only then streams its answer
        (this docstring's own token events paragraph above), so a raw model level first token is
        never observable from this edge at all; measuring anything narrower would be dishonest about
        what the number means. A path that never produces a token (truncated/error) still opens the
        span (every path calls `_start()` at least once, including both except blocks below) but
        never closes it, so it never exports -- the SAME "a never closed stage never exports"
        contract `otel_tracer.py` documents, unchanged by fix round 2.
        """
        source_fn = stream_events_fn or (lambda g, s, c: g.astream_events(s, c, version="v2", durability="sync"))
        emitted_mode = DEGRADATION_MODE_NONE
        final_text: str | None = None
        seen_docs: set[str] = set()
        trace_id: str | None = None
        started = False
        ttft_seq: int | None = None
        # `mark()` reads the tracer's own clock right here, at true turn start, before the loop below
        # ever asks `source_fn` for an event (SP6 task 2 fix round 2). The actual ttft SPAN is not
        # created yet -- it opens later, inside `_start()`, once `trace_root` is known, so it can
        # nest under the real turn instead of becoming an unparented, independently random OTel trace
        # (the connectivity regression fix round 1 introduced by opening with `parent=None` here; see
        # this function's own docstring above). `open(..., start_at=ttft_mark)` backdates the reported
        # duration to THIS reading regardless of when the span object itself is created.
        ttft_mark = tracer.mark()

        def _start(trace_root=None):
            nonlocal trace_id, ttft_seq, started
            trace_id = _resolve_trace_id(trace_root)
            parent = trace_root if isinstance(trace_root, int) and trace_root >= 0 else None
            ttft_seq = tracer.open("ttft", "stage", parent, start_at=ttft_mark)
            started = True
            return _frame({"event": "message_start", "session_id": session_id, "trace_id": trace_id})

        try:
            async for ev in source_fn(graph, state, cfg):
                if ev.get("event") != "on_chain_end":
                    continue
                node = (ev.get("metadata") or {}).get("langgraph_node")
                if node is None or ev.get("name") != node:
                    continue
                output = (ev.get("data") or {}).get("output")
                if not isinstance(output, dict):
                    continue
                if not started:
                    yield _start(output.get("trace_root"))
                mode = output.get("degradation_mode")
                if mode and mode != DEGRADATION_MODE_NONE and mode != emitted_mode:
                    emitted_mode = mode
                    yield _frame({"event": "degradation", "mode": mode})
                if node == "tools_read":
                    for citation in _citations_from_messages(output.get("messages") or [], seen_docs):
                        yield _frame(citation)
                if "final_response" in output and output["final_response"] is not None:
                    final_text = output["final_response"]
            if not started:
                yield _start()
        except GraphRecursionError:
            if not started:
                yield _start()
            yield _frame({"event": "message_end", "finish_reason": "truncated"})
            return
        except Exception as exc:  # log server side first, then the in band error, then the terminal
            if not started:
                yield _start()
            # span_id (I1 fix, SP6 final review): always None here. ttft's span never closes on this
            # path (this function's own docstring), so it never exports on a real tracer either;
            # naming `ttft_seq` as `span_id` would claim a span that does not exist in the actual
            # export, on every real (non hermetic) run. `trace_id` alone is what this edge can
            # honestly stand behind.
            with bind_trace_context(trace_id, None):
                _log.exception("chat stream failed mid turn (session_id=%s trace_id=%s)", session_id, trace_id)
            yield _frame(_error_payload(exc))
            yield _frame({"event": "message_end", "finish_reason": "error"})
            return
        if final_text is None:
            final_text = _PENDING_CONFIRMATION_NOTICE
            tracer.close(ttft_seq)
            for chunk in _token_chunks(final_text):
                yield _frame({"event": "token", "text": chunk})
            yield _frame({"event": "message_end", "finish_reason": "complete"})
            return
        tracer.close(ttft_seq)
        for chunk in _token_chunks(final_text):
            yield _frame({"event": "token", "text": chunk})
        reason = "refusal" if final_text.startswith(HANDOFF_PREFIX) else "complete"
        yield _frame({"event": "message_end", "finish_reason": reason})

    @app.post("/chat/stream")
    async def chat_stream(body: ChatBody, ctx: dict = Depends(require_scope("read"))) -> EventSourceResponse:
        customer_id = ctx["customer_id"]  # from the token claim, NEVER the body -- same as /chat
        state = {"messages": [HumanMessage(body.message)], "session": {"customer_id": customer_id}}
        session_id = f"{customer_id}::{body.thread_id}"
        return EventSourceResponse(_run_stream(state, _thread(customer_id, body.thread_id), session_id))

    @app.post("/chat/resume", response_model=ChatOut)
    async def resume(body: ResumeBody, ctx: dict = Depends(require_scope("write"))) -> dict:
        customer_id = ctx["customer_id"]
        cfg = _thread(customer_id, body.thread_id)
        # resuming a thread with nothing to confirm (never proposed, or already executed) is a
        # client state conflict, not a server error: 409, and the write surface stays untouched.
        snapshot = await graph.aget_state(cfg)
        if not any(task.interrupts for task in snapshot.tasks):
            raise HTTPException(status_code=409, detail="no pending confirmation on this thread")
        result = await graph.ainvoke(Command(resume=body.confirmation), cfg, durability="sync")
        return {
            "type": "final",
            "thread_id": body.thread_id,
            "final_response": result["final_response"],
            "result": result.get("result"),
        }

    return app
