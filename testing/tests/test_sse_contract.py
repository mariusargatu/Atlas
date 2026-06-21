"""The SSE streaming contract at the endpoint (SP4 task 6): the hermetic conformance test that
finally drives `contracts/sse/schema.json` from its OWN producer, `POST /chat/stream`, over the
replayed gateway.

`test_contract_sse.py` (SP1) already proves the frozen golden EXAMPLES validate; this file proves
the SERVED endpoint's REAL emitted events validate too, using the same jsonschema draft the SP1
test uses (`contract_tools.loader.load_schema`). Same cassette discipline as `test_chat_app.py`:
`GatewayChatModel` in replay mode, `InMemoryRetriever` (or a typed error stub, for the ladder
case), the frozen clock. No Docker, no network.
"""
from __future__ import annotations

import io
import json
import logging

import jsonschema
import pytest
from contract_tools import loader
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphRecursionError

from determinism.canonical import serialize_tool_result
from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory, fixture_kit
from replay.gateway import GatewayChatModel

from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.adapters.resilience import ProviderError, RerankServiceError, RetrievalError
from atlas.domain.accounts import apply_write
from atlas.domain.actions import ActionsBackend
from atlas.domain.retrieval import RetrievalConfig
from atlas.orchestration.atlas_graph import HANDOFF_PREFIX, build_atlas_graph
from atlas.ports.knowledge import Chunk
from atlas.chat_app import make_chat_app

_FALSE_ANSWER = "Your plan is contract-free, no fee, cancel any time."


@pytest.fixture(scope="module")
def schema() -> dict:
    return loader.load_schema("sse")


def _app(tmp_path, **kwargs):
    kit = fixture_kit()
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    backend = ActionsBackend(IdFactory("ref"), writer=apply_write)
    graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer())
    return make_chat_app(kit.clock, graph, **kwargs)


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _login(client, customer_id):
    r = await client.post("/auth/login", json={"customer_id": customer_id})
    assert r.status_code == 200
    return r.json()["access_token"]


def _parse_block(lines: list[str]) -> dict:
    data = "".join(ln[len("data: "):] for ln in lines if ln.startswith("data: "))
    return json.loads(data)


async def _stream(client, token, message, thread_id="s1"):
    """POST /chat/stream and return every SSE event, parsed back into a dict."""
    events: list[dict] = []
    buffer: list[str] = []
    async with client.stream(
        "POST", "/chat/stream",
        json={"message": message, "thread_id": thread_id},
        headers={"authorization": f"Bearer {token}"},
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line == "":
                if buffer:
                    events.append(_parse_block(buffer))
                    buffer = []
                continue
            buffer.append(line)
        if buffer:
            events.append(_parse_block(buffer))
    return events


def _validate_all(events: list[dict], schema: dict) -> None:
    for event in events:
        jsonschema.validate(event, schema)


# ---- the happy path: every event validates, message_start first, message_end(complete) last ----


@pytest.mark.asyncio
async def test_stream_happy_path_validates_and_ends_complete(tmp_path, seed_cassette, schema):
    seed_cassette(tmp_path, [HumanMessage("What is your name?")], {"content": "Hi, I'm Atlas.", "tool_calls": []})
    async with _client(_app(tmp_path)) as client:
        token = await _login(client, "cust_current")
        events = await _stream(client, token, "What is your name?")

    _validate_all(events, schema)
    assert events[0]["event"] == "message_start"
    assert events[0]["session_id"] == "cust_current::s1"
    assert isinstance(events[0]["trace_id"], str) and events[0]["trace_id"]
    assert events[-1] == {"event": "message_end", "finish_reason": "complete"}
    tokens = [e for e in events if e["event"] == "token"]
    assert tokens, "the guarded final answer must reach the client as token events"
    assert "".join(t["text"] for t in tokens) == "Hi, I'm Atlas."


@pytest.mark.asyncio
async def test_stream_trace_id_is_a_fresh_value_per_call_not_reused_from_session_id(tmp_path, seed_cassette, schema):
    seed_cassette(tmp_path, [HumanMessage("hi")], {"content": "hello", "tool_calls": []})
    app = _app(tmp_path)
    async with _client(app) as client:
        token = await _login(client, "cust_current")
        first = await _stream(client, token, "hi", thread_id="a")
        second = await _stream(client, token, "hi", thread_id="a")
    assert first[0]["session_id"] == second[0]["session_id"] == "cust_current::a"
    assert first[0]["trace_id"] != second[0]["trace_id"]  # per turn, never per thread


# ---- citations: doc_id from a real search_knowledge round trip, entity_ids absent (never serialized) ----


@pytest.mark.asyncio
async def test_stream_search_knowledge_emits_a_citation_with_doc_id(tmp_path, seed_cassette, schema):
    query = "plan contract term cancel fee"
    user = HumanMessage("Is my plan contract-free?")
    toolcall = [{"name": "search_knowledge", "args": {"query": query}, "id": "k1"}]
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": toolcall})

    chunks = InMemoryRetriever().search_chunks(query, config=RetrievalConfig())
    passages = serialize_tool_result(
        [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": c.score, "text": c.text} for c in chunks]
    )
    ai = AIMessage(content="", tool_calls=toolcall)
    tool_msg = ToolMessage(content=passages, tool_call_id="k1", name="search_knowledge")
    seed_cassette(tmp_path, [user, ai, tool_msg], {"content": _FALSE_ANSWER, "tool_calls": []})

    async with _client(_app(tmp_path)) as client:
        token = await _login(client, "cust_current")  # current, not legacy: the render guard lets it through
        events = await _stream(client, token, "Is my plan contract-free?")

    _validate_all(events, schema)
    citations = [e for e in events if e["event"] == "citation"]
    assert citations, "a search_knowledge round trip must surface at least one citation event"
    assert citations[0]["doc_id"] == chunks[0].doc_id
    assert "entity_ids" not in citations[0]  # knowledge_server never serializes entity_ids today


@pytest.mark.asyncio
async def test_stream_citation_events_come_before_the_final_answer_tokens(tmp_path, seed_cassette, schema):
    """Retrieval always completes before the guarded final answer is known, so a citation from it
    is always available to the client strictly before the tokens that answer using it."""
    query = "plan contract term cancel fee"
    user = HumanMessage("Is my plan contract-free?")
    toolcall = [{"name": "search_knowledge", "args": {"query": query}, "id": "k1"}]
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": toolcall})
    chunks = InMemoryRetriever().search_chunks(query, config=RetrievalConfig())
    passages = serialize_tool_result(
        [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": c.score, "text": c.text} for c in chunks]
    )
    ai = AIMessage(content="", tool_calls=toolcall)
    tool_msg = ToolMessage(content=passages, tool_call_id="k1", name="search_knowledge")
    seed_cassette(tmp_path, [user, ai, tool_msg], {"content": _FALSE_ANSWER, "tool_calls": []})

    async with _client(_app(tmp_path)) as client:
        token = await _login(client, "cust_current")
        events = await _stream(client, token, "Is my plan contract-free?")

    kinds = [e["event"] for e in events]
    assert kinds.index("citation") < kinds.index("token")


# ---- refusal: the render guard catches a false answer for the legacy customer ----


@pytest.mark.asyncio
async def test_stream_render_guard_refusal_sets_finish_reason_refusal(tmp_path, seed_cassette, schema):
    seed_cassette(tmp_path, [HumanMessage("Is my plan contract-free?")], {"content": _FALSE_ANSWER, "tool_calls": []})
    async with _client(_app(tmp_path)) as client:
        token = await _login(client, "cust_legacy_term")
        events = await _stream(client, token, "Is my plan contract-free?")

    _validate_all(events, schema)
    assert events[-1] == {"event": "message_end", "finish_reason": "refusal"}
    tokens = "".join(e["text"] for e in events if e["event"] == "token")
    assert tokens.startswith(HANDOFF_PREFIX)


# ---- degradation: a real ladder rung fires end to end, over the real endpoint ----


class _RerankDownRetriever:
    """Rerank is down; the same query succeeds once knowledge_server's own ladder walk retries
    with rerank disabled (mirrors test_ladder.py's stub of the identical shape)."""

    def search_chunks(self, query, k, config):
        if config.rerank_enabled:
            raise RerankServiceError("tei-rerank down", provider_key="tei-rerank")
        return [Chunk(doc_id="doc-1", chunk_id="chunk-1", score=0.5, text="rerank-off answer")][:k]


@pytest.mark.asyncio
async def test_stream_emits_a_degradation_event_when_a_ladder_rung_fires(tmp_path, seed_cassette, schema):
    query = "contract question"
    user = HumanMessage("Is my plan contract-free?")
    toolcall = [{"name": "search_knowledge", "args": {"query": query}, "id": "k1"}]
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": toolcall})
    degraded_passages = serialize_tool_result(
        [{"doc_id": "doc-1", "chunk_id": "chunk-1", "score": 0.5, "text": "rerank-off answer"}]
    )
    ai = AIMessage(content="", tool_calls=toolcall)
    tool_msg = ToolMessage(content=degraded_passages, tool_call_id="k1", name="search_knowledge")
    seed_cassette(tmp_path, [user, ai, tool_msg], {"content": "Here is what I found.", "tool_calls": []})

    kit = fixture_kit()
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    backend = ActionsBackend(IdFactory("ref"), writer=apply_write)
    graph = build_atlas_graph(
        gw, IdFactory("idem"), backend, new_checkpointer(), retriever=_RerankDownRetriever()
    )
    app = make_chat_app(kit.clock, graph)

    async with _client(app) as client:
        token = await _login(client, "cust_current")
        events = await _stream(client, token, "Is my plan contract-free?")

    _validate_all(events, schema)
    degradations = [e for e in events if e["event"] == "degradation"]
    assert degradations == [{"event": "degradation", "mode": "drop_rerank"}]
    assert events[-1] == {"event": "message_end", "finish_reason": "complete"}


# ---- SP4 final fix wave regression (F1): a healthy fallback turn right after a worse degraded turn
#      on the SAME thread must not inherit that earlier turn's persisted mode ----


class _AlwaysDownRetriever:
    """Every ladder rung exhausted: forces turn 1 straight to refusal (never a config change saves
    it), so turn 1's checkpointed `degradation_mode` lands on "refusal" -- the worse prior state F1's
    regression needs on the thread before turn 2 ever runs."""

    def search_chunks(self, query, k, config):
        raise RetrievalError("postgres down", provider_key="postgres")


class _TurnCounter:
    """A plain mutable object, not a pydantic field mutation (`BaseChatModel` subclasses in this
    suite hold state this way, see `test_fault_lane.py`'s own `_GenerationFault`): counts real
    generation calls so the model below can tell turn 1 from turn 2 without depending on the
    checkpointed message history's shape."""

    def __init__(self) -> None:
        self.n = 0


class _SearchRefusalThenRetryableFallbackModel(BaseChatModel):
    """Turn 1 always emits one `search_knowledge` tool call (paired with `_AlwaysDownRetriever`
    above); turn 2, a FRESH turn on the SAME thread, raises a RETRYABLE `ProviderError` instead --
    reproducing the reviewer's exact scenario: a healthy fallback turn immediately after a worse
    degraded one. `agent()` calls the model exactly once per turn here (turn 1's mid batch refusal
    short circuits before a second model call; turn 2 never reaches a tool call at all), so
    `counter.n` is a reliable turn discriminator."""

    counter: _TurnCounter

    @property
    def _llm_type(self) -> str:
        return "search-refusal-then-retryable-fallback"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("the graph is async only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.counter.n += 1
        if self.counter.n == 1:
            msg = AIMessage(content="", tool_calls=[{"name": "search_knowledge", "args": {"query": "q"}, "id": "k1"}])
            return ChatResult(generations=[ChatGeneration(message=msg)])
        raise ProviderError("primary down on turn 2", retryable=True, provider_key="primary-model")


class _PlainAnswerModel(BaseChatModel):
    """The fallback model: always answers with fixed content, never calls a tool."""

    answer: str = "plain answer"

    @property
    def _llm_type(self) -> str:
        return "plain-answer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("the graph is async only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.answer))])


def _f1_graph():
    counter = _TurnCounter()
    model = _SearchRefusalThenRetryableFallbackModel(counter=counter)
    fallback = _PlainAnswerModel(answer="fallback answered turn 2")
    graph = build_atlas_graph(
        model, IdFactory("idem"), ActionsBackend(IdFactory("ref")), new_checkpointer(),
        retriever=_AlwaysDownRetriever(), fallback_model=fallback,
    )
    return graph


@pytest.mark.asyncio
async def test_provider_fallback_after_a_worse_refusal_turn_is_not_stamped_as_refusal():
    """SP4 final fix wave regression (F1): `atlas_graph.py`'s fallback branch used to escalate from
    the PREVIOUS turn's PERSISTED `degradation_mode` (read off `state`, before the per turn reset in
    `extra` had merged), so a healthy fallback turn immediately after a worse degraded turn on the
    SAME thread was wrongly stamped with the earlier turn's rung. Turn 1 exhausts retrieval
    (refusal); turn 2, same thread, hits a retryable generation blip and the fallback answers -- the
    fix bases the escalation on `extra`'s own per turn reset first, the same resolved before merge
    pattern `turn_intent` already uses one line above it. Real checkpointer (`new_checkpointer()`),
    same thread id both calls -- state carries across the two `ainvoke`s the way it would in
    production, never a hand built state dict standing in for turn 2's starting point."""
    graph = _f1_graph()
    cfg = {"configurable": {"thread_id": "f1-regression"}}
    session = {"customer_id": "cust_current"}

    turn1 = await graph.ainvoke({"messages": [HumanMessage("Is my plan contract-free?")], "session": session}, cfg)
    assert turn1["degradation_mode"] == "refusal"

    turn2 = await graph.ainvoke({"messages": [HumanMessage("What are your hours?")], "session": session}, cfg)
    assert turn2["degradation_mode"] == "provider_fallback"
    assert turn2["final_response"] == "fallback answered turn 2"


@pytest.mark.asyncio
async def test_stream_turn_2_provider_fallback_is_not_mislabeled_refusal_after_a_worse_turn_1(schema):
    """The wire visible counterpart of the regression above: `chat_app.py`'s `/chat/stream` keys its
    `degradation` event straight off node output (`_run_stream`'s own mapping), so the stale
    escalation base (F1) was not only a state bug, it shipped a WRONG event to a real client --
    `degradation: refusal` on a turn that answered fine. Same thread id across two separate
    `/chat/stream` POSTs; the checkpointer, not a hand built state dict, carries turn 1's persisted
    `degradation_mode` into turn 2's escalation."""
    app = make_chat_app(fixture_kit().clock, _f1_graph())

    async with _client(app) as client:
        token = await _login(client, "cust_current")
        turn1_events = await _stream(client, token, "Is my plan contract-free?", thread_id="f1-wire")
        turn2_events = await _stream(client, token, "What are your hours?", thread_id="f1-wire")

    _validate_all(turn1_events, schema)
    _validate_all(turn2_events, schema)

    turn1_degradations = [e for e in turn1_events if e["event"] == "degradation"]
    assert turn1_degradations == [{"event": "degradation", "mode": "refusal"}]

    turn2_degradations = [e for e in turn2_events if e["event"] == "degradation"]
    assert turn2_degradations == [{"event": "degradation", "mode": "provider_fallback"}]
    assert {"event": "degradation", "mode": "refusal"} not in turn2_degradations
    assert turn2_events[-1] == {"event": "message_end", "finish_reason": "complete"}


# ---- truncated: the graph exhausts its recursion limit ----


class _ExplodingStreamGraph:
    """Stands in for a graph whose event stream blows the recursion limit before ever yielding a
    single event -- the streaming counterpart of test_chat_app.py's `_ExplodingGraph`."""

    async def astream_events(self, *args, **kwargs):
        if False:  # pragma: no cover - makes this a real async generator function
            yield {}
        raise GraphRecursionError("recursion limit reached")


@pytest.mark.asyncio
async def test_stream_graph_recursion_error_ends_with_finish_reason_truncated(schema):
    app = make_chat_app(fixture_kit().clock, _ExplodingStreamGraph())
    async with _client(app) as client:
        token = await _login(client, "cust_current")
        events = await _stream(client, token, "hi")

    _validate_all(events, schema)
    assert events[0]["event"] == "message_start"
    assert events[-1] == {"event": "message_end", "finish_reason": "truncated"}
    assert not any(e["event"] == "error" for e in events)  # a controlled outcome, not a failure


# ---- error: a real cassette miss, mid stream, always ends in error then message_end(error) ----


@pytest.mark.asyncio
async def test_stream_cassette_miss_emits_error_then_terminal_message_end(tmp_path, schema, caplog):
    # a KNOWN exception type: its own message text (CassetteMiss precedent) reaches the client
    # verbatim, AND is also logged server side -- (a) applies to every exception, not only the
    # unrecognized ones (b) scopes the client message for.
    with caplog.at_level(logging.ERROR, logger="atlas.chat_app"):
        async with _client(_app(tmp_path)) as client:  # an existing but empty cassette dir: misses
            token = await _login(client, "cust_current")
            events = await _stream(client, token, "hi")

    _validate_all(events, schema)
    assert events[0]["event"] == "message_start"
    assert events[-2]["event"] == "error"
    assert events[-2]["code"] == "cassette_miss"
    assert "cassette miss" in events[-2]["message"]  # the specific, user facing text is kept
    assert events[-2]["recoverable"] is False
    assert events[-1] == {"event": "message_end", "finish_reason": "error"}
    assert any(r.exc_info and r.exc_info[0].__name__ == "CassetteMiss" for r in caplog.records)


# ---- the test only mid stream injection hook: proves the error + terminal guarantee ----


@pytest.mark.asyncio
async def test_injected_mid_stream_failure_after_real_content_still_terminates_cleanly(
    tmp_path, schema, caplog
):
    """A citation event is delivered for real (a completed `tools_read` step), THEN the injected
    source raises an UNRECOGNIZED exception type: the client must still see exactly one error event
    and message_end(error) as the absolute last event, never a hung or truncated connection. This
    is `stream_events_fn` (test only, never wired by server.py): a caller substitutes the async
    iterator `make_chat_app` drives instead of `graph.astream_events(...)`, so a failure can be
    injected at any point in the sequence without needing a real graph failure to reach it.

    The exception's own message ("provider connection dropped mid turn") must NEVER reach the
    client (an unrecognized type's detail could be anything): the client gets the fixed generic
    notice, and the real detail is proven to have reached the server log instead (`caplog`)."""

    async def _flaky_source(graph, state, config):
        yield {
            "event": "on_chain_end",
            "name": "tools_read",
            "metadata": {"langgraph_node": "tools_read"},
            "data": {"output": {
                "messages": [ToolMessage(
                    content=serialize_tool_result([{"doc_id": "doc-9", "chunk_id": "c-9", "score": 1.0, "text": "x"}]),
                    tool_call_id="k1", name="search_knowledge",
                )],
                "degradation_mode": "none",
            }},
        }
        raise RuntimeError("provider connection dropped mid turn")

    app = make_chat_app(fixture_kit().clock, object(), stream_events_fn=_flaky_source)
    with caplog.at_level(logging.ERROR, logger="atlas.chat_app"):
        async with _client(app) as client:
            token = await _login(client, "cust_current")
            events = await _stream(client, token, "hi")

    _validate_all(events, schema)
    assert events[0]["event"] == "message_start"
    assert events[1] == {"event": "citation", "doc_id": "doc-9"}
    assert events[-2]["event"] == "error"
    assert events[-2]["code"] == "RuntimeError"
    assert events[-2]["message"] == "internal error; see server logs for detail"  # scoped, not raw
    assert "provider connection dropped mid turn" not in events[-2]["message"]
    assert events[-2]["recoverable"] is False
    assert events[-1] == {"event": "message_end", "finish_reason": "error"}
    # the real detail reached the server log, full exception included, never only the client
    matches = [
        r for r in caplog.records
        if r.exc_info and str(r.exc_info[1]) == "provider connection dropped mid turn"
    ]
    assert matches, f"expected a logged RuntimeError with the real detail; got {caplog.records}"


@pytest.mark.asyncio
async def test_stream_error_log_line_is_json_and_carries_the_message_starts_trace_id(schema):
    """SP6 task 4: the stream error path's server side `_log.exception` line (Task 2's own
    correlation carry) now also lands as a structured JSON `trace_id` field, not merely embedded in
    the message text -- proven end to end here, through the REAL `/chat/stream` call site
    (`chat_app.py`'s `_run_stream`), not only at the `atlas.logging_setup` mechanism level
    (`test_logging_setup.py`'s own unit tests). A mechanism unit tested in isolation but never
    actually wired into the real call site is exactly the class of gap CLAUDE.md asks a reviewer to
    watch for.

    I1 fix (SP6 final review): `span_id` is deliberately ABSENT here now, on every tracer, not only
    the hermetic `NullTracer` default. `InMemoryTracer` (used below, not `NullTracer`) records a
    span the instant `open()` returns regardless of whether anything ever closes it, which used to
    make `span_id` look "genuinely present" here -- but the ttft stage span this path used to name
    never actually closes on THIS error path (`chat_app._run_stream`'s own docstring), so with the
    REAL adapter (`OtelTracer`, `test_trace_id_handoff.py`'s own
    `test_mid_stream_error_never_logs_a_span_id_for_the_never_exported_ttft_stage`) it never exports
    either. Naming it as `span_id` here was a claim `InMemoryTracer`'s own looser bookkeeping could
    never have caught; the fix is scoped to `chat_app.py` itself (backend agnostic), so this
    hermetic, `InMemoryTracer` backed test must see the SAME absence."""
    from tracing import InMemoryTracer

    from atlas.logging_setup import JsonFormatter

    async def _flaky_source(graph, state, config):
        yield {
            "event": "on_chain_end", "name": "tools_read",
            "metadata": {"langgraph_node": "tools_read"},
            "data": {"output": {"messages": [], "degradation_mode": "none"}},
        }
        raise RuntimeError("provider connection dropped mid turn")

    app = make_chat_app(
        fixture_kit().clock, object(), stream_events_fn=_flaky_source, tracer=InMemoryTracer(),
    )
    stream_buf = io.StringIO()
    handler = logging.StreamHandler(stream_buf)
    handler.setFormatter(JsonFormatter())
    chat_app_log = logging.getLogger("atlas.chat_app")
    chat_app_log.addHandler(handler)
    try:
        async with _client(app) as client:
            token = await _login(client, "cust_current")
            events = await _stream(client, token, "hi")
    finally:
        chat_app_log.removeHandler(handler)

    _validate_all(events, schema)
    message_start = events[0]
    assert message_start["event"] == "message_start"

    lines = [json.loads(ln) for ln in stream_buf.getvalue().splitlines() if ln.strip()]
    error_lines = [ln for ln in lines if ln["level"] == "ERROR"]
    assert error_lines, f"expected a JSON ERROR line from atlas.chat_app; got {lines}"
    assert error_lines[-1]["trace_id"] == message_start["trace_id"]
    assert "span_id" not in error_lines[-1]


@pytest.mark.asyncio
async def test_injected_synthetic_citation_carries_entity_ids_when_present(schema):
    """knowledge_server.py never serializes entity_ids today (documented gap); this proves the
    citation code path itself honors the schema's optional field when a passage DOES carry one,
    via a synthetic source rather than waiting on a future MCP change."""

    async def _source(graph, state, config):
        yield {
            "event": "on_chain_end",
            "name": "tools_read",
            "metadata": {"langgraph_node": "tools_read"},
            "data": {"output": {
                "messages": [ToolMessage(
                    content=json.dumps([{"doc_id": "doc-5", "entity_ids": ["plan-fiber-500"], "text": "x"}]),
                    tool_call_id="k1", name="search_knowledge",
                )],
                "degradation_mode": "none",
            }},
        }
        yield {
            "event": "on_chain_end",
            "name": "pre_render_guard",
            "metadata": {"langgraph_node": "pre_render_guard"},
            "data": {"output": {"final_response": "Fiber 500 is $39.99/mo."}},
        }

    app = make_chat_app(fixture_kit().clock, object(), stream_events_fn=_source)
    async with _client(app) as client:
        token = await _login(client, "cust_current")
        events = await _stream(client, token, "hi")

    _validate_all(events, schema)
    citation = next(e for e in events if e["event"] == "citation")
    assert citation == {"event": "citation", "doc_id": "doc-5", "entity_ids": ["plan-fiber-500"]}
    assert events[-1] == {"event": "message_end", "finish_reason": "complete"}


# ---- a pending write confirmation has no dedicated event in the frozen v0.1.0 vocabulary: a fixed,
#      safe notice plus finish_reason complete, documented in chat_app.py as a named contract gap ----


@pytest.mark.asyncio
async def test_stream_pending_write_confirmation_ends_complete_with_a_fixed_notice(tmp_path, seed_cassette, schema):
    seed_cassette(
        tmp_path, [HumanMessage("Switch me to the fast plan")],
        {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]},
    )
    async with _client(_app(tmp_path)) as client:
        token = await _login(client, "cust_current")
        events = await _stream(client, token, "Switch me to the fast plan")

    _validate_all(events, schema)
    assert events[-1] == {"event": "message_end", "finish_reason": "complete"}
    tokens = [e for e in events if e["event"] == "token"]
    assert tokens  # a fixed, safe notice, never the raw unconfirmed proposal


# ---- non streaming mode is untouched: /chat still answers exactly as before ----


@pytest.mark.asyncio
async def test_non_streaming_chat_endpoint_is_unaffected(tmp_path, seed_cassette):
    seed_cassette(tmp_path, [HumanMessage("hi")], {"content": "hello", "tool_calls": []})
    async with _client(_app(tmp_path)) as client:
        token = await _login(client, "cust_current")
        r = await client.post(
            "/chat", json={"message": "hi", "thread_id": "s1"}, headers={"authorization": f"Bearer {token}"}
        )
        assert r.status_code == 200
        assert r.json()["final_response"] == "hello"


@pytest.mark.asyncio
async def test_stream_without_a_token_is_401(tmp_path):
    async with _client(_app(tmp_path)) as client:
        r = await client.post("/chat/stream", json={"message": "hi"}, headers={"authorization": "Bearer garbage"})
        assert r.status_code == 401
