"""SP6 task 2: the response envelope `trace_id` (`/chat/stream`'s `message_start` event) becomes
the REAL tracer's own turn root id whenever one exists, with a deterministic fallback under the
hermetic default (`NullTracer`, whose `open()` always returns the SAME sentinel, `-1`); the SAME id
is stamped into the server side log line on a mid stream failure (`chat_app._run_stream`'s
`_log.exception`), closing the SP4 error correlation carry -- a reported failure is a direct grep,
not a session_id and timestamp guess. Hermetic: `OtelTracer` here is constructed exactly like
`test_otel_tracer.py` does (an injected in memory exporter, no network).
"""
from __future__ import annotations

import io
import json
import logging

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from atlas.adapters.otel_tracer import OtelTracer
from atlas.chat_app import make_chat_app
from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph
from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory, fixture_kit
from replay.gateway import GatewayChatModel


class _FakeClock:
    """A plain incrementing fake clock: never the real `time.monotonic`, so this whole file stays
    inside the Global Constraints' "no wall clock in hermetically exercised runtime paths" -- the
    real ttft/stage duration MATH is `test_otel_tracer.py`'s own concern; this file only proves
    chat_app's wiring calls `open`/`close` at all, at the right moments."""

    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        self._value += 0.01
        return self._value


def _otel_tracer():
    exporter = InMemorySpanExporter()
    tracer = OtelTracer(
        endpoint="http://example.invalid:4318", config_hash="h", exporter=exporter, clock=_FakeClock()
    )
    return tracer, exporter


def _app_with_tracer(tmp_path, tracer):
    kit = fixture_kit()
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    backend = ActionsBackend(IdFactory("ref"))
    graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer(), tracer=tracer)
    return make_chat_app(kit.clock, graph, tracer=tracer)


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _login(client, customer_id="cust_current"):
    r = await client.post("/auth/login", json={"customer_id": customer_id})
    assert r.status_code == 200
    return r.json()["access_token"]


def _parse_block(lines: list[str]) -> dict:
    data = "".join(ln[len("data: "):] for ln in lines if ln.startswith("data: "))
    return json.loads(data)


async def _stream(client, token, message, thread_id="s1"):
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


@pytest.mark.asyncio
async def test_trace_id_is_the_real_tracers_turn_root_id_not_a_synthetic_counter(tmp_path, seed_cassette):
    seed_cassette(tmp_path, [HumanMessage("hi")], {"content": "hello", "tool_calls": []})
    tracer, exporter = _otel_tracer()
    app = _app_with_tracer(tmp_path, tracer)
    async with _client(app) as client:
        token = await _login(client)
        events = await _stream(client, token, "hi")

    turn_span = next(s for s in exporter.get_finished_spans() if s.name == "turn")
    # the graph's OWN turn span id -- an OTel context span_id, not the atlas Tracer seq -- is not
    # itself what chat_app reads (chat_app reads the astream_events output's own trace_root int, the
    # SAME seq value tracer.open("turn", ...) returned); this test instead proves message_start's
    # trace_id is a real, stable integer shaped id (never the fallback IdFactory's "trace-000001"
    # shape) whenever a real tracer is wired in.
    assert events[0]["trace_id"].isdigit()
    assert turn_span is not None


@pytest.mark.asyncio
async def test_trace_id_falls_back_to_a_deterministic_id_under_the_hermetic_nulltracer_default(tmp_path, seed_cassette):
    """No tracer passed at all (`build_atlas_graph`'s own `tracer or NullTracer()` default, and
    `make_chat_app`'s matching default): every turn's `trace_root` is the SAME sentinel (`-1`), so
    the fallback `IdFactory` -- never deleted, only demoted to a documented fallback -- must still
    hand out a fresh, non "-1" value per call, or the frozen SSE uniqueness test would break."""
    seed_cassette(tmp_path, [HumanMessage("hi")], {"content": "hello", "tool_calls": []})
    kit = fixture_kit()
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    backend = ActionsBackend(IdFactory("ref"))
    graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer())  # no tracer given
    app = make_chat_app(kit.clock, graph)  # no tracer given either

    async with _client(app) as client:
        token = await _login(client)
        events = await _stream(client, token, "hi")

    assert events[0]["trace_id"] != "-1"
    assert not events[0]["trace_id"].isdigit()  # the fallback IdFactory's own "trace-NNNNNN" shape


@pytest.mark.asyncio
async def test_mid_stream_error_logs_the_same_trace_id_the_client_already_received(tmp_path, caplog):
    """The SP4 Task 6 carry: a generic client facing error message with no correlation id back to
    the server log. `message_start` (the stream's first event) already carries `trace_id`; this
    proves the SAME value reaches the server side `_log.exception` line on a later failure in the
    SAME stream -- closing the gap without touching the frozen `error` event's own shape (which has
    no `trace_id` field, `additionalProperties: false`)."""

    async def _flaky_source(graph, state, config):
        raise RuntimeError("provider connection dropped mid turn")
        yield {}  # pragma: no cover - makes this a real async generator function

    tracer, _ = _otel_tracer()
    kit = fixture_kit()
    backend = ActionsBackend(IdFactory("ref"))
    graph = build_atlas_graph(
        GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay"),
        IdFactory("idem"), backend, new_checkpointer(), tracer=tracer,
    )
    app = make_chat_app(kit.clock, graph, tracer=tracer, stream_events_fn=_flaky_source)

    with caplog.at_level(logging.ERROR, logger="atlas.chat_app"):
        async with _client(app) as client:
            token = await _login(client)
            events = await _stream(client, token, "hi")

    assert events[0]["event"] == "message_start"
    client_trace_id = events[0]["trace_id"]
    assert events[-2]["event"] == "error"
    assert "trace_id" not in events[-2]  # the frozen error event shape stays untouched

    matches = [r for r in caplog.records if r.exc_info and str(r.exc_info[1]) == "provider connection dropped mid turn"]
    assert matches, f"expected a logged RuntimeError; got {caplog.records}"
    assert any(f"trace_id={client_trace_id}" in r.getMessage() for r in matches)


@pytest.mark.asyncio
async def test_mid_stream_error_never_logs_a_span_id_for_the_never_exported_ttft_stage(tmp_path):
    """I1 fix (SP6 final review): on this exact error path, `ttft` never closes (this class's own
    docstring, and `chat_app._run_stream`'s own docstring: every except block calls `_start()` at
    least once but never reaches the `tracer.close(ttft_seq)` call further down), so with a REAL
    `OtelTracer` it never exports either -- `SimpleSpanProcessor` only flushes an ended span. The
    log used to name `ttft_seq` as `span_id` here anyway, a claim nothing in the actual export could
    ever back up (the review's I1 finding, reproduced live: a real ttft seq logged as `span_id` while
    no such span ever left the process). The fix: this path never claims a `span_id` at all now,
    since it has no span left here it can honestly point to."""
    from atlas.logging_setup import JsonFormatter

    async def _flaky_source(graph, state, config):
        raise RuntimeError("provider connection dropped mid turn")
        yield {}  # pragma: no cover - makes this a real async generator function

    tracer, exporter = _otel_tracer()
    kit = fixture_kit()
    backend = ActionsBackend(IdFactory("ref"))
    graph = build_atlas_graph(
        GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay"),
        IdFactory("idem"), backend, new_checkpointer(), tracer=tracer,
    )
    app = make_chat_app(kit.clock, graph, tracer=tracer, stream_events_fn=_flaky_source)

    stream_buf = io.StringIO()
    handler = logging.StreamHandler(stream_buf)
    handler.setFormatter(JsonFormatter())
    chat_app_log = logging.getLogger("atlas.chat_app")
    chat_app_log.addHandler(handler)
    try:
        async with _client(app) as client:
            token = await _login(client)
            await _stream(client, token, "hi")
    finally:
        chat_app_log.removeHandler(handler)

    lines = [json.loads(ln) for ln in stream_buf.getvalue().splitlines() if ln.strip()]
    error_lines = [ln for ln in lines if ln["level"] == "ERROR"]
    assert error_lines, f"expected a JSON ERROR line from atlas.chat_app; got {lines}"
    assert "span_id" not in error_lines[-1]

    # not merely a formatting choice: no ttft span actually left the process either, so there was
    # never anything honest left here to name.
    assert not [s for s in exporter.get_finished_spans() if s.name == "ttft"]


@pytest.mark.asyncio
async def test_atlas_turn_seq_joins_the_envelope_trace_id_to_its_exported_span_both_directions(tmp_path, seed_cassette):
    """I1 fix (SP6 final review): the actual point of `atlas.turn.seq` -- proven end to end here
    through the REAL `/chat/stream` call site with a REAL `OtelTracer` and an in memory OTLP
    exporter (no network), not only at the adapter's own unit test level
    (`test_otel_tracer.py::test_atlas_turn_seq_is_stamped_on_the_turn_span_...`). Both directions:
    (1) given the envelope's `trace_id` (what a client or a log line actually holds), find the real
    exported span it names; (2) given any span in that same real OTel trace, find the envelope's own
    `trace_id` back. Before this fix, `atlas.turn.seq` did not exist: an operator holding a client's
    trace_id had no attribute to search Phoenix or the raw archive by (I1's own finding)."""
    seed_cassette(tmp_path, [HumanMessage("hi")], {"content": "hello", "tool_calls": []})
    tracer, exporter = _otel_tracer()
    app = _app_with_tracer(tmp_path, tracer)
    async with _client(app) as client:
        token = await _login(client)
        events = await _stream(client, token, "hi")

    envelope_trace_id = events[0]["trace_id"]
    finished = exporter.get_finished_spans()

    # (1) envelope -> span: search every exported span's atlas.turn.seq for the envelope's id.
    found = [s for s in finished if s.attributes.get("atlas.turn.seq") == envelope_trace_id]
    assert len(found) == 1, f"expected exactly one span matching atlas.turn.seq={envelope_trace_id!r}"
    turn_span = found[0]
    assert turn_span.name == "turn"  # the ONLY seq ever surfaced to a client is the turn root's own
    assert turn_span.parent is None  # confirms it really is the trace's own root

    # (2) span -> envelope: from ANY span sharing that real OTel trace, walk back to the root (the
    # one with no parent) and read its atlas.turn.seq -- recovers the envelope's trace_id with no
    # foreknowledge of which span was the root going in.
    same_trace = [s for s in finished if s.context.trace_id == turn_span.context.trace_id]
    assert len(same_trace) > 1, "expected more than just the turn span on this real trace"
    (root_of_trace,) = [s for s in same_trace if s.parent is None]
    assert root_of_trace.attributes["atlas.turn.seq"] == envelope_trace_id


@pytest.mark.asyncio
async def test_ttft_stage_span_is_marked_on_a_real_tracer(tmp_path, seed_cassette):
    seed_cassette(tmp_path, [HumanMessage("hi")], {"content": "hello there", "tool_calls": []})
    tracer, exporter = _otel_tracer()
    app = _app_with_tracer(tmp_path, tracer)
    async with _client(app) as client:
        token = await _login(client)
        await _stream(client, token, "hi")

    ttft_spans = [s for s in exporter.get_finished_spans() if s.name == "ttft"]
    assert len(ttft_spans) == 1
    assert ttft_spans[0].attributes["atlas.stage.ttft_ms"] >= 0


@pytest.mark.asyncio
async def test_ttft_is_never_marked_on_a_recursion_truncated_turn():
    """A truncated turn never produces a token; ttft's own span must never close (and therefore
    never export) either -- the SAME "never closed stage never exports" contract embed/retrieve/rerank
    already follow, so nothing misleading is ever reported for a turn nothing was rendered on."""
    from langgraph.errors import GraphRecursionError

    class _ExplodingStreamGraph:
        async def astream_events(self, *args, **kwargs):
            if False:  # pragma: no cover
                yield {}
            raise GraphRecursionError("recursion limit reached")

    tracer, exporter = _otel_tracer()
    kit = fixture_kit()
    app = make_chat_app(kit.clock, _ExplodingStreamGraph(), tracer=tracer)
    async with _client(app) as client:
        token = await _login(client)
        events = await _stream(client, token, "hi")

    assert events[-1] == {"event": "message_end", "finish_reason": "truncated"}
    assert not [s for s in exporter.get_finished_spans() if s.name == "ttft"]


class _ScriptedClock:
    """Returns exactly the values handed to it, in call order -- never `time.monotonic`, still
    inside the Global Constraints' "no wall clock in hermetically exercised runtime paths". Lets a
    test pin an exact, large delta to one window and a small one to another without ever sleeping or
    touching a real wall clock (the SAME "inject a controllable double" discipline `_FakeClock` above
    already uses, just with an explicit script instead of a fixed per call increment, which is what
    this fix round's regression test needs to pin down WHEN a span opened, not just whether it did)."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        return self._values.pop(0)


@pytest.mark.asyncio
async def test_ttft_span_measures_from_turn_start_not_after_the_first_graph_event(tmp_path, seed_cassette):
    """The regression this fix round closes (review Important 1): `atlas.stage.ttft_ms` must open at
    THIS edge's true turn start, before the graph has produced a single event -- not merely once the
    first `on_chain_end` arrives. For a direct answer turn that is already AFTER the graph's own
    "agent" node has made its LLM call and returned (`atlas_graph.py` opens "turn" then "agent" -- an
    `llm` kind span, ended instantly, no monotonic wrap of its own -- both complete before this edge
    ever sees an event); opening ttft there would silently exclude generation entirely and measure
    only post generation guard/render overhead, never true time to first token.

    Proven with a scripted clock, not a real sleep: a "slow" ~50 second wide stage runs and fully
    completes (via a fake `stream_events_fn`) BEFORE the graph yields its first real event. The FIXED
    implementation's `tracer.mark()` call (fix round 2 decoupled the clock read from the span's own
    creation; see `chat_app._run_stream`'s docstring) is the very first clock read of the whole turn,
    so it lands BEFORE that stage's own two reads and the reported `atlas.stage.ttft_ms` is dominated
    by that width, regardless of when the ttft SPAN OBJECT itself is later created. Against the
    original buggy implementation (ttft's mark/open both happened inside `_start`, which only fires
    on the first real node event, i.e. AFTER the slow stage has already finished and consumed its own
    two clock reads), the SAME script makes that read land AFTER the slow stage instead, so the
    reported duration would be small (~1 second) -- this assertion fails on that code, which is
    exactly how the bug slipped past the shipped suite's `>= 0` only assertion
    (`test_ttft_stage_span_is_marked_on_a_real_tracer` above)."""
    seed_cassette(tmp_path, [HumanMessage("hi")], {"content": "hello", "tool_calls": []})
    kit = fixture_kit()
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    backend = ActionsBackend(IdFactory("ref"))
    # Call order, in seconds: [0.0, 50.0, 50.0, 51.0, 52.0] -- mark(), the pre generation stand in's
    # open/close, ttft's own backdate calculation at open (fix round 2, unused for the duration
    # itself), then ttft's close. Whichever pair of reads becomes ttft's OWN duration baseline and
    # close reading determines the reported number -- see this test's own docstring for both
    # orderings.
    clock = _ScriptedClock([0.0, 50.0, 50.0, 51.0, 52.0])
    exporter = InMemorySpanExporter()
    tracer = OtelTracer(endpoint="http://example.invalid:4318", config_hash="h", exporter=exporter, clock=clock)
    graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer(), tracer=tracer)

    async def _slow_pre_generation_source(g, state, cfg):
        # Stands in for real, slow work that happens before this edge ever learns `trace_root` (a
        # slow model call, a slow guard, anything upstream of the first `on_chain_end`). A fake
        # "embed" stage span is a convenient, already reserved stage name to carry the scripted
        # clock's two reads for that window; this is not claiming the turn actually retrieved
        # anything.
        pre_seq = tracer.open("embed", "stage", None)
        tracer.close(pre_seq)
        async for ev in g.astream_events(state, cfg, version="v2", durability="sync"):
            yield ev

    app = make_chat_app(kit.clock, graph, tracer=tracer, stream_events_fn=_slow_pre_generation_source)
    async with _client(app) as client:
        token = await _login(client)
        await _stream(client, token, "hi")

    ttft_spans = [s for s in exporter.get_finished_spans() if s.name == "ttft"]
    assert len(ttft_spans) == 1
    ttft_ms = ttft_spans[0].attributes["atlas.stage.ttft_ms"]
    # The pre generation stage alone spans 50 scripted seconds (50_000 ms). ttft measuring only POST
    # generation overhead (the bug) reads ~1_000 ms (the last, small gap) -- nowhere near this.
    # Measuring from true turn start captures the pre generation width too.
    assert ttft_ms >= 50_000, (
        f"ttft_ms={ttft_ms} does not include the pre generation window; ttft is opening too late "
        "(after the graph's first event) instead of at this edge's true turn start"
    )


@pytest.mark.asyncio
async def test_ttft_span_shares_its_turns_trace_id_and_still_measures_from_turn_start(tmp_path, seed_cassette):
    """The regression this fix round closes (review's new Important, surfaced by re review of
    `83c7036`): opening the ttft span with `parent=None` fixed the timing window (fix round 1) but
    orphaned the span at the OTel level -- `parent=None` does not mean "no parent within this trace,"
    it means "start a brand new root span with a brand new, independently random trace_id." A
    downstream consumer that groups spans by trace_id to reconstruct one turn (Phoenix, T7 freeze
    evidence, any T5 alerting that correlates ttft with its own turn) would never find it attached.

    Both properties must hold at once: the exported `ttft` span shares its turn's trace_id and has
    the turn root as its parent (this test, new), AND its reported duration still excludes only true
    post generation overhead, i.e. still measures from turn start (the SAME scripted "slow pre
    generation" scenario `test_ttft_span_measures_from_turn_start_not_after_the_first_graph_event`
    above proves, kept green, never weakened by this fix)."""
    seed_cassette(tmp_path, [HumanMessage("hi")], {"content": "hello", "tool_calls": []})
    kit = fixture_kit()
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    backend = ActionsBackend(IdFactory("ref"))
    # Five scripted reads: mark() at turn start, the pre generation stand in's open/close, ttft's own
    # backdate calculation at open, then ttft's close. See this test's own docstring for the
    # reasoning; a pre fix run simply never consumes the 5th value (4 calls total), which is fine --
    # `_ScriptedClock` only pops what is actually asked for.
    clock = _ScriptedClock([0.0, 50.0, 50.0, 51.0, 52.0])
    exporter = InMemorySpanExporter()
    tracer = OtelTracer(endpoint="http://example.invalid:4318", config_hash="h", exporter=exporter, clock=clock)
    graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer(), tracer=tracer)

    async def _slow_pre_generation_source(g, state, cfg):
        pre_seq = tracer.open("embed", "stage", None)
        tracer.close(pre_seq)
        async for ev in g.astream_events(state, cfg, version="v2", durability="sync"):
            yield ev

    app = make_chat_app(kit.clock, graph, tracer=tracer, stream_events_fn=_slow_pre_generation_source)
    async with _client(app) as client:
        token = await _login(client)
        await _stream(client, token, "hi")

    spans = {s.name: s for s in exporter.get_finished_spans()}
    turn_span = spans["turn"]
    ttft_span = spans["ttft"]

    assert ttft_span.context.trace_id == turn_span.context.trace_id, (
        "ttft exported as a disconnected trace (a brand new, independently random trace_id) instead "
        "of nesting under its own turn"
    )
    assert ttft_span.parent is not None and ttft_span.parent.span_id == turn_span.context.span_id, (
        "ttft is not a direct child of its own turn root"
    )
    ttft_ms = ttft_span.attributes["atlas.stage.ttft_ms"]
    assert ttft_ms >= 50_000, (
        f"ttft_ms={ttft_ms} regressed on the timing fix: it must still include the pre generation "
        "window, not just measure from whenever the span object itself was created"
    )
