"""OtelTracer: the OTel backed sibling to `InMemoryTracer`/`NullTracer` (testing/harness/tracing),
`server.py`'s `ATLAS_TRACING=otel` opt in gate, and (SP6 task 2) the real attribute translation +
stage duration wiring.

Every test here constructs `OtelTracer` directly with an injected in memory exporter, no network,
no real OTLP endpoint -- the hermetic lane never touches `OTEL_EXPORTER_OTLP_*` auto configuration
or a live collector. The gate tests at the bottom prove the OTHER half of the SP6 global constraint:
`server.create_app()`'s default boot path (every OTHER test file in this suite) never constructs an
OtelTracer at all, only `ATLAS_TRACING=otel` does.

Every non "stage" `open(...)` call below uses a REAL, committed inventory shape (the exact
`(name, kind, attr_keys)` tuples `contracts/trace/span_inventory.json` records), never a synthetic
one: since Task 2 wires in fail closed translation, an uninventoried shape now raises rather than
recording verbatim (see `test_trace_translation.py` for the fail closed behavior itself, and
`test_atlas_graph_stage_spans.py` for the real call sites this mirrors).
"""
from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from atlas.adapters.otel_tracer import OtelTracer
from atlas.adapters.trace_translation import BUILD_ATTRIBUTES, TraceTranslationError, UninventoriedSpanError


def _tracer(
    config_hash: str = "hash123", clock=None, corpus_version: str = "", index_build_id: str = "",
    max_tracked_spans: int | None = None,
):
    exporter = InMemorySpanExporter()
    kwargs = {} if max_tracked_spans is None else {"max_tracked_spans": max_tracked_spans}
    tracer = OtelTracer(
        endpoint="http://example.invalid:4318", config_hash=config_hash,
        corpus_version=corpus_version, index_build_id=index_build_id,
        exporter=exporter, clock=clock, **kwargs,
    )
    return tracer, exporter


def test_open_translates_the_guard_ok_key_to_the_dotted_contract_name():
    tracer, exporter = _tracer()
    tracer.open("pre_render_guard", "guard", ok=False, reason="held")
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "pre_render_guard"
    assert spans[0].attributes["atlas.guard.decision"] == "block"
    assert spans[0].attributes["reason"] == "held"  # no reserved counterpart: passthrough, unrenamed
    assert "ok" not in spans[0].attributes  # the raw informal key never reaches the wire


def test_open_returns_an_int_seq_like_every_other_tracer_adapter():
    tracer, _ = _tracer()
    seq = tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")
    assert isinstance(seq, int)


def test_atlas_config_hash_appears_on_the_turn_span_when_tracing_is_active():
    tracer, exporter = _tracer(config_hash="abc123hash")
    tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")
    spans = exporter.get_finished_spans()
    turn_span = next(s for s in spans if s.name == "turn")
    assert turn_span.attributes["atlas.config.hash"] == "abc123hash"


def test_non_turn_spans_do_not_carry_the_config_hash():
    tracer, exporter = _tracer()
    tracer.open("pre_render_guard", "guard", ok=True, reason="")
    spans = exporter.get_finished_spans()
    assert "atlas.config.hash" not in spans[0].attributes


# ---- SP6 task 7 (the v1 freeze): settings sourced turn attributes + build wide constants --------


def test_corpus_version_and_index_build_id_appear_on_the_turn_span_when_given():
    tracer, exporter = _tracer(corpus_version="corpus-0.1.1", index_build_id="idx-8c1d77aa")
    tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")
    span = exporter.get_finished_spans()[0]
    assert span.attributes["atlas.corpus.version"] == "corpus-0.1.1"
    assert span.attributes["atlas.index.build_id"] == "idx-8c1d77aa"


def test_corpus_version_and_index_build_id_are_omitted_not_blank_when_unset():
    tracer, exporter = _tracer()  # defaults: corpus_version="", index_build_id=""
    tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")
    span = exporter.get_finished_spans()[0]
    assert "atlas.corpus.version" not in span.attributes
    assert "atlas.index.build_id" not in span.attributes


def test_corpus_version_and_index_build_id_do_not_appear_on_non_turn_spans():
    tracer, exporter = _tracer(corpus_version="corpus-0.1.1", index_build_id="idx-8c1d77aa")
    tracer.open("pre_render_guard", "guard", ok=True, reason="")
    span = exporter.get_finished_spans()[0]
    assert "atlas.corpus.version" not in span.attributes
    assert "atlas.index.build_id" not in span.attributes


def test_build_attributes_appear_on_every_non_stage_span():
    tracer, exporter = _tracer()
    tracer.open("pre_render_guard", "guard", ok=True, reason="")
    span = exporter.get_finished_spans()[0]
    for key, value in BUILD_ATTRIBUTES.items():
        assert span.attributes[key] == value


def test_build_attributes_appear_on_stage_spans_too():
    """Stage spans bypass `translate_span` entirely (a task defined mechanism, not inventory
    derived), so `BUILD_ATTRIBUTES` must be stamped explicitly in the stage branch, not just
    inherited from the ordinary `translate_span` merge every other kind gets."""
    tracer, exporter = _tracer(clock=_FakeClock())
    seq = tracer.open("embed", "stage")
    tracer.close(seq)
    span = exporter.get_finished_spans()[0]
    for key, value in BUILD_ATTRIBUTES.items():
        assert span.attributes[key] == value


def test_every_span_carries_the_two_universally_required_contract_attributes():
    tracer, exporter = _tracer()
    tracer.open("pre_render_guard", "guard", ok=True, reason="")
    span = exporter.get_finished_spans()[0]
    assert span.attributes["atlas.privacy.synthetic"] is True
    assert span.attributes["atlas.contract.trace_version"]


def test_child_spans_nest_under_their_parent_via_the_returned_seq():
    tracer, exporter = _tracer()
    root = tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")
    tracer.open("agent", "llm", root, model="claude-test")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["agent"].parent.span_id == spans["turn"].context.span_id
    assert spans["agent"].context.trace_id == spans["turn"].context.trace_id


def test_a_root_span_with_no_parent_has_none_parent():
    tracer, exporter = _tracer()
    tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")
    span = exporter.get_finished_spans()[0]
    assert span.parent is None


# ---- I1 fix (SP6 final review): atlas.turn.seq, the envelope/log <-> exported span join key -----


def test_atlas_turn_seq_is_stamped_on_the_turn_span_matching_its_own_returned_seq():
    """`chat_app._resolve_trace_id` hands the client (and the JSON logs) this exact seq, string
    formed, for the turn root -- this is the one attribute that lets a holder of that id find the
    real exported span, and the fix is proven bidirectionally end to end in
    test_trace_id_handoff.py; this is the adapter level unit proof that the attribute exists at
    all, on the span it must match."""
    tracer, exporter = _tracer()
    root = tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")
    span = exporter.get_finished_spans()[0]
    assert span.attributes["atlas.turn.seq"] == str(root)


def test_atlas_turn_seq_is_stamped_on_every_span_kind_with_its_own_distinct_value():
    tracer, exporter = _tracer()
    root = tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")
    tracer.open("agent", "llm", root, model="claude-test")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["turn"].attributes["atlas.turn.seq"] == str(root)
    assert spans["agent"].attributes["atlas.turn.seq"] not in ("", None)
    assert spans["agent"].attributes["atlas.turn.seq"] != spans["turn"].attributes["atlas.turn.seq"]


def test_atlas_turn_seq_is_stamped_on_stage_spans_too():
    tracer, exporter = _tracer(clock=_FakeClock())
    seq = tracer.open("embed", "stage")
    tracer.close(seq)
    span = exporter.get_finished_spans()[0]
    assert span.attributes["atlas.turn.seq"] == str(seq)


def test_dict_valued_attrs_are_json_encoded_not_silently_dropped():
    tracer, exporter = _tracer()
    tracer.open("get_account_summary", "tool", args={"customer_id": "cust_1"}, result="ok")
    span = exporter.get_finished_spans()[0]
    assert span.attributes["args"] == '{"customer_id": "cust_1"}'
    assert span.attributes["result"] == "ok"


def test_list_valued_attrs_of_primitives_pass_through_as_a_sequence():
    tracer, exporter = _tracer()
    tracer.open("bind_guard", "guard", ok=False, intent="policy_question", tools=["change_plan", "cancel_plan"])
    span = exporter.get_finished_spans()[0]
    assert list(span.attributes["tools"]) == ["change_plan", "cancel_plan"]
    assert span.attributes["atlas.guard.decision"] == "block"


def test_none_valued_attrs_are_omitted_not_recorded_as_null():
    tracer, exporter = _tracer()
    tracer.open("refusal", "node", degradation_mode=None)
    span = exporter.get_finished_spans()[0]
    assert "atlas.degradation.mode" not in span.attributes  # the translated key, still omitted
    assert "degradation_mode" not in span.attributes  # the raw key never appears either


def test_annotate_on_a_known_seq_does_not_raise():
    tracer, _ = _tracer()
    seq = tracer.open("pre_render_guard", "guard", ok=True, reason="")
    tracer.annotate(seq, extra="value")  # must not raise, even though the span already ended


def test_annotate_on_an_unknown_seq_is_a_safe_no_op():
    tracer, _ = _tracer()
    tracer.annotate(999, x=1)  # must not raise


def test_default_exporter_is_console_when_none_injected():
    tracer = OtelTracer(endpoint="http://example.invalid:4318", config_hash="h")
    tracer.open("turn", "turn", input="hi", intent="account", customer_id="cust_1")  # must not raise; console exporter


# ---- SP6 task 2: fail closed translation, wired into the adapter's own export path -------------


def test_an_uninventoried_span_shape_raises_instead_of_exporting():
    tracer, _ = _tracer()
    with pytest.raises(UninventoriedSpanError):
        tracer.open("bind_guard", "guard", ok=False, brand_new_key="x")


def test_an_unmapped_key_on_an_otherwise_real_shape_still_fails_closed():
    # `trace_translation.translate_attributes` is exercised directly for THIS specific case in
    # test_trace_translation.py (an inventoried tuple whose key has no rule); here it is enough to
    # prove OtelTracer's own `open()` is really wired to the SAME fail closed function, not a copy.
    tracer, _ = _tracer()
    with pytest.raises(TraceTranslationError):
        tracer.open("nonexistent_span_shape_entirely", "guard", ok=True, unheard_of_key=1)


# ---- SP6 task 2: close(seq) and the five atlas.stage.*ms durations ------------------------------


class _FakeClock:
    """A plain incrementing fake clock (`time.monotonic`'s own shape): each call advances by a
    fixed step, so a stage's elapsed duration is exactly `step_seconds * 1000` ms, deterministic."""

    def __init__(self, step_seconds: float = 0.25) -> None:
        self._value = 0.0
        self._step = step_seconds

    def __call__(self) -> float:
        current = self._value
        self._value += self._step
        return current


def test_close_ends_a_pending_stage_span_and_stamps_its_duration():
    clock = _FakeClock(step_seconds=0.25)  # open at t=0, close reads t=0.25 -> 250ms
    tracer, exporter = _tracer(clock=clock)
    seq = tracer.open("embed", "stage")
    assert not exporter.get_finished_spans()  # still open: a stage span never ends inside open()
    tracer.close(seq)
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "embed"
    assert spans[0].attributes["atlas.stage.embed_ms"] == pytest.approx(250.0)


@pytest.mark.parametrize(
    ("stage_name", "attr"),
    [
        ("embed", "atlas.stage.embed_ms"),
        ("retrieve", "atlas.stage.retrieve_ms"),
        ("rerank", "atlas.stage.rerank_ms"),
        ("assemble", "atlas.stage.assemble_ms"),
        ("ttft", "atlas.stage.ttft_ms"),
    ],
)
def test_every_reserved_stage_name_maps_to_its_own_duration_attribute(stage_name, attr):
    tracer, exporter = _tracer(clock=_FakeClock())
    seq = tracer.open(stage_name, "stage")
    tracer.close(seq)
    span = exporter.get_finished_spans()[0]
    assert attr in span.attributes


def test_an_unrecognized_stage_name_fails_closed():
    tracer, _ = _tracer()
    with pytest.raises(TraceTranslationError):
        tracer.open("not_a_real_stage", "stage")


def test_a_never_closed_stage_span_never_exports():
    """A skipped rung (`atlas_graph.py`'s read loop only closes the stages that actually ran): the
    OTel `SimpleSpanProcessor` only flushes an ended span, so a never closed stage's ABSENCE from the
    export IS the signal (matches `degraded_turn.json`: no `atlas.stage.rerank_ms` on a drop_rerank
    turn), never a bug to work around."""
    tracer, exporter = _tracer()
    tracer.open("rerank", "stage")  # never closed
    assert not exporter.get_finished_spans()


def test_close_on_an_already_closed_seq_is_a_safe_no_op():
    tracer, exporter = _tracer(clock=_FakeClock())
    seq = tracer.open("embed", "stage")
    tracer.close(seq)
    tracer.close(seq)  # must not raise, must not double count or re end
    assert len(exporter.get_finished_spans()) == 1


def test_close_on_a_non_stage_seq_is_a_safe_no_op():
    """Every non "stage" kind already ended inside `open()` (Task 1's own behavior, unchanged); a
    caller that (harmlessly) calls `close()` on one anyway must not raise or re end it."""
    tracer, exporter = _tracer()
    seq = tracer.open("pre_render_guard", "guard", ok=True, reason="")
    tracer.close(seq)  # must not raise
    assert len(exporter.get_finished_spans()) == 1


def test_close_on_an_unknown_seq_is_a_safe_no_op():
    tracer, _ = _tracer()
    tracer.close(999)  # must not raise


# ---- I2 fix (SP6 final review): _spans/_pending_stage stay bounded, not retained forever ---------


def test_spans_and_pending_stage_stay_bounded_by_the_configured_cap():
    """`OtelTracer._spans` used to retain every span ever opened, ended ones included, for the life
    of the process -- one OtelTracer per process, dev/prod only, so this leak existed exactly and
    only in the long running configuration no hermetic test measured. Fixed with a bounded FIFO
    cache (module level `_MAX_TRACKED_SPANS`), not a "drop the previous turn's entries the moment a
    fresh turn opens" sweep: that obvious sounding fix is wrong for THIS adapter specifically, since
    one instance is shared across every request an async server handles concurrently, and
    interleaving two overlapping turns' own `open()` calls would sweep away a still in flight turn's
    bookkeeping, not just a finished one, silently orphaning its next child span into a brand new,
    disconnected trace -- the exact regression fix round 2 already fixed once
    (`test_open_with_start_at_still_nests_under_the_given_parent` and friends), reintroduced by
    construction (see `test_interleaved_concurrent_turns_never_corrupt_each_others_parent_linkage`
    below, which reproduces exactly this against the CURRENT, fixed implementation and stays green).
    `max_tracked_spans` is injected small here (test only; production leaves the generous module
    default) so eviction is observed deterministically without needing thousands of real spans.
    Reaching into the private `_spans`/`_pending_stage` dicts is a deliberate exception to normal
    black box testing here: a memory leak has no other observable surface."""
    tracer, exporter = _tracer(clock=_FakeClock(), max_tracked_spans=10)
    for _ in range(50):
        root = tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")
        tracer.open("agent", "llm", root, model="claude-test")
        embed_seq = tracer.open("embed", "stage", root)
        tracer.close(embed_seq)
        tracer.open("retrieve", "stage", root)  # never closed: a skipped rung / an error path

    # every turn's own spans still exported correctly -- bookkeeping eviction never touches anything
    # that has already left the process via the exporter.
    assert len(exporter.get_finished_spans()) == 50 * 3  # turn + agent + embed; retrieve never exports

    # bounded by the configured cap, never 50 turns' worth (200 entries).
    assert len(tracer._spans) <= 10
    assert len(tracer._pending_stage) <= 10


def test_interleaved_concurrent_turns_never_corrupt_each_others_parent_linkage():
    """The concurrency property the bounded FIFO design (above) exists to protect, reproduced
    directly: two turns opened INTERLEAVED on the SAME tracer instance (turn B opens before turn A's
    own child span does, the same ordering two overlapping `/chat/stream` requests on a shared async
    server would produce) -- turn A's later child must still nest under turn A's own root, never
    come out as a new, disconnected root. This is exactly the shape a naive "clear all bookkeeping
    when a fresh turn opens" fix (rejected, see the sibling test above) would have broken."""
    tracer, exporter = _tracer()
    root_a = tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")   # turn A starts
    root_b = tracer.open("turn", "turn", input="hi", intent="account", customer_id="cust_1")      # turn B starts, interleaved
    tracer.open("agent", "llm", root_a, model="claude-test")                # A's own child, opened LAST
    tracer.open("agent", "llm", root_b, model="claude-test")

    spans = list(exporter.get_finished_spans())
    turn_a = next(s for s in spans if s.attributes.get("atlas.turn.seq") == str(root_a))
    turn_b = next(s for s in spans if s.attributes.get("atlas.turn.seq") == str(root_b))
    agents = [s for s in spans if s.name == "agent"]
    assert len(agents) == 2
    agent_a = next(s for s in agents if s.parent is not None and s.parent.span_id == turn_a.context.span_id)
    agent_b = next(s for s in agents if s.parent is not None and s.parent.span_id == turn_b.context.span_id)
    assert agent_a.context.trace_id == turn_a.context.trace_id
    assert agent_b.context.trace_id == turn_b.context.trace_id


def test_stage_spans_nest_under_their_turn_like_any_other_span():
    tracer, exporter = _tracer(clock=_FakeClock())
    root = tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")
    seq = tracer.open("embed", "stage", root)
    tracer.close(seq)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["embed"].parent.span_id == spans["turn"].context.span_id


class _ListClock:
    """Returns exactly the values handed to it, in call order -- a `_ScriptedClock` in miniature
    (`test_trace_id_handoff.py`'s own version, not imported here since this file tests the adapter
    in isolation). Lets a test pin an exact reading to `mark()` and a DIFFERENT, later reading to
    whatever `open()`/`close()` read next, the precision `start_at` backdating needs to prove."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        return self._values.pop(0)


def test_open_with_start_at_backdates_the_reported_duration_to_the_earlier_mark():
    """SP6 task 2 fix round 2: a caller can `mark()` now and `open(..., start_at=mark)` LATER (once
    it knows the real parent to nest a span under), and the reported `atlas.stage.*ms` duration is
    still measured from that earlier mark, not from whenever `open()` itself happens to run -- the
    ttft use case (`chat_app.py` marks true turn start but cannot create the actual, correctly
    parented span until the graph's own turn root exists, the fix for the trace connectivity
    regression `parent=None` introduced in fix round 1)."""
    clock = _ListClock([0.0, 10.0, 40.0])  # mark(), open()'s own backdate read, close()'s read
    tracer, exporter = _tracer(clock=clock)
    mark = tracer.mark()
    seq = tracer.open("ttft", "stage", start_at=mark)
    tracer.close(seq)
    span = exporter.get_finished_spans()[0]
    # 40.0 - 0.0 (the mark), never 40.0 - 10.0 (open()'s own, later read).
    assert span.attributes["atlas.stage.ttft_ms"] == pytest.approx(40_000.0)


def test_open_with_start_at_still_nests_under_the_given_parent():
    clock = _ListClock([0.0, 10.0, 40.0])
    tracer, exporter = _tracer(clock=clock)
    root = tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")
    mark = tracer.mark()
    seq = tracer.open("ttft", "stage", root, start_at=mark)
    tracer.close(seq)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert spans["ttft"].parent.span_id == spans["turn"].context.span_id
    assert spans["ttft"].context.trace_id == spans["turn"].context.trace_id


# ---- server.py's ATLAS_TRACING=otel opt in gate ----------------------------------------------


def test_default_boot_never_constructs_otel_tracer(tmp_path, monkeypatch):
    from atlas import server

    monkeypatch.delenv("ATLAS_TRACING", raising=False)
    monkeypatch.setenv("ATLAS_MODE", "replay")
    monkeypatch.setenv("ATLAS_CASSETTES", str(tmp_path))

    def _boom(*_a, **_k):
        raise AssertionError("OtelTracer must not be constructed when ATLAS_TRACING is unset")

    monkeypatch.setattr("atlas.adapters.otel_tracer.OtelTracer", _boom)
    app = server.create_app()  # must not raise
    assert app is not None


def test_an_unrecognised_atlas_tracing_value_also_keeps_nulltracer(tmp_path, monkeypatch):
    """`server.py`'s gate is deliberately permissive here (unlike ATLAS_MODE/ATLAS_CHECKPOINTER/
    ATLAS_RETRIEVER, which fail fast on a typo): a typo'd ATLAS_TRACING must never accidentally
    construct a real, network capable adapter. Only the exact string "otel" opts in."""
    from atlas import server

    monkeypatch.setenv("ATLAS_MODE", "replay")
    monkeypatch.setenv("ATLAS_CASSETTES", str(tmp_path))
    monkeypatch.setenv("ATLAS_TRACING", "otle")  # typo

    def _boom(*_a, **_k):
        raise AssertionError("a typo'd ATLAS_TRACING must not construct OtelTracer")

    monkeypatch.setattr("atlas.adapters.otel_tracer.OtelTracer", _boom)
    app = server.create_app()  # must not raise
    assert app is not None


def test_atlas_tracing_otel_constructs_the_real_adapter_with_the_config_hash(tmp_path, monkeypatch):
    from atlas import server

    monkeypatch.setenv("ATLAS_MODE", "replay")
    monkeypatch.setenv("ATLAS_CASSETTES", str(tmp_path))
    monkeypatch.setenv("ATLAS_TRACING", "otel")

    built = []

    def _fake_otel_tracer(**kwargs):
        built.append(kwargs)
        return object()

    monkeypatch.setattr("atlas.adapters.otel_tracer.OtelTracer", _fake_otel_tracer)
    server.create_app()
    assert len(built) == 1
    assert built[0]["endpoint"] == "http://localhost:4318"
    assert len(built[0]["config_hash"]) == 64  # a sha256 hex digest, not the raw settings


def test_atlas_tracing_otel_constructs_exactly_one_tracer_shared_by_the_graph_and_chat_app(tmp_path, monkeypatch):
    """SP6 task 2: `server.py`'s `create_app()` now wires the SAME tracer instance into BOTH
    `build_atlas_graph` (so its own `tracer.open("turn", ...)` mints the real trace id) and
    `make_chat_app` (so chat_app can mark ttft on that SAME span tree) -- constructed exactly ONCE,
    never twice, so real tracing never opens two independent OTel providers per process."""
    from atlas import server

    monkeypatch.setenv("ATLAS_MODE", "replay")
    monkeypatch.setenv("ATLAS_CASSETTES", str(tmp_path))
    monkeypatch.setenv("ATLAS_TRACING", "otel")

    built = []

    def _fake_otel_tracer(**kwargs):
        built.append(kwargs)
        return object()

    monkeypatch.setattr("atlas.adapters.otel_tracer.OtelTracer", _fake_otel_tracer)
    server.create_app()
    assert len(built) == 1
