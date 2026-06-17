"""OtelTracer: the OTel backed sibling to `InMemoryTracer`/`NullTracer` (``testing/harness/tracing``).
Constructed behind `ATLAS_TRACING=otel` ONLY (`server.py`'s `_tracer` opt in gate); never the
default anywhere else, and NEVER constructed by a hermetic/collected test path (SP6's own
determinism constraint) -- `InMemoryTracer` stays the CI adapter, this is its dev/prod sibling.
Implements the SAME `Tracer` protocol (``open``/``annotate``/``close``), so `atlas_graph.py`'s ~20
`tracer.open(...)` call sites, owned by SP3/SP4, stay completely untouched.

SP6 task 2: every non "stage" span is translated (`atlas.adapters.trace_translation`) before it
lands on the wire -- the informal short names call sites use (``ok``, ``model``,
``degradation_mode``, ...) become the frozen contract's dotted ``atlas.*``/``gen_ai.*`` names, and
an unreviewed shape FAILS CLOSED (raises) rather than exporting the wrong vocabulary. A "stage" kind
span (embed/retrieve/rerank/assemble/ttft, opened by `atlas_graph.py`'s read loop and `chat_app.py`
at first token) is a SEPARATE, task defined mechanism: it carries no informal attributes to
translate, stays open past `open()` (unlike every other kind, still ended immediately -- Task 1's
own "opens AND ends inside open()" behavior, byte unchanged for those), and `close(seq)` computes
its real elapsed duration via the injected monotonic clock and stamps the ONE matching
`atlas.stage.*ms` attribute (`trace_translation.STAGE_DURATION_ATTRIBUTE`) before ending it. A stage
span that never closes (a skipped rung: `atlas_graph.py`'s read loop only closes the stages that
actually ran) simply never exports -- `SimpleSpanProcessor` only flushes an ended span, so a
never closed stage's absence from the exported trace IS the signal, not a bug.

The OTLP wire exporter lives in the optional `observability` dependency group (Task 3), synced only
by an operator running the collector profile; the hermetic lane still exports in memory (tests, via
an injected exporter) or console only. `endpoint` is the explicit constructor argument; nothing in
this module reads `OTEL_EXPORTER_OTLP_*` auto configuration env vars, so there is nothing here for
those vars to silently influence (the SP6 global constraint this module exists to satisfy).
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any, Optional

from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span as OtelSpan

from atlas.adapters import trace_translation

_INSTRUMENTATION_NAME = "atlas"
_PRIMITIVE_ATTR_TYPES = (bool, str, bytes, int, float)
_OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
# I1 fix (SP6 final review): the join key between the envelope/log trace id
# (`chat_app._resolve_trace_id`, always this exact seq, string formed, for the turn root) and a real
# exported span -- stamped on EVERY span this adapter exports, not only "turn", so any span (not
# just the root) can be traced back to the process local id a client or a log line holds, and the
# root's own value (the only one a client ever sees as `trace_id`) is what actually joins. A NEW
# `RESERVED_TRACE_ATTRIBUTES` member (`testing/harness/contract_tools/loader.py`), a MINOR contract
# bump (`contracts/trace/schema.json` 1.0.0 -> 1.1.0, CHANGELOG.md, `contract_tools.freeze_check`).
_ATLAS_TURN_SEQ_ATTRIBUTE = "atlas.turn.seq"

# I2 fix (SP6 final review): `_spans`/`_pending_stage` used to retain every span forever, for the
# life of the process (one OtelTracer per process, dev/prod only). A generous, constructor
# overridable cap on how many `_spans` entries this adapter ever tracks at once -- see `open()`'s
# own comment for why this is a bounded FIFO cache, not a "drop the previous turn" sweep (the
# obvious sounding fix, and wrong: this adapter is one shared instance across every concurrent
# request an async server handles, and two overlapping turns interleaving their own `open()` calls
# would sweep each other's still needed bookkeeping, silently orphaning a child span into a brand
# new, disconnected trace -- the exact regression fix round 2 already fixed once, reintroduced by
# construction). A real turn opens on the order of 10-20 spans in at most a few seconds; this cap
# leaves generous headroom for realistic overlap while still bounding growth firmly across a long
# running process's thousands of turns.
_MAX_TRACKED_SPANS = 4096


def _coerce_attr(value: Any) -> Any:
    """OTel span attributes accept only primitives, or a homogeneous sequence of them; anything
    else (today: the ``args``/``result`` style dict payloads a few call sites pass) is JSON encoded
    rather than silently dropped -- the SDK's own `set_attribute` drops an unsupported type with
    only a logged warning, which would lose information with no test able to catch it. `None` is
    omitted outright (absent, never a null attribute), the same "absent fields omitted not nulled"
    convention Task 4's structured logs use for the same reason."""
    if value is None:
        return None
    if isinstance(value, _PRIMITIVE_ATTR_TYPES):
        return value
    if isinstance(value, (list, tuple)) and all(isinstance(v, _PRIMITIVE_ATTR_TYPES) for v in value):
        return list(value)
    return json.dumps(value, sort_keys=True, default=str)


def _set_attrs(span: OtelSpan, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        coerced = _coerce_attr(value)
        if coerced is not None:
            span.set_attribute(key, coerced)


class OtelTracer:
    """The `Tracer` port's OTel adapter (dev/prod). Constructed ONLY behind `ATLAS_TRACING=otel`;
    the hermetic lane never builds one. `clock` is an injected `Callable[[], float]`
    (`time.monotonic`'s own shape, the SAME "inject with a live default, never inside the primitive
    itself" discipline `CircuitBreaker`/`PgvectorRetriever` already use), so a test can walk the
    stage duration math deterministically instead of racing a real clock.

    `max_tracked_spans` (I2 fix, SP6 final review): bounds how many `_spans`/`_pending_stage`
    entries this ONE process wide instance ever retains at once (module level `_MAX_TRACKED_SPANS`
    comment has the full reasoning and why this is a bounded FIFO cache, never a "drop the previous
    turn" sweep, which would corrupt a concurrently in flight turn's own parent linkage). Test only
    override; production code never passes it.

    `mark()` / `open(..., start_at=...)` (SP6 task 2 fix round 2): fixes a trace connectivity
    regression fix round 1 introduced. Round 1 opened the `ttft` stage span with `parent=None` at
    true turn start, before the graph's own "turn" span existed to nest under; at the OTel level
    `parent=None` does not mean "no parent within this trace," it means "start a brand new root span
    with a brand new, independently random `trace_id`" (`trace_api.set_span_in_context` is only
    called when a parent span object is actually found). `ttft` shipped as a disconnected, single
    span trace, once per turn, forever. The fix: `mark()` reads `self._clock()` with NO span created
    at all, so a caller can capture "now" before it knows the real parent; `open(..., start_at=mark)`
    creates the actual span LATER, once the real parent IS known, so it nests correctly, but backdates
    both the reported duration (`_pending_stage`'s own baseline) AND the span's native OTel
    `start_time` to that earlier mark, never to whenever `open()` itself happens to run. `start_at`
    defaults to `None` for every other call site (unchanged): `atlas_graph.py`'s embed/retrieve/
    rerank/assemble opens still read `self._clock()` fresh, inside `open()`, exactly as before this
    addition."""

    def __init__(
        self, endpoint: str, config_hash: str, *,
        corpus_version: str = "", index_build_id: str = "",
        exporter: Optional[SpanExporter] = None, clock: Optional[Callable[[], float]] = None,
        max_tracked_spans: int = _MAX_TRACKED_SPANS,
    ) -> None:
        self._endpoint = endpoint  # carried for Task 3's OTLP exporter; unused here on purpose
        self._config_hash = config_hash
        # SP6 task 7 (the v1 freeze): `AtlasSettings`' own `corpus_version`/`index_build_id` fields
        # (SP6 task 6, D37) already name which corpus/index this process serves -- the SAME "config
        # identity, stamped on every turn span" treatment `atlas.config.hash` already gets, just two
        # fields this module did not yet map. Defaulted to "" (never None) so an unthreaded caller
        # (every existing hermetic test that builds an `OtelTracer` directly) keeps working
        # unchanged; `open()`'s own turn branch below only stamps either one when it is truthy (an
        # unbuilt index's own empty default never exports a blank attribute).
        self._corpus_version = corpus_version
        self._index_build_id = index_build_id
        self._clock = clock or time.monotonic
        self._provider = TracerProvider(resource=Resource.create({"service.name": _INSTRUMENTATION_NAME}))
        self._provider.add_span_processor(SimpleSpanProcessor(exporter or ConsoleSpanExporter()))
        self._tracer = self._provider.get_tracer(_INSTRUMENTATION_NAME)
        self._spans: dict[int, OtelSpan] = {}
        # Stage spans (SP6 task 2) stay open past `open()`, unlike every other kind: seq -> (stage
        # name, monotonic open time), popped by `close(seq)` once the matching duration is known.
        self._pending_stage: dict[int, tuple[str, float]] = {}
        self._next_seq = 0
        # I2 fix (SP6 final review): a generous, constructor overridable cap (test only; production
        # code never overrides it) -- see the module level `_MAX_TRACKED_SPANS` comment for why this
        # is a bounded FIFO cache, never a "drop the previous turn" sweep.
        self._max_tracked_spans = max_tracked_spans

    def mark(self) -> float:
        """A raw monotonic clock reading (SP6 task 2 fix round 2), with NO span attached -- see this
        class's own docstring for why a caller needs this decoupled from `open()`."""
        return self._clock()

    def open(
        self, name: str, kind: str, parent: Optional[int] = None, *,
        start_at: Optional[float] = None, **attrs,
    ) -> int:
        parent_span = self._spans.get(parent) if parent is not None else None
        context = trace_api.set_span_in_context(parent_span) if parent_span is not None else None
        start_time_ns = None
        if start_at is not None:
            # Convert an earlier MONOTONIC mark into an approximate EPOCH start time for THIS span
            # only (never used for the duration math below, which stays purely monotonic): "now, in
            # epoch terms, minus how much monotonic time has elapsed since the mark." `time.time_ns`
            # lives here, confined to the adapter (never a hermetically exercised runtime path; this
            # class is never constructed there), the SAME discipline `self._clock`'s own default
            # (`time.monotonic`) already relies on.
            elapsed_ns = max(0, int((self._clock() - start_at) * 1_000_000_000))
            start_time_ns = time.time_ns() - elapsed_ns
        span = self._tracer.start_span(name, context=context, start_time=start_time_ns)
        seq = self._next_seq
        self._next_seq += 1
        self._spans[seq] = span
        if len(self._spans) > self._max_tracked_spans:
            # I2 fix (SP6 final review): bounded FIFO eviction, not a "drop the previous turn" sweep
            # (see the module level `_MAX_TRACKED_SPANS` comment for why that alternative is
            # concurrency unsafe for this one adapter instance shared across every in flight
            # request). Plain dicts are insertion ordered (Python's own guarantee), so the OLDEST
            # tracked seq is always `next(iter(...))`; evicting it here just means "if this span is
            # ever needed as a PARENT again, it no longer resolves and a later child becomes a new,
            # disconnected root" -- the same honest degradation an in memory cache anywhere else in
            # this codebase already accepts once truly ancient. With a turn opening on the order of
            # 10-20 spans in at most a few seconds, only a span from a turn long since finished (or
            # an extreme concurrency spike far beyond this reference system's own demo load) is ever
            # actually at risk.
            oldest_seq = next(iter(self._spans))
            del self._spans[oldest_seq]
            self._pending_stage.pop(oldest_seq, None)
        # I1 fix (SP6 final review): set BEFORE any branch below ends the span (non-stage kinds end
        # inside this same call) -- an attribute write on an already ended span is silently dropped
        # by the OTel SDK (M6, a documented, separate loaded seam), so this must land first.
        span.set_attribute(_ATLAS_TURN_SEQ_ATTRIBUTE, str(seq))

        if kind == "stage":
            # A task defined mechanism (trace_translation.py's own module docstring), not derived
            # from the observed span inventory: no informal attrs to translate, so this bypasses
            # translate_span entirely and fails closed on its OWN, simpler check instead (an
            # unrecognized stage name) rather than the inventory's "was this ever observed" one.
            if name not in trace_translation.STAGE_DURATION_ATTRIBUTE:
                raise trace_translation.TraceTranslationError(
                    f"unknown stage span name {name!r}; expected one of "
                    f"{sorted(trace_translation.STAGE_DURATION_ATTRIBUTE)}"
                )
            _set_attrs(span, trace_translation.REQUIRED_SPAN_ATTRIBUTES)
            # SP6 task 7: the same build wide constants every OTHER exported span carries
            # (`translate_span`'s own merge) -- a stage span bypasses `translate_span` entirely
            # (this branch's own comment above), so it must pick these up explicitly instead.
            _set_attrs(span, trace_translation.BUILD_ATTRIBUTES)
            span.set_attribute(_OPENINFERENCE_SPAN_KIND, trace_translation.span_kind_for(name, kind))
            # The duration baseline is the earlier MARK when one was given, never a fresh read here
            # -- that is the whole point (this span may have been created well after its true
            # measurement start; see this class's own docstring).
            self._pending_stage[seq] = (name, start_at if start_at is not None else self._clock())
            return seq  # left open: close(seq) ends it once its duration is known

        translated = trace_translation.translate_span(name, kind, attrs)
        if kind == "turn":  # the production spine's config identity, on every trace from span one
            span.set_attribute("atlas.config.hash", self._config_hash)
            # SP6 task 7: `atlas.corpus.version`/`atlas.index.build_id` are settings sourced facts,
            # like `atlas.config.hash` just above -- `trace_translation.py` owns no `AtlasSettings`
            # dependency (it stays framework/settings free, `test_trace_translation_module_stays_pure`),
            # so these two are stamped here, the one place this adapter already reaches into
            # constructor injected identity, rather than in that module's own `BUILD_ATTRIBUTES`.
            if self._corpus_version:
                span.set_attribute("atlas.corpus.version", self._corpus_version)
            if self._index_build_id:
                span.set_attribute("atlas.index.build_id", self._index_build_id)
        _set_attrs(span, translated["attributes"])
        span.set_attribute(_OPENINFERENCE_SPAN_KIND, translated["span_kind"])
        span.end()
        return seq

    def annotate(self, seq: int, **attrs) -> None:
        span = self._spans.get(seq)
        if span is None:
            return
        _set_attrs(span, attrs)

    def close(self, seq: int) -> None:
        """Ends a still open STAGE span (SP6 task 2) and stamps its real elapsed duration onto the
        ONE matching `atlas.stage.*ms` attribute. A safe no op for an unknown `seq` or a `seq` that
        already ended inside `open()` (every non stage kind, and a stage span this same `seq`
        already closed once) -- mirrors `annotate`'s own documented no op on an ended span."""
        pending = self._pending_stage.pop(seq, None)
        if pending is None:
            return
        name, opened_at = pending
        elapsed_ms = (self._clock() - opened_at) * 1000.0
        span = self._spans.get(seq)
        if span is not None:
            span.set_attribute(trace_translation.STAGE_DURATION_ATTRIBUTE[name], elapsed_ms)
            span.end()


__all__ = ["OtelTracer"]
