"""The join run after the fact against Phoenix spans (SP9 task 6, D31): `k6/chat_sse_load.js` never
talks to Phoenix directly. Each SSE iteration stamps its own real, backend assigned `trace_id` (the
`message_start` event's own field, `/chat/stream`'s response envelope) onto a small structured
record; THIS module is the pure, hermetically testable join between a whole burst run's worth of
those records and a Phoenix span export, using the EXACT two step mechanism SP6's final review fix
wave built and proved end to end -- verified still present on this branch's HEAD, not assumed
(`backend/atlas/adapters/otel_tracer.py`'s `_ATLAS_TURN_SEQ_ATTRIBUTE`,
`testing/tests/test_trace_id_handoff.py::test_atlas_turn_seq_joins_the_envelope_trace_id_to_its_exported_span_both_directions`,
cross checked again here by `test_load_phoenix_join.py`'s own
`test_the_join_key_constant_matches_what_a_real_otel_tracer_actually_stamps` against a real,
in process `OtelTracer`):

  (1) `atlas.turn.seq` is stamped on EVERY exported span with its OWN, distinct sequence value
      (never shared across a turn's whole span tree) -- so the ONE span anywhere in a flat export of
      the whole run whose `atlas.turn.seq` equals a k6 iteration's own `trace_id` is that
      iteration's anchor (`find_anchor_span`). Zero or more than one match is reported as an
      unjoinable `JoinMiss`, never a guess: an ambiguous anchor is not a safe join.
  (2) every OTHER span sharing that anchor's own REAL `trace_id` (Phoenix's native grouping key, a
      128 bit OTel trace id -- never `atlas.turn.seq` itself, which belongs to one span, not to a
      whole trace) belongs to the SAME turn; the STAGE spans among them (`STAGE_NAMES`, matching
      `trace_translation.STAGE_DURATION_ATTRIBUTE`'s own keys exactly) each carry the ONE
      `atlas.stage.<name>_ms` attribute `otel_tracer.py`'s `close(seq)` stamps
      (`stage_latencies_for_real_trace_id`).

`SpanRecord`'s shape (`span_id`/`trace_id`/`name`/`attributes`) is the generic OTel span shape, not
a literal Phoenix REST/GraphQL export schema -- turning a REAL live Phoenix export into this exact
shape is the live burst wiring step this task does not build (mirrors SP9 task 2's own Neo4j live
extraction arm: the mechanism and its hermetic proof land here, the live client is an operator or
dev/prod concern, documented, not silently skipped).

A k6 iteration whose `trace_id` matches no exported span at all (a genuinely lost, or not yet
flushed, turn) is a `JoinMiss`, always listed, never silently dropped -- the same "never silent"
doctrine the matrix's own `dropped_cells` already applies to a cell the spend gate skipped
(`matrix.spend_gate`).

Fully pure: every dataclass here is frozen, every function a plain transform over already loaded
Python data (a fixture span set in every hermetic test; a real k6 NDJSON capture and a real Phoenix
export, both read from a local path, are the ONLY file I/O this module performs -- no network
anywhere in this file). Determinism: `join_iterations_to_spans` sorts every list and dict key it
produces (concurrency step, stage name, then the latency values themselves), never leaving that
order to a dict/set's own insertion accident.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

# The SP6 fix wave's own join key (`otel_tracer._ATLAS_TURN_SEQ_ATTRIBUTE`), exported again here
# under the SAME literal rather than declared a second time independently -- see this module's own
# docstring for why an independently maintained second copy of this string would be exactly the
# kind of silent drift the "verify it exists" instruction (SP9 task 6) is guarding against.
TURN_SEQ_ATTRIBUTE = "atlas.turn.seq"

# The five stage names this repo's own tracer ever closes a duration for
# (`trace_translation.STAGE_DURATION_ATTRIBUTE`'s keys); cross checked, never derived a second time
# by hand, in `test_load_phoenix_join.py::test_stage_names_cover_every_stage_this_repo_actually_emits`.
STAGE_NAMES: tuple[str, ...] = ("embed", "retrieve", "rerank", "assemble", "ttft")

_ITER_LINE_PREFIX = "LOAD_ITER "


@dataclass(frozen=True)
class SpanRecord:
    """One exported span, the generic OTel shape a real Phoenix export is adapted into (this
    module's own docstring covers the live wiring gap)."""

    span_id: str
    trace_id: str
    name: str
    attributes: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class IterationRecord:
    """One k6 SSE iteration's own capture (`k6/chat_sse_load.js`'s `LOAD_ITER` console line).
    `ttft_ms` is `None` for a turn that never streamed a first token at all (a truncated/error
    path) -- never a fabricated zero standing in for "we do not know"."""

    trace_id: str
    concurrency: int
    ttft_ms: Optional[float]
    tokens_per_sec: float
    e2e_ms: float
    goodput: bool
    prompt_id: str = ""


@dataclass(frozen=True)
class JoinMiss:
    """One k6 iteration whose `trace_id` matched no span (or an ambiguous set of more than one
    span) in the given export -- listed, never silently dropped."""

    trace_id: str
    concurrency: int
    reason: str


@dataclass(frozen=True)
class JoinResult:
    """`per_concurrency`: `{concurrency_step: {stage_name: (sorted latency ms values, ...)}}`.
    `misses`: every k6 iteration this join could not anchor to a real span, with why."""

    per_concurrency: Mapping[int, Mapping[str, tuple[float, ...]]]
    misses: tuple[JoinMiss, ...]


def stage_duration(span: SpanRecord) -> Optional[tuple[str, float]]:
    """`(stage_name, duration_ms)` for the ONE `atlas.stage.*_ms` attribute a stage span carries, or
    `None` for a span that is not a stage at all (a "turn"/"agent"/etc span carries no such
    attribute)."""
    for stage in STAGE_NAMES:
        key = f"atlas.stage.{stage}_ms"
        if key in span.attributes:
            return stage, float(span.attributes[key])
    return None


def find_anchor_span(spans: Sequence[SpanRecord], client_trace_id: str) -> Optional[SpanRecord]:
    """Direction one: the client visible `trace_id` (`message_start`'s own field) -> the ONE
    exported span whose `atlas.turn.seq` equals it. Zero or more than one match returns `None`
    rather than guessing -- an ambiguous anchor is not a safe join (this module's own docstring)."""
    matches = [s for s in spans if str(s.attributes.get(TURN_SEQ_ATTRIBUTE)) == str(client_trace_id)]
    if len(matches) != 1:
        return None
    return matches[0]


def stage_latencies_for_real_trace_id(spans: Sequence[SpanRecord], real_trace_id: str) -> dict[str, float]:
    """Direction two: every span sharing the SAME real (OTel) `trace_id` belongs to one turn;
    return its stage durations, keyed by stage name. Spans from every OTHER trace in the export are
    never touched, even if this turn ran concurrently with dozens of others."""
    out: dict[str, float] = {}
    for span in spans:
        if span.trace_id != real_trace_id:
            continue
        stage = stage_duration(span)
        if stage is not None:
            name, ms = stage
            out[name] = ms
    return out


def join_iterations_to_spans(iterations: Sequence[IterationRecord], spans: Sequence[SpanRecord]) -> JoinResult:
    """Both directions, end to end, grouped by the iteration's own concurrency step. Sorted at every
    level (concurrency step, stage name, and the latency values within a stage) so two joins over
    the same inputs agree byte for byte, never an accident of dict/set iteration order."""
    buckets: dict[int, dict[str, list[float]]] = {}
    misses: list[JoinMiss] = []
    for iteration in iterations:
        anchor = find_anchor_span(spans, iteration.trace_id)
        if anchor is None:
            misses.append(JoinMiss(
                trace_id=iteration.trace_id, concurrency=iteration.concurrency,
                reason=f"no unique span with {TURN_SEQ_ATTRIBUTE}={iteration.trace_id!r} in the given export",
            ))
            continue
        stages = stage_latencies_for_real_trace_id(spans, anchor.trace_id)
        bucket = buckets.setdefault(iteration.concurrency, {})
        for stage, ms in stages.items():
            bucket.setdefault(stage, []).append(ms)
    per_concurrency = {
        step: {stage: tuple(sorted(values)) for stage, values in sorted(stages.items())}
        for step, stages in sorted(buckets.items())
    }
    return JoinResult(per_concurrency=per_concurrency, misses=tuple(misses))


def percentile(values: Sequence[float], pct: float) -> float:
    """A plain nearest rank quantile -- deliberately NOT `quality.stats`' bootstrap CI machinery
    (that answers "how uncertain is this estimate," a different question from "what is the p95 of
    these already observed latencies"). Raises on an empty sequence or a percentage outside the
    valid range rather than returning a silently meaningless number."""
    if not values:
        raise ValueError("percentile of an empty sequence is undefined")
    if not 0 <= pct <= 100:
        raise ValueError(f"pct must be within [0, 100], got {pct}")
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[rank]


def summarize_by_concurrency(join_result: JoinResult) -> dict[int, dict[str, dict[str, float]]]:
    """`{concurrency_step: {stage_name: {"n": ..., "p50": ..., "p95": ...}}}` -- the reporting layer
    over an already joined result, the shape a saturation knee narrative reads directly (D31: the
    knee is predicted first at the reranker, a cross encoder scoring k candidates per query on CPU)."""
    return {
        step: {
            stage: {"n": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95)}
            for stage, values in stages.items()
        }
        for step, stages in join_result.per_concurrency.items()
    }


def load_iteration_records(path: Path) -> tuple[IterationRecord, ...]:
    """Parses a raw k6 stdout capture (`k6 run ... | tee run.log`): k6 emits plenty of its own
    console/progress noise on stdout, so only lines carrying the `LOAD_ITER ` prefix
    (`k6/chat_sse_load.js`'s own `console.log` convention) are treated as records at all -- an
    ordinary noise line is silently skipped (expected), but a PREFIXED line that fails to parse as
    JSON is a real anomaly and raises, never silently dropped."""
    path = Path(path)
    records: list[IterationRecord] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.startswith(_ITER_LINE_PREFIX):
            continue
        payload = line[len(_ITER_LINE_PREFIX):]
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: malformed LOAD_ITER record: {exc}") from exc
        ttft_ms = data.get("ttft_ms")
        records.append(IterationRecord(
            trace_id=str(data["trace_id"]),
            concurrency=int(data["concurrency"]),
            ttft_ms=float(ttft_ms) if ttft_ms is not None else None,
            tokens_per_sec=float(data.get("tokens_per_sec") or 0.0),
            e2e_ms=float(data["e2e_ms"]),
            goodput=bool(data["goodput"]),
            prompt_id=str(data.get("prompt_id", "")),
        ))
    return tuple(records)


def load_span_export(path: Path) -> tuple[SpanRecord, ...]:
    """Parses a JSON array of spans in `SpanRecord`'s own shape. Turning a REAL Phoenix export into
    this shape is the live/burst wiring step this task documents but does not build (module
    docstring); this loader is what a fixture, or that future adapter's own output, both feed."""
    path = Path(path)
    data = json.loads(path.read_text())
    return tuple(
        SpanRecord(
            span_id=entry["span_id"], trace_id=entry["trace_id"], name=entry["name"],
            attributes=entry.get("attributes") or {},
        )
        for entry in data
    )


__all__ = [
    "STAGE_NAMES",
    "TURN_SEQ_ATTRIBUTE",
    "IterationRecord",
    "JoinMiss",
    "JoinResult",
    "SpanRecord",
    "find_anchor_span",
    "join_iterations_to_spans",
    "load_iteration_records",
    "load_span_export",
    "percentile",
    "stage_duration",
    "stage_latencies_for_real_trace_id",
    "summarize_by_concurrency",
]
