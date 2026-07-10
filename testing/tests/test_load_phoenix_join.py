"""`load.phoenix_join`, hermetic (SP9 task 6): the join against Phoenix spans, run after a k6
iteration capture, over a FIXTURE span set only -- the real load sweep (a real k6 binary, a real
Phoenix instance) is live, burst tier only, deferred, exactly like the rest of SP9's live backlog.
What IS gated here is the join mechanism itself, both directions, mirroring
`test_trace_id_handoff.py::test_atlas_turn_seq_joins_the_envelope_trace_id_to_its_exported_span_both_directions`
one layer up (client visible id -> exported span -> every sibling span in that real trace).
"""
from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from atlas.adapters.otel_tracer import OtelTracer

from load.phoenix_join import (
    STAGE_NAMES,
    TURN_SEQ_ATTRIBUTE,
    IterationRecord,
    JoinMiss,
    SpanRecord,
    find_anchor_span,
    join_iterations_to_spans,
    load_iteration_records,
    load_span_export,
    percentile,
    stage_duration,
    stage_latencies_for_real_trace_id,
    summarize_by_concurrency,
)


# ---- "verify it exists": the join key this whole module leans on is SP6's real one, not a --------
# ---- reinvented literal that could silently drift from what otel_tracer.py actually stamps. ------


def test_the_join_key_constant_matches_what_a_real_otel_tracer_actually_stamps():
    exporter = InMemorySpanExporter()
    tracer = OtelTracer(endpoint="http://example.invalid:4318", config_hash="h", exporter=exporter, clock=lambda: 0.0)
    root = tracer.open("turn", "turn", input="hello", intent="account", customer_id="cust_1")
    span = exporter.get_finished_spans()[0]
    assert TURN_SEQ_ATTRIBUTE in span.attributes
    assert span.attributes[TURN_SEQ_ATTRIBUTE] == str(root)


# ---- fixture span set: two turns' worth of spans, flattened into one export, the way a real ------
# ---- Phoenix export of a whole burst run would mix every turn's spans together. ------------------


def _fixture_spans() -> tuple[SpanRecord, ...]:
    """Turn A (`atlas.turn.seq` anchor "10", real trace id "trace-aaa") ran at concurrency 1 in the
    iteration fixture below; turn B (anchor "55", real trace id "trace-bbb") ran at concurrency 8.
    Every span of a turn shares that turn's own real `trace_id` (Phoenix's native grouping key) but
    carries its OWN, distinct `atlas.turn.seq` value, exactly as `otel_tracer.py` stamps it."""
    return (
        SpanRecord("s1", "trace-aaa", "turn", {TURN_SEQ_ATTRIBUTE: "10"}),
        SpanRecord("s2", "trace-aaa", "agent", {TURN_SEQ_ATTRIBUTE: "11"}),
        SpanRecord("s3", "trace-aaa", "embed", {TURN_SEQ_ATTRIBUTE: "12", "atlas.stage.embed_ms": 40.0}),
        SpanRecord("s4", "trace-aaa", "retrieve", {TURN_SEQ_ATTRIBUTE: "13", "atlas.stage.retrieve_ms": 55.0}),
        SpanRecord("s5", "trace-aaa", "rerank", {TURN_SEQ_ATTRIBUTE: "14", "atlas.stage.rerank_ms": 210.0}),
        SpanRecord("s6", "trace-aaa", "ttft", {TURN_SEQ_ATTRIBUTE: "15", "atlas.stage.ttft_ms": 350.0}),
        SpanRecord("s7", "trace-bbb", "turn", {TURN_SEQ_ATTRIBUTE: "55"}),
        SpanRecord("s8", "trace-bbb", "embed", {TURN_SEQ_ATTRIBUTE: "56", "atlas.stage.embed_ms": 90.0}),
        SpanRecord("s9", "trace-bbb", "retrieve", {TURN_SEQ_ATTRIBUTE: "57", "atlas.stage.retrieve_ms": 130.0}),
        SpanRecord("s10", "trace-bbb", "rerank", {TURN_SEQ_ATTRIBUTE: "58", "atlas.stage.rerank_ms": 480.0}),
        SpanRecord("s11", "trace-bbb", "ttft", {TURN_SEQ_ATTRIBUTE: "59", "atlas.stage.ttft_ms": 820.0}),
    )


# ---- stage_duration: reads the ONE atlas.stage.*_ms attribute a span carries, if any --------------


def test_stage_duration_reads_the_one_matching_stage_attribute():
    span = SpanRecord("s", "t", "rerank", {TURN_SEQ_ATTRIBUTE: "1", "atlas.stage.rerank_ms": 210.0})
    assert stage_duration(span) == ("rerank", 210.0)


def test_stage_duration_is_none_for_a_non_stage_span():
    span = SpanRecord("s", "t", "agent", {TURN_SEQ_ATTRIBUTE: "1"})
    assert stage_duration(span) is None


def test_stage_names_cover_every_stage_this_repo_actually_emits():
    """Cross checked against `trace_translation.STAGE_DURATION_ATTRIBUTE`'s own keys (the real
    emitter), never an independently maintained second list that could silently fall out of sync."""
    from atlas.adapters import trace_translation

    assert set(STAGE_NAMES) == set(trace_translation.STAGE_DURATION_ATTRIBUTE)


# ---- find_anchor_span: direction one, envelope/client id -> the ONE matching exported span -------


def test_find_anchor_span_locates_the_one_span_whose_turn_seq_matches():
    spans = _fixture_spans()
    anchor = find_anchor_span(spans, "10")
    assert anchor is not None
    assert anchor.span_id == "s1"
    assert anchor.trace_id == "trace-aaa"


def test_find_anchor_span_finds_the_second_turns_own_anchor_too():
    spans = _fixture_spans()
    anchor = find_anchor_span(spans, "55")
    assert anchor.span_id == "s7"
    assert anchor.trace_id == "trace-bbb"


def test_find_anchor_span_returns_none_when_no_span_matches():
    spans = _fixture_spans()
    assert find_anchor_span(spans, "does-not-exist") is None


def test_find_anchor_span_returns_none_on_an_ambiguous_duplicate_rather_than_guessing():
    spans = _fixture_spans() + (SpanRecord("dup", "trace-ccc", "turn", {TURN_SEQ_ATTRIBUTE: "10"}),)
    assert find_anchor_span(spans, "10") is None


# ---- stage_latencies_for_real_trace_id: direction two, real trace id -> every stage in that turn -


def test_stage_latencies_for_real_trace_id_collects_only_that_turns_own_stages():
    spans = _fixture_spans()
    latencies = stage_latencies_for_real_trace_id(spans, "trace-aaa")
    assert latencies == {"embed": 40.0, "retrieve": 55.0, "rerank": 210.0, "ttft": 350.0}


def test_stage_latencies_for_real_trace_id_never_leaks_a_sibling_turns_stages():
    spans = _fixture_spans()
    latencies = stage_latencies_for_real_trace_id(spans, "trace-bbb")
    assert latencies["rerank"] == 480.0
    assert "embed" in latencies and latencies["embed"] == 90.0


# ---- join_iterations_to_spans: both directions, wired end to end, grouped by concurrency step -----


def test_join_iterations_to_spans_groups_by_concurrency_step():
    spans = _fixture_spans()
    iterations = (
        IterationRecord("10", concurrency=1, ttft_ms=350.0, tokens_per_sec=12.0, e2e_ms=900.0, goodput=True),
        IterationRecord("55", concurrency=8, ttft_ms=820.0, tokens_per_sec=6.0, e2e_ms=1500.0, goodput=True),
    )
    result = join_iterations_to_spans(iterations, spans)
    assert result.per_concurrency[1]["rerank"] == (210.0,)
    assert result.per_concurrency[8]["rerank"] == (480.0,)
    assert result.misses == ()


def test_join_iterations_to_spans_records_a_miss_never_silently_dropping_it():
    spans = _fixture_spans()
    iterations = (
        IterationRecord("nope", concurrency=1, ttft_ms=None, tokens_per_sec=0.0, e2e_ms=5000.0, goodput=False),
    )
    result = join_iterations_to_spans(iterations, spans)
    assert result.per_concurrency == {}
    assert len(result.misses) == 1
    miss = result.misses[0]
    assert isinstance(miss, JoinMiss)
    assert miss.trace_id == "nope"
    assert miss.concurrency == 1
    assert miss.reason  # never silent: a human readable reason is always present


def test_join_iterations_to_spans_sorts_values_deterministically():
    spans = (
        SpanRecord("a", "t1", "turn", {TURN_SEQ_ATTRIBUTE: "1"}),
        SpanRecord("b", "t1", "rerank", {TURN_SEQ_ATTRIBUTE: "2", "atlas.stage.rerank_ms": 300.0}),
        SpanRecord("c", "t2", "turn", {TURN_SEQ_ATTRIBUTE: "3"}),
        SpanRecord("d", "t2", "rerank", {TURN_SEQ_ATTRIBUTE: "4", "atlas.stage.rerank_ms": 100.0}),
    )
    iterations = (
        IterationRecord("1", concurrency=2, ttft_ms=1.0, tokens_per_sec=1.0, e2e_ms=1.0, goodput=True),
        IterationRecord("3", concurrency=2, ttft_ms=1.0, tokens_per_sec=1.0, e2e_ms=1.0, goodput=True),
    )
    result = join_iterations_to_spans(iterations, spans)
    assert result.per_concurrency[2]["rerank"] == (100.0, 300.0)  # sorted, never insertion order


# ---- percentile: a plain nearest rank quantile, never quality.stats' bootstrap CI machinery -------


def test_percentile_p50_of_five_known_values():
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 50) == 30.0


def test_percentile_p95_of_five_known_values():
    assert percentile([10.0, 20.0, 30.0, 40.0, 50.0], 95) == 50.0


def test_percentile_of_a_single_value_is_that_value():
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 95) == 42.0


def test_percentile_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)


def test_percentile_rejects_an_out_of_range_percentage():
    with pytest.raises(ValueError, match="0, 100"):
        percentile([1.0, 2.0], 150)


# ---- summarize_by_concurrency: the reporting layer over an already joined result ------------------


def test_summarize_by_concurrency_reports_n_p50_p95_per_stage():
    spans = _fixture_spans()
    iterations = (
        IterationRecord("10", concurrency=1, ttft_ms=350.0, tokens_per_sec=12.0, e2e_ms=900.0, goodput=True),
        IterationRecord("55", concurrency=8, ttft_ms=820.0, tokens_per_sec=6.0, e2e_ms=1500.0, goodput=True),
    )
    result = join_iterations_to_spans(iterations, spans)
    summary = summarize_by_concurrency(result)
    assert summary[1]["rerank"] == {"n": 1, "p50": 210.0, "p95": 210.0}
    assert summary[8]["rerank"] == {"n": 1, "p50": 480.0, "p95": 480.0}


# ---- load_iteration_records / load_span_export: file loading halves of the join -------------------


def test_load_iteration_records_skips_ordinary_k6_console_noise(tmp_path):
    path = tmp_path / "k6_stdout.log"
    path.write_text(
        "INFO[0001] some k6 startup banner\n"
        'LOAD_ITER {"trace_id": "10", "concurrency": 1, "ttft_ms": 350.0, '
        '"tokens_per_sec": 12.0, "e2e_ms": 900.0, "goodput": true, "prompt_id": "short-price"}\n'
        "     data_received..........: 12 kB\n"
    )
    records = load_iteration_records(path)
    assert len(records) == 1
    assert records[0] == IterationRecord(
        trace_id="10", concurrency=1, ttft_ms=350.0, tokens_per_sec=12.0, e2e_ms=900.0,
        goodput=True, prompt_id="short-price",
    )


def test_load_iteration_records_raises_on_a_malformed_prefixed_line_never_silently_skipping_it(tmp_path):
    path = tmp_path / "bad.log"
    path.write_text("LOAD_ITER {not valid json\n")
    with pytest.raises(ValueError, match="malformed"):
        load_iteration_records(path)


def test_load_iteration_records_defaults_a_missing_ttft_to_none_for_a_turn_that_never_streamed_a_token(tmp_path):
    path = tmp_path / "no_ttft.log"
    path.write_text(
        'LOAD_ITER {"trace_id": "1", "concurrency": 1, "ttft_ms": null, '
        '"tokens_per_sec": 0.0, "e2e_ms": 5000.0, "goodput": false}\n'
    )
    records = load_iteration_records(path)
    assert records[0].ttft_ms is None
    assert records[0].goodput is False


def test_load_span_export_parses_a_json_array_of_spans(tmp_path):
    path = tmp_path / "spans.json"
    path.write_text(json.dumps([
        {"span_id": "s1", "trace_id": "t1", "name": "turn", "attributes": {TURN_SEQ_ATTRIBUTE: "1"}},
        {"span_id": "s2", "trace_id": "t1", "name": "rerank",
         "attributes": {TURN_SEQ_ATTRIBUTE: "2", "atlas.stage.rerank_ms": 210.0}},
    ]))
    spans = load_span_export(path)
    assert len(spans) == 2
    assert spans[1].name == "rerank"
    assert spans[1].attributes["atlas.stage.rerank_ms"] == 210.0
