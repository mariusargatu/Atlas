"""Scores are what make a trace searchable by quality, and they must not lie when off.

Nothing here reaches a network.
"""

from __future__ import annotations

import importlib
from dataclasses import fields

import pytest

from atlas.contracts import Record
from evals.results import QuestionRow


def test_the_tracer_is_inert_rather_than_broken_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off must mean off for every entry point, not just the decorator: a client with no
    credentials otherwise prints an authentication error per span.

    `atlas.trace` decides once at import from the environment, so the keys are cleared and
    the module reimported rather than read as they happen to be. Without that, this passes
    or fails on whether the developer's `.env` has Langfuse keys and never exercises the
    branch it names.
    """
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    trace = importlib.reload(importlib.import_module("atlas.trace"))
    try:
        def unchanged(x: int) -> int:
            return x

        assert trace.observe(unchanged) is unchanged
        assert trace.current_trace_id() is None
        assert trace.flush() is None
        assert trace.score("any-trace", "ndcg@k", 0.42) is None
        assert trace.score("any-trace", "judge", "pass", "because") is None
    finally:
        # Restore, so a reloaded module does not leave every later import holding the stubs.
        monkeypatch.undo()
        importlib.reload(trace)


def test_a_record_carries_the_trace_it_was_written_to() -> None:
    # The field has to exist even with tracing off: nothing else in the repository joins a
    # measurement computed after the pipeline finished back to its request.
    names = {f.name for f in fields(Record)}
    assert "trace_id" in names, (
        "Record no longer carries a trace id, so a score computed after the span closed "
        "has no way to reach the request it describes"
    )


def test_the_question_ledger_carries_the_trace_too() -> None:
    # The runner scores from the ledger rather than from memory, so the id has to survive
    # the round trip to disk. What it does not buy is the right to reuse it later; see the
    # test on resumed rows below.
    assert "trace_id" in {f.name for f in fields(QuestionRow)}


def test_a_resumed_question_is_not_scored_against_the_trace_it_had_last_time() -> None:
    """A trace id is a key into one Langfuse instance, and the ledger outlives it.

    Langfuse accepts a score against a trace id no run ever opened: no error, queryable by
    that id, and fetching the trace itself is a 404. So a resumed run reporting "scores
    sent for 3 of 3" is silent on both sides. A score goes only to a trace this process
    opened, because that is the only claim available without a lookup that can go stale.
    """
    # Imported locally: `scripts.summarise` imports `atlas.trace`, which decides whether
    # tracing is on at import, and the first test in this file reloads that module.
    from scripts.summarise import scoreable_traces

    def row(name: str, trace: str) -> QuestionRow:
        return QuestionRow(
            schema="1.0.0", run_id="r", question=name, generated=False, judged=False,
            rankings={"reranked": ()}, correct=(), primary=(), timings_ms={},
            trace_id=trace,
        )

    rows = (row("fresh", "trace-from-this-process"), row("stale", "trace-from-a-dead-server"))

    traced, declined = scoreable_traces(rows, frozenset({"trace-from-this-process"}))

    assert set(traced) == {"fresh"}, (
        f"a trace this process did not open was offered up for scoring: {sorted(traced)}"
    )
    assert declined == 1, (
        "the skipped question was not counted, so the run would report a smaller number "
        "with no explanation rather than saying what it declined to do"
    )


def test_every_stage_of_one_question_lands_in_one_trace() -> None:
    """`observe` starts a new trace for every decorated call with no active parent, so
    without a root span each stage becomes its own trace, `current_trace_id()` returns None
    after they close, and a fully configured run publishes zero scores while printing
    exactly what an unconfigured run prints.

    Run against a real client with placeholder credentials and an in-memory OTel exporter,
    so this asserts the tracer's behaviour rather than a mock's. The exporter never reaches
    the base URL.
    """
    import langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    client = langfuse.Langfuse(
        public_key="pk-lf-placeholder", secret_key="sk-lf-placeholder",
        base_url="http://127.0.0.1:59999", span_exporter=exporter,
    )

    def stage(x: int) -> int:
        return x

    @langfuse.observe(as_type="chain")
    def one_question() -> str | None:
        langfuse.observe(as_type="retriever")(stage)(1)
        langfuse.observe(as_type="generation")(stage)(2)
        return client.get_current_trace_id()

    trace_id = one_question()
    client.flush()
    spans = exporter.get_finished_spans()

    assert len(spans) == 3, f"expected a root and two stages, got {len(spans)}"
    assert len({s.context.trace_id for s in spans}) == 1, (
        "the stages of one question landed in more than one trace, so a score attached "
        "after the fact has no single request to attach to"
    )
    assert trace_id is not None, "no trace id was readable from inside the root span"
    assert sum(1 for s in spans if s.parent is None) == 1, "more than one root span"
