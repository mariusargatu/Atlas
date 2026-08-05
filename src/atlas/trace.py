"""Tracing, timing, and the one line a run prints about what it spent. Tracing is the
only genuinely optional thing in this package: a reader with no Langfuse credentials
loses the trace view and nothing else."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from types import ModuleType
from typing import Literal

from atlas.contracts import Record, StageTiming

SpanKind = Literal["span", "generation", "embedding", "retriever", "evaluator", "tool", "chain"]


def _tracer() -> ModuleType | None:
    """The langfuse module when this machine can actually trace, None otherwise. Checks
    credentials, not just the package: without them the client prints an authentication
    error per span rather than failing quietly. Read from the environment rather than
    passed in, since tracing is operational and stays out of `run_id`."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    try:
        import langfuse
    except Exception:
        return None
    return langfuse


# Published so a caller needing to say "this needs tracing and you have none" reads the
# outcome rather than re-deriving it from the environment (the import can still fail
# with both keys set).
_LANGFUSE = _tracer()
ENABLED = _LANGFUSE is not None


def observe[F: Callable[..., object]](
    func: F, kind: SpanKind = "span", name: str | None = None
) -> F:
    """Wraps one callable in a span, and returns it untouched when tracing is off."""
    if _LANGFUSE is None:
        return func
    # Two-step form, not observe(func, as_type=...): the library's overloads make
    # as_type keyword-only on the decorator factory shape.
    wrapped: F = _LANGFUSE.observe(as_type=kind, name=name)(func)
    return wrapped


def flush() -> None:
    """Sends what the client has buffered; a process that exits without this loses the
    traces it paid to produce."""
    if _LANGFUSE is not None:
        _LANGFUSE.get_client().flush()


def current_trace_id() -> str | None:
    """The id of the trace being written right now, so a score computed later can be
    attached to it. None when tracing is off. Returned rather than stored: the pipeline
    records it on the Record and everything downstream reads it from there."""
    if _LANGFUSE is None:
        return None
    identifier: str | None = _LANGFUSE.get_client().get_current_trace_id()
    return identifier


def score(trace_id: str, name: str, value: float | str, comment: str = "") -> None:
    """Attach one measurement to one trace, so a trace view is searchable by quality
    (nDCG, judge verdict) and not just by what `observe` recorded. Against the trace id
    rather than the current span, since these numbers are only known after the pipeline
    has finished and the span has closed."""
    if _LANGFUSE is None:
        return
    _LANGFUSE.get_client().create_score(
        trace_id=trace_id, name=name, value=value, comment=comment or None
    )


@contextmanager
def stage_timer(stage: str, timings: list[StageTiming]) -> Generator[None]:
    """Records a StageTiming from two perf_counter readings inside a finally block, so a
    stage that raises still records a time."""
    started = time.perf_counter()
    try:
        yield
    finally:
        timings.append(
            StageTiming(stage=stage, wall_ms=(time.perf_counter() - started) * 1000.0)
        )


def run_summary(
    elapsed_seconds: float, records: Sequence[Record], embedding_cost_usd: float = 0.0
) -> str:
    """One line naming a run's wall clock and what it spent, read back from every Usage
    the run recorded rather than tracked a second time. `embedding_cost_usd` is an
    argument because embedding happens once per corpus, not per question, so it leaves
    no Usage on any Record."""
    cost = embedding_cost_usd
    cost += sum(r.answer.usage.cost_usd for r in records if r.answer is not None)
    cost += sum(r.judge.usage.cost_usd for r in records if r.judge is not None)
    line = f"run: {elapsed_seconds:.1f}s wall clock, ${cost:.4f} spent"
    print(line)
    return line
