"""SP6 task 4: structured JSON logs for the "atlas" logger tree, correlated with a turn's trace id
(and, where a real span is known, span id) via a contextvar carried across the log call, never a
second id minted here -- `chat_app.py`'s stream error path is the ONE production caller, and it
reuses Task 2's already resolved `trace_id`/`ttft_seq` verbatim (see `bind_trace_context`'s own
docstring).

Extends, in place, the "atlas" logger tree attachment `server.py`'s own `_configure_logging` used to
own directly (SP3/SP4): ONE `StreamHandler` on the "atlas" logger only (never root, never a child
logger directly -- every "atlas.*" child, `atlas.chat_app` among them, reaches this one handler by
plain propagation, unchanged), idempotent across a repeated `create_app()` call or a repeated
module import.
`configure_logging()` below IS that same attachment point; the only change is what the handler
formats each record INTO (one JSON object per line, not a plain text line) and WHERE it writes
(`sys.stdout`, pinned explicitly -- the prior bare `StreamHandler()` default was `sys.stderr`, a
destination this task deliberately corrects to match the plan's own "JSON lines to stdout" wording,
not merely a reformat).

Correlation (`trace_id`/`span_id`): two `contextvars.ContextVar`s, bound for the duration of one
`with bind_trace_context(...):` block by whichever call site already knows the turn's own id. Absent
fields are OMITTED from the JSON object, never emitted as `null`: a log line produced with no bound
context (server startup, any log call outside a turn, every "atlas" line from a process that never
threads a trace at all) carries neither key at all -- the same "absent, not nulled" convention
`otel_tracer.py`'s own `_coerce_attr` and the SSE `citation` event's optional `entity_ids` field
already use in this codebase.

Redaction of structured extras: any keyword passed via `extra={...}` to a log call becomes a plain
attribute on the `LogRecord` (Python's own `logging` mechanism); this module walks every attribute
NOT part of `logging.LogRecord`'s own standard set and keeps only the ones in the SAME allowlist
Task 3's collector redaction processor enforces (`contract_tools.redaction.allowed_attributes()`,
imported directly and computed ONCE at import time -- never a second, hand copied list, the exact
drift bug this repo's own contract tooling exists to catch; see `testing/tests/test_logging_setup.py
::test_redaction_allowlist_is_the_same_set_contract_tools_redaction_computes`). A dropped key is
named in a `redacted_keys` diagnostic field (mirroring the real OTel Collector redaction processor's
own `redaction.redacted.keys` diagnostic, confirmed live in Task 3's own verification), never
silently vanished with no trace it was ever there.

`ts` is real wall clock (`record.created`, ISO 8601 UTC), a deliberate, disclosed exception to this
repo's "no wall clock in runtime paths" determinism contract: a log line's own operational timestamp
is never asserted for an exact value by any hermetic test (unlike a trace id, a stage duration, or a
cassette key), so it carries none of the reproducibility risk those do.

D13, explicitly: NO log aggregation stack (Loki/ELK/Datadog/etc.) is added or planned here.
Structured JSON to stdout is the full extent of this task's own logging surface; an operator wanting
log shipping attaches their own sidecar/agent to the container's stdout, outside this repo's scope.
Refused by design, not deferred -- see the SP6 plan's own "Deferred" section.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from contract_tools.redaction import allowed_attributes

_ATLAS_LOGGER_NAME = "atlas"

# The turn correlation pair (Task 2's own trace_id, and where known, the currently relevant span's
# id): `None` (absent) outside any `bind_trace_context` block, the default every ContextVar carries
# until explicitly set. Never read directly outside this module; `bind_trace_context` is the one
# writer, `JsonFormatter.format` is the one reader.
_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("atlas_trace_id", default=None)
_span_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("atlas_span_id", default=None)

# `logging.makeLogRecord({})` is a real `LogRecord` built by Python's own machinery -- reading its
# `__dict__` keys (rather than hand typing "name"/"msg"/"args"/... and risking a stale list across a
# Python version bump) is the same "derive, don't hand copy" discipline this module's own redaction
# import follows. "message"/"asctime" are added because `Formatter` itself sets them (via
# `getMessage()`/`%(asctime)s`) even though a bare `LogRecord` never carries either at construction.
_STANDARD_LOG_RECORD_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}

# Computed ONCE, at import time, from the SAME generator Task 3's collector config uses -- never
# recomputed per log line (`emitted_gen_ai_attributes` re reads and re translates the committed span
# inventory on every call; a hot logging path calling it per record would be real, avoidable I/O on
# every single "atlas" log line). Mirrors `trace_translation.py`'s own `_INVENTORY`/
# `_CONTRACT_TRACE_VERSION` pattern: a frozen module level constant, computed eagerly, never mutated.
_ALLOWED_EXTRA_KEYS: frozenset[str] = frozenset(allowed_attributes())


@contextmanager
def bind_trace_context(trace_id: str, span_id: str | None = None) -> Iterator[None]:
    """Binds `trace_id` (required) and `span_id` (optional) onto every JSON log line formatted
    inside this `with` block, on THIS async task/thread only (`contextvars`' own isolation), reset
    to absent on exit regardless of how the block exits (normal return or an exception -- the exact
    case the one production caller, `chat_app.py`'s stream error path, needs: the log call itself
    never raises, but the surrounding generator frame keeps running after it).

    Never mints a new id: `trace_id` is whatever the caller already resolved (Task 2's
    `_resolve_trace_id` -- the tracer's real turn root id, or the demoted `IdFactory` fallback under
    the hermetic `NullTracer` default); `span_id` is left `None` (absent) by that same caller for
    `NullTracer`'s `-1` sentinel (no real span exists to name), and passed as the string form of a
    real, non negative span sequence otherwise."""
    trace_token = _trace_id_var.set(trace_id)
    span_token = _span_id_var.set(span_id)
    try:
        yield
    finally:
        _trace_id_var.reset(trace_token)
        _span_id_var.reset(span_token)


def _structured_extras(record: logging.LogRecord) -> tuple[dict[str, object], list[str]]:
    """Every attribute a caller attached via `extra={...}` (i.e. everything on the record that is
    NOT one of `logging.LogRecord`'s own standard attributes), split into (kept, redacted) by
    `_ALLOWED_EXTRA_KEYS`. `redacted` is sorted so the diagnostic field this feeds is byte stable,
    never dependent on a plain dict's own (insertion ordered but here irrelevant) iteration."""
    kept: dict[str, object] = {}
    redacted: list[str] = []
    for key, value in record.__dict__.items():
        if key in _STANDARD_LOG_RECORD_ATTRS:
            continue
        if key in _ALLOWED_EXTRA_KEYS:
            kept[key] = value
        else:
            redacted.append(key)
    return kept, sorted(redacted)


class JsonFormatter(logging.Formatter):
    """One JSON object per log line: `ts`/`level`/`logger`/`message` always present;
    `trace_id`/`span_id` present only under an active `bind_trace_context` block (never `null`);
    `redacted_keys` present only when a structured extra was actually dropped; every other allowed
    structured extra merged in verbatim. See this module's own docstring for the full design."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = _trace_id_var.get()
        extras, redacted_keys = _structured_extras(record)
        if redacted_keys:
            payload["redacted_keys"] = redacted_keys
        payload.update(extras)
        # Context derived ids land AFTER the extras merge so no allowlisted extra can ever shadow
        # them: the protection is structural (ordering), not contingent on the allowlist contents.
        if trace_id is not None:
            payload["trace_id"] = trace_id
        span_id = _span_id_var.get()
        if span_id is not None:
            payload["span_id"] = span_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging() -> None:
    """Idempotent (a second call, or a second `create_app()`/import, is a no op): attaches ONE JSON
    `StreamHandler` to the "atlas" logger (never root -- uvicorn's/alembic's own logging config
    stays completely untouched; see `testing/tests/test_logging_setup.py`'s own reproduction of the
    SP4 alembic `fileConfig` order bug for the regression this must never reintroduce). Pinned to
    `sys.stdout` explicitly -- see this module's own docstring for why that is a deliberate
    correction, not merely a format change."""
    atlas_log = logging.getLogger(_ATLAS_LOGGER_NAME)
    if atlas_log.handlers:
        return
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    atlas_log.addHandler(handler)
    atlas_log.setLevel(logging.INFO)


__all__ = ["JsonFormatter", "bind_trace_context", "configure_logging"]
