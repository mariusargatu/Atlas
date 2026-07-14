"""SP6 task 4: structured JSON logs for the "atlas" logger tree (`atlas.logging_setup`), hermetic.

Four things this file proves, mirroring the plan's own deliverable list:

1. JSON shape: every line `atlas.logging_setup.JsonFormatter` produces is one parseable JSON object
   carrying `ts`/`level`/`logger`/`message`.
2. Correlation: `trace_id`/`span_id`, contextvar carried (`bind_trace_context`), present together
   under an active bind and OMITTED (never `null`) outside one.
3. Redaction: a structured `extra` key not in the SAME allowlist Task 3's collector redaction
   processor uses (`contract_tools.redaction.allowed_attributes()`, imported directly, never a hand
   copied second list) is dropped and named in a `redacted_keys` diagnostic field; an allowed key
   passes through untouched.
4. No regression: `configure_logging()` survives the exact alembic `fileConfig` reproduction order
   bug `backend/atlas/persistence/env.py`'s own `disable_existing_loggers=False` fix guards against
   (SP4), reproduced directly here rather than relying on incidental test file ordering the way the
   original discovery did (`test_persistence.py`'s own comment names the mechanism).

`test_sse_contract.py` carries the end to end wiring proof (the real `/chat/stream` error path
actually calls `bind_trace_context`, not just that the mechanism works in isolation).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from atlas.logging_setup import JsonFormatter, bind_trace_context, configure_logging

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "backend" / "atlas" / "persistence" / "alembic.ini"


@pytest.fixture(autouse=True)
def _restore_atlas_logger():
    """Mirrors `test_chat_app.py`'s own `_restore_atlas_logger`: `configure_logging()` attaches a
    handler to the PROCESS GLOBAL "atlas" logger, so its handlers/level are snapshotted and restored
    around every test in this module, keeping that process global effect test scoped."""
    atlas_log = logging.getLogger("atlas")
    saved_handlers, saved_level, saved_disabled = atlas_log.handlers[:], atlas_log.level, atlas_log.disabled
    try:
        yield
    finally:
        atlas_log.handlers[:] = saved_handlers
        atlas_log.setLevel(saved_level)
        atlas_log.disabled = saved_disabled


def _emit(logger: logging.Logger, stream, *, msg: str, extra: dict | None = None, exc: bool = False) -> dict:
    """Attach a throwaway `JsonFormatter` handler to `logger`, emit exactly one record, detach, and
    return the parsed JSON payload. Isolated per call: never touches the "atlas" logger's own real
    handler (this module's tests exercise `JsonFormatter`/`bind_trace_context` directly, not
    `configure_logging()`'s own attachment, except in the dedicated tests below that call it)."""
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        if exc:
            try:
                raise RuntimeError("boom")
            except RuntimeError:
                logger.exception(msg, extra=extra or {})
        else:
            logger.info(msg, extra=extra or {})
    finally:
        logger.removeHandler(handler)
    line = stream.getvalue().strip().splitlines()[-1]
    return json.loads(line)


# ---- 1. JSON shape ----


def test_json_line_has_ts_level_logger_message(tmp_path):
    import io

    logger = logging.getLogger("atlas.logging_setup.test.shape")
    payload = _emit(logger, io.StringIO(), msg="hello %s", extra=None)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "atlas.logging_setup.test.shape"
    assert payload["message"] == "hello %s"
    assert isinstance(payload["ts"], str) and payload["ts"]  # ISO 8601, not asserted byte exact (wall clock)


def test_json_line_is_valid_json_on_its_own_line():
    import io

    logger = logging.getLogger("atlas.logging_setup.test.oneline")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info("first")
        logger.info("second")
    finally:
        logger.removeHandler(handler)
    lines = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert [json.loads(ln)["message"] for ln in lines] == ["first", "second"]


def test_exception_record_carries_a_formatted_traceback():
    import io

    logger = logging.getLogger("atlas.logging_setup.test.exc")
    payload = _emit(logger, io.StringIO(), msg="failed", exc=True)
    assert "exception" in payload
    assert "RuntimeError" in payload["exception"]
    assert "boom" in payload["exception"]


# ---- 2. correlation: trace_id/span_id, contextvar carried ----


def test_trace_id_and_span_id_present_together_under_an_active_bind():
    import io

    logger = logging.getLogger("atlas.logging_setup.test.bound")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with bind_trace_context("trace-abc", "span-1"):
            logger.info("inside")
    finally:
        logger.removeHandler(handler)
    payload = json.loads(stream.getvalue().strip())
    assert payload["trace_id"] == "trace-abc"
    assert payload["span_id"] == "span-1"


def test_trace_id_and_span_id_absent_not_null_outside_any_bind():
    import io

    logger = logging.getLogger("atlas.logging_setup.test.unbound")
    payload = _emit(logger, io.StringIO(), msg="outside")
    assert "trace_id" not in payload
    assert "span_id" not in payload


def test_trace_id_present_without_span_id_when_span_id_is_none():
    import io

    logger = logging.getLogger("atlas.logging_setup.test.no_span")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with bind_trace_context("trace-only"):
            logger.info("inside")
    finally:
        logger.removeHandler(handler)
    payload = json.loads(stream.getvalue().strip())
    assert payload["trace_id"] == "trace-only"
    assert "span_id" not in payload


def test_bind_trace_context_resets_on_exception_not_just_normal_exit():
    import io

    logger = logging.getLogger("atlas.logging_setup.test.reset_on_exc")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with pytest.raises(ValueError):
            with bind_trace_context("trace-will-not-leak"):
                raise ValueError("boom")
        logger.info("after")
    finally:
        logger.removeHandler(handler)
    payload = json.loads(stream.getvalue().strip())
    assert "trace_id" not in payload


def test_nested_binds_restore_the_outer_context_on_exit():
    import io

    logger = logging.getLogger("atlas.logging_setup.test.nested")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with bind_trace_context("outer", "span-outer"):
            with bind_trace_context("inner", "span-inner"):
                logger.info("nested")
            logger.info("back to outer")
    finally:
        logger.removeHandler(handler)
    lines = [json.loads(ln) for ln in stream.getvalue().splitlines() if ln.strip()]
    assert lines[0]["trace_id"] == "inner" and lines[0]["span_id"] == "span-inner"
    assert lines[1]["trace_id"] == "outer" and lines[1]["span_id"] == "span-outer"


# ---- 3. redaction of structured extras, the SAME allowlist as Task 3's collector ----


def test_disallowed_extra_is_dropped_and_named_in_redacted_keys():
    import io

    logger = logging.getLogger("atlas.logging_setup.test.redact")
    payload = _emit(
        logger, io.StringIO(), msg="turn", extra={"reason": "because", "atlas.guard.decision": "allow"},
    )
    assert "reason" not in payload
    assert payload["atlas.guard.decision"] == "allow"
    assert payload["redacted_keys"] == ["reason"]


def test_allowed_extra_only_never_adds_a_redacted_keys_field():
    import io

    logger = logging.getLogger("atlas.logging_setup.test.no_redact")
    payload = _emit(logger, io.StringIO(), msg="turn", extra={"atlas.guard.decision": "allow"})
    assert payload["atlas.guard.decision"] == "allow"
    assert "redacted_keys" not in payload


def test_multiple_disallowed_extras_are_all_named_sorted():
    import io

    logger = logging.getLogger("atlas.logging_setup.test.redact_many")
    payload = _emit(
        logger, io.StringIO(), msg="turn",
        extra={"zeta_secret": "z", "alpha_secret": "a", "atlas.guard.decision": "allow"},
    )
    assert payload["redacted_keys"] == ["alpha_secret", "zeta_secret"]
    assert "zeta_secret" not in payload and "alpha_secret" not in payload


def test_redaction_allowlist_is_the_same_set_contract_tools_redaction_computes():
    """No hand copied second list (the drift bug this repo's own contract tooling exists to
    catch): `atlas.logging_setup`'s own allowed set must equal `contract_tools.redaction
    .allowed_attributes()`'s current output exactly, proving it was derived from that single
    source, not retyped."""
    from contract_tools.redaction import allowed_attributes

    from atlas import logging_setup

    assert logging_setup._ALLOWED_EXTRA_KEYS == frozenset(allowed_attributes())


def test_every_reserved_trace_attribute_survives_log_redaction_unconditionally():
    from contract_tools.loader import RESERVED_TRACE_ATTRIBUTES

    import io

    logger = logging.getLogger("atlas.logging_setup.test.reserved")
    extra = {name: "x" for name in RESERVED_TRACE_ATTRIBUTES}
    payload = _emit(logger, io.StringIO(), msg="turn", extra=extra)
    for name in RESERVED_TRACE_ATTRIBUTES:
        assert payload[name] == "x"
    assert "redacted_keys" not in payload


# ---- 4. configure_logging(): attachment, idempotency, destination ----


def test_configure_logging_attaches_exactly_one_json_handler_to_atlas_only():
    atlas_log = logging.getLogger("atlas")
    atlas_log.handlers.clear()
    configure_logging()
    assert len(atlas_log.handlers) == 1
    assert isinstance(atlas_log.handlers[0].formatter, JsonFormatter)
    assert atlas_log.level == logging.INFO


def test_configure_logging_is_idempotent_across_repeated_calls():
    atlas_log = logging.getLogger("atlas")
    atlas_log.handlers.clear()
    configure_logging()
    first = atlas_log.handlers[:]
    configure_logging()
    configure_logging()
    assert atlas_log.handlers == first  # never stacks a second handler


def test_configure_logging_pins_stdout_explicitly():
    import sys

    atlas_log = logging.getLogger("atlas")
    atlas_log.handlers.clear()
    configure_logging()
    assert atlas_log.handlers[0].stream is sys.stdout


def test_configure_logging_never_touches_root_or_uvicorn_loggers():
    root = logging.getLogger()
    uvicorn_log = logging.getLogger("uvicorn")
    root_handlers, root_level = root.handlers[:], root.level
    uvicorn_handlers, uvicorn_level = uvicorn_log.handlers[:], uvicorn_log.level

    atlas_log = logging.getLogger("atlas")
    atlas_log.handlers.clear()
    configure_logging()

    assert root.handlers == root_handlers and root.level == root_level
    assert uvicorn_log.handlers == uvicorn_handlers and uvicorn_log.level == uvicorn_level


# ---- 5. no regression: the SP4 alembic fileConfig reproduction order bug ----


def test_configure_logging_survives_the_alembic_fileconfig_reproduction_order_bug(monkeypatch, capsys):
    """`backend/atlas/persistence/env.py` calls `fileConfig(..., disable_existing_loggers=False)`
    specifically because the default (`True`) sets `.disabled = True` on every already existing
    logger not named in `alembic.ini`'s own `[loggers]` section (`root`, `sqlalchemy`, `alembic`
    only) -- "atlas"/"atlas.chat_app" among them, since both exist at plain module import, well
    before alembic's `env.py` ever runs (`test_persistence.py`'s own comment documents the original
    discovery, via incidental alphabetical test order in one pytest process). Reproduced directly
    here: run the SAME alembic invocation `test_persistence.py`'s own
    `test_alembic_env_refuses_to_run_when_atlas_pg_dsn_is_unset` uses (it reaches `fileConfig`
    before its own `ATLAS_PG_DSN` check raises), then prove the "atlas" JSON logging setup this
    task adds is functionally undisturbed -- not merely that its attributes look unchanged, but that
    a log line emitted AFTER still reaches stdout as valid JSON."""
    from alembic import command
    from alembic.config import Config

    atlas_log = logging.getLogger("atlas")
    chat_app_log = logging.getLogger("atlas.chat_app")
    uvicorn_log = logging.getLogger("uvicorn")
    uvicorn_saved_disabled = uvicorn_log.disabled

    atlas_log.handlers.clear()  # force a fresh attachment bound to THIS test's own capsys stream
    configure_logging()

    monkeypatch.delenv("ATLAS_PG_DSN", raising=False)
    cfg = Config(str(ALEMBIC_INI))
    with pytest.raises(RuntimeError, match="ATLAS_PG_DSN"):
        command.upgrade(cfg, "head")

    assert atlas_log.disabled is False
    assert chat_app_log.disabled is False
    assert uvicorn_log.disabled is uvicorn_saved_disabled
    assert len(atlas_log.handlers) == 1
    assert isinstance(atlas_log.handlers[0].formatter, JsonFormatter)

    capsys.readouterr()  # discard alembic's own stderr/stdout chatter from the reproduction above
    logging.getLogger("atlas.chat_app").info("still alive")
    captured = capsys.readouterr()
    lines = [json.loads(ln) for ln in captured.out.splitlines() if ln.strip()]
    assert any(ln["message"] == "still alive" and ln["logger"] == "atlas.chat_app" for ln in lines), (
        f"expected a JSON line from atlas.chat_app on stdout after the alembic reproduction; got {lines}"
    )
