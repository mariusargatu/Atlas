"""Prometheus text exposition (SP6 task 5): `server.py`'s own `GET /metrics` route is the only
caller, rendering the string this module builds directly, media type `text/plain; version=0.0.4`
(the Prometheus exposition format's own documented content type).

`prometheus_client` is not a transitive dependency anywhere in `uv.lock` today (checked before
writing this module, the same floor discipline Task 1's OTel pins document: a new direct dependency
is not free); the surface this task needs is four small metric families, small enough that a hand
written, framework free formatter is the honest choice rather than pulling in a client library for a
handful of `# HELP`/`# TYPE` lines. This module holds NO framework import (no fastapi/starlette),
mirroring `contract_tools`'/`trace_translation.py`'s own "pure, tested, no framework" style.

Four families:

  - `atlas_http_requests_total{status_class}`: a COARSE (2xx/3xx/4xx/5xx) request counter,
    incremented by `server.py`'s own middleware via `record_request(status_code)`. Coarse on
    purpose: a per path/per method label set would blow up cardinality for no alerting value D29's
    "error rate" rule actually needs (`sum(rate(...{status_class="5xx"}[15m])) /
    sum(rate(...[15m])) > 0.05`, the exact figure HLD section 5.6 names). State lives in one module
    level `Counter`, the same "one mutable boundary, documented as such" pattern
    `domain/accounts.py:_STATE` already established in this codebase; process lifetime, reset only
    by `reset_request_counts()` (test only) or a process restart.

  - `atlas_circuit_breaker_state{provider_key,state}`: a gauge read FRESH on every call to
    `render()` from `CircuitBreaker.state(provider_key)`
    (`backend/atlas/adapters/resilience.py`), never cached, never re derived -- the controller
    adjudication landing the resilience seam's breaker half at the metrics layer, not the trace
    layer (`adapters/trace_translation.py`'s own module docstring names this exact decision and
    this exact task). 0 closed, 1 half_open, 2 open, so a numeric threshold alert (`> 1`, "open") and
    a human readable `state` label both work off the same time series. `render()` takes the caller's
    `CircuitBreaker` instance as a parameter (never constructs or imports one itself): `server.py`
    passes `getattr(retriever, "breaker", None)`, `None` when the active retriever has no breaker at
    all (the hermetic default, `InMemoryRetriever`) -- absence there is correct, not a row of
    zeroes for a resilience mechanism this configuration does not have.

  - `atlas_corpus_staleness`: a 0/1 gauge, `AtlasSettings.registry_version` vs the active index's
    own `build_manifest.json:corpus_version` (D29: "the registry_version vs ingested corpus_version
    staleness gauge... the only drift signal"). 0 when `registry_version` is unset (no comparison
    configured) or the manifest cannot be read -- absence of a signal is never treated as drift, the
    same "no false alarms from an unset value" discipline `config_hash()`'s own secret/identity
    exclusion holds itself to.

  - `atlas_judge_pass_total` / `atlas_judge_fail_total`: incremented by `record_judge_pass()`/
    `record_judge_fail()` (SP8 Task 4 remainder, correcting this docstring's own former "SP7 writes
    the calibrated judge" label -- the split that created SP8 predates this file's own last edit,
    ADR-029's own rule that owner changes travel with the emitter). The one call site is `judge.
    emission.emit_verdict` (`testing/harness/judge/emission.py`), the SINGLE place a computed judge
    verdict crosses the trace boundary: a grounded verdict increments the pass counter, an
    ungrounded verdict the fail counter, a thin call right beside the trace attribute emission it
    already performs, never a second independent counting mechanism. Process lifetime state, the
    same "one mutable boundary" discipline `_REQUEST_COUNTS` above already holds itself to;
    `reset_judge_counts()` is test only, mirroring `reset_request_counts()`.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# The mutable boundary (mirrors domain/accounts.py's own single mutable store, `_STATE`): the ONLY
# place this module holds live state. Process lifetime counters, not per request/per context (unlike
# resilience.py's `_last_attempt_count`/pgvector_retriever.py's `_search_result`, both contextvar
# scoped for a different reason -- those answer "what did THIS call just do," this answers "how many
# requests has this PROCESS served").
_REQUEST_COUNTS: Counter[str] = Counter()

# The second mutable boundary this module holds: judge verdict counts, keyed "pass"/"fail" (the same
# vocabulary `atlas.adapters.label_store.LabelRecord.verdict` uses for a HUMAN label, distinct from
# the judge's own wire vocabulary "grounded"/"ungrounded" -- `judge.emission.emit_verdict`, the one
# writer, translates at its own call site so this module never has to know the judge's vocabulary).
_JUDGE_VERDICT_COUNTS: Counter[str] = Counter()

_STATUS_CLASSES: tuple[str, ...] = ("2xx", "3xx", "4xx", "5xx")

# CircuitBreaker.state()'s own three names (resilience.py), in the fixed numeric order this module
# exposes them under.
_BREAKER_STATE_VALUE: dict[str, int] = {"closed": 0, "half_open": 1, "open": 2}

# The provider keys resilience.py's own call sites use today (pgvector_retriever.py: "tei-embed",
# "tei-rerank", "postgres"), named here once rather than only discovered by inspecting a live
# breaker's internal dict -- a provider that has never failed has no entry there at all
# (`CircuitBreaker.state()`'s own "closed" default already handles that), so this list exists only
# so a NEVER FAILED provider still gets a `0` row on every scrape instead of silently having no time
# series until its first failure (a gap Prometheus's own `absent()`/alerting story handles badly).
KNOWN_PROVIDER_KEYS: tuple[str, ...] = ("tei-embed", "tei-rerank", "postgres")


def status_class(status_code: int) -> str:
    """`200` -> `"2xx"`, `404` -> `"4xx"`, and so on: the hundreds digit, the coarse bucket
    `atlas_http_requests_total` groups by."""
    return f"{status_code // 100}xx"


def record_request(status_code: int) -> None:
    """`server.py`'s own middleware calls this once per completed (or exception raising, counted as
    5xx there) request. The only writer `_REQUEST_COUNTS` ever has."""
    _REQUEST_COUNTS[status_class(status_code)] += 1


def reset_request_counts() -> None:
    """Test only: process lifetime counters otherwise never reset, so a test suite that boots
    `create_app()` more than once in one process (as `test_chat_app.py`'s own `_server_app` helper
    already does, repeatedly) needs an explicit reset between tests to stay isolated."""
    _REQUEST_COUNTS.clear()


def record_judge_pass() -> None:
    """The judge emitted a grounded verdict. `judge.emission.emit_verdict` (`testing/harness/judge/
    emission.py`) is the one call site: the single place a computed verdict crosses the trace
    boundary, per D29's batch teardown stage design. Mirrors `record_request`'s own shape: one
    function, one job, called once per real verdict."""
    _JUDGE_VERDICT_COUNTS["pass"] += 1


def record_judge_fail() -> None:
    """The judge emitted an ungrounded verdict. Mirrors `record_judge_pass`, same one call site."""
    _JUDGE_VERDICT_COUNTS["fail"] += 1


def reset_judge_counts() -> None:
    """Test only: mirrors `reset_request_counts`, the same process lifetime isolation need between
    tests that call `record_judge_pass`/`record_judge_fail` (directly, or through `judge.emission.
    emit_verdict`) more than once in one process."""
    _JUDGE_VERDICT_COUNTS.clear()


def _corpus_staleness(registry_version: str, index_dir: str) -> float:
    if not registry_version:
        return 0.0
    manifest_path = Path(index_dir) / "build_manifest.json"
    if not manifest_path.is_file():
        return 0.0
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return 0.0
    return 0.0 if manifest.get("corpus_version") == registry_version else 1.0


def render(*, breaker=None, registry_version: str = "", index_dir: str = "") -> str:
    """Build the full exposition body. `breaker`/`registry_version`/`index_dir` are read fresh on
    every call (no caching anywhere in this module): `server.py`'s `/metrics` route passes whatever
    is current at scrape time, so a state transition between two scrapes always shows up on the very
    next one."""
    lines: list[str] = []

    lines.append("# HELP atlas_http_requests_total Total HTTP requests served, coarse status class only.")
    lines.append("# TYPE atlas_http_requests_total counter")
    for cls in _STATUS_CLASSES:
        lines.append(f'atlas_http_requests_total{{status_class="{cls}"}} {_REQUEST_COUNTS.get(cls, 0)}')

    lines.append(
        "# HELP atlas_circuit_breaker_state CircuitBreaker.state() per provider key: "
        "0 closed, 1 half_open, 2 open."
    )
    lines.append("# TYPE atlas_circuit_breaker_state gauge")
    if breaker is not None:
        for key in KNOWN_PROVIDER_KEYS:
            state = breaker.state(key)
            value = _BREAKER_STATE_VALUE.get(state, 0)
            lines.append(f'atlas_circuit_breaker_state{{provider_key="{key}",state="{state}"}} {value}')

    lines.append(
        "# HELP atlas_corpus_staleness registry_version vs the active index's corpus_version "
        "(D29's drift signal): 0 fresh or unconfigured, 1 stale."
    )
    lines.append("# TYPE atlas_corpus_staleness gauge")
    staleness = _corpus_staleness(registry_version, index_dir)
    lines.append(f"atlas_corpus_staleness {staleness:.0f}")

    lines.append(
        "# HELP atlas_judge_pass_total Judge grounded verdicts, incremented by record_judge_pass "
        "(judge.emission.emit_verdict's own call site)."
    )
    lines.append("# TYPE atlas_judge_pass_total counter")
    lines.append(f"atlas_judge_pass_total {_JUDGE_VERDICT_COUNTS.get('pass', 0)}")
    lines.append(
        "# HELP atlas_judge_fail_total Judge ungrounded verdicts, incremented by record_judge_fail "
        "(judge.emission.emit_verdict's own call site)."
    )
    lines.append("# TYPE atlas_judge_fail_total counter")
    lines.append(f"atlas_judge_fail_total {_JUDGE_VERDICT_COUNTS.get('fail', 0)}")

    return "\n".join(lines) + "\n"


__all__ = [
    "KNOWN_PROVIDER_KEYS",
    "record_judge_fail",
    "record_judge_pass",
    "record_request",
    "render",
    "reset_judge_counts",
    "reset_request_counts",
    "status_class",
]
