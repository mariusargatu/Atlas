"""`atlas.metrics` (SP6 task 5): the hand written Prometheus text exposition module.

`prometheus_client` is not a transitive dependency anywhere in `uv.lock` today (checked directly
before writing this module: `grep prometheus uv.lock` finds nothing), so this is a plain, framework
free formatter over a tiny in process registry, mirroring `structured logs`'/`resilience.py`'s own
"module level mutable state, documented as the one boundary" precedent (`domain/accounts.py:_STATE`
is the original instance of this pattern in this codebase).

Four families, one test class each below: a coarse HTTP status class counter (what the D29 "error
rate" alert reads), a circuit breaker state gauge (the controller adjudication landing the resilience
seam's breaker half at the metrics layer, per `adapters/trace_translation.py`'s own docstring), a
corpus staleness gauge (D29's "the only drift signal"), and the judge counter pair (registered here,
zero writers, the seam SP7 fills in).
"""
from __future__ import annotations

import json

from atlas import metrics
from atlas.adapters.resilience import CircuitBreaker

# conftest.py's own `_reset_metrics_counters` autouse fixture resets `atlas.metrics`'s process
# lifetime request counter before and after every test in this suite; no local fixture needed here.

# ---- atlas_http_requests_total: the coarse status class counter -----------------------------------


def test_status_class_buckets_by_hundreds_digit():
    assert metrics.status_class(200) == "2xx"
    assert metrics.status_class(201) == "2xx"
    assert metrics.status_class(301) == "3xx"
    assert metrics.status_class(404) == "4xx"
    assert metrics.status_class(503) == "5xx"


def test_record_request_accumulates_per_status_class():
    metrics.record_request(200)
    metrics.record_request(200)
    metrics.record_request(500)
    body = metrics.render()
    assert 'atlas_http_requests_total{status_class="2xx"} 2' in body
    assert 'atlas_http_requests_total{status_class="5xx"} 1' in body
    assert 'atlas_http_requests_total{status_class="4xx"} 0' in body


def test_render_declares_help_and_type_for_the_request_counter():
    body = metrics.render()
    assert "# TYPE atlas_http_requests_total counter" in body
    assert "# HELP atlas_http_requests_total" in body


def test_reset_request_counts_clears_accumulated_state():
    metrics.record_request(200)
    metrics.reset_request_counts()
    body = metrics.render()
    assert 'atlas_http_requests_total{status_class="2xx"} 0' in body


# ---- atlas_circuit_breaker_state: read fresh from CircuitBreaker.state(), never cached ------------


def test_breaker_gauge_absent_when_no_breaker_is_supplied():
    """The hermetic default (InMemoryRetriever) has no breaker at all -- absence here, not a row of
    zeroes for a resilience mechanism that does not exist in this configuration."""
    body = metrics.render(breaker=None)
    assert "atlas_circuit_breaker_state{" not in body


def test_breaker_gauge_reports_closed_as_zero_for_every_known_provider_key():
    clock = iter([0.0] * 100).__next__
    breaker = CircuitBreaker(clock)
    body = metrics.render(breaker=breaker)
    for key in metrics.KNOWN_PROVIDER_KEYS:
        assert f'atlas_circuit_breaker_state{{provider_key="{key}",state="closed"}} 0' in body


def test_breaker_gauge_reports_open_as_two_after_the_threshold_trips():
    ticks = iter([0.0, 0.0, 0.0, 0.0, 0.0])
    breaker = CircuitBreaker(lambda: next(ticks))
    for _ in range(3):  # _FAILURE_THRESHOLD in resilience.py
        breaker.record_failure("tei-embed")
    body = metrics.render(breaker=breaker)
    assert 'atlas_circuit_breaker_state{provider_key="tei-embed",state="open"} 2' in body
    # untouched provider keys stay closed, read fresh, not left stale from a previous scrape
    assert 'atlas_circuit_breaker_state{provider_key="tei-rerank",state="closed"} 0' in body


def test_breaker_gauge_is_read_fresh_not_cached_across_two_renders():
    """No caching anywhere in this module: a state transition between two scrapes must show up on
    the very next render, proving `breaker.state(...)` is called at render time, not memoized."""
    clock = iter([0.0] * 100).__next__
    breaker = CircuitBreaker(clock)
    before = metrics.render(breaker=breaker)
    assert 'state="closed"' in before
    for _ in range(3):
        breaker.record_failure("postgres")
    after = metrics.render(breaker=breaker)
    assert 'atlas_circuit_breaker_state{provider_key="postgres",state="open"} 2' in after


# ---- atlas_corpus_staleness: registry_version vs the active index's own corpus_version ------------


def test_staleness_is_zero_when_registry_version_is_unset():
    """Absence of a configured comparison is never treated as drift (a discipline adjacent to the
    Global Constraints this module holds itself to: no false alarms from an unset value)."""
    body = metrics.render(registry_version="", index_dir="/does/not/matter")
    assert "atlas_corpus_staleness 0" in body


def test_staleness_is_zero_when_registry_version_matches_the_build_manifest(tmp_path):
    manifest = tmp_path / "build_manifest.json"
    manifest.write_text(json.dumps({"corpus_version": "corpus-0.1.1"}))
    body = metrics.render(registry_version="corpus-0.1.1", index_dir=str(tmp_path))
    assert "atlas_corpus_staleness 0" in body


def test_staleness_is_one_when_registry_version_disagrees_with_the_build_manifest(tmp_path):
    manifest = tmp_path / "build_manifest.json"
    manifest.write_text(json.dumps({"corpus_version": "corpus-0.1.0"}))
    body = metrics.render(registry_version="corpus-0.1.1", index_dir=str(tmp_path))
    assert "atlas_corpus_staleness 1" in body


def test_staleness_is_zero_when_the_build_manifest_is_missing(tmp_path):
    """No index dir on disk (a fresh checkout with no committed build, or a misconfigured path) is
    an honest "cannot tell" case, not a manufactured drift signal."""
    body = metrics.render(registry_version="corpus-0.1.1", index_dir=str(tmp_path / "nope"))
    assert "atlas_corpus_staleness 0" in body


def test_render_declares_help_and_type_for_the_staleness_gauge():
    body = metrics.render()
    assert "# TYPE atlas_corpus_staleness gauge" in body
    assert "# HELP atlas_corpus_staleness" in body


# ---- atlas_judge_pass_total / atlas_judge_fail_total: SP8 Task 4 remainder wires the writers -------


def test_judge_counters_start_at_zero():
    body = metrics.render()
    assert "atlas_judge_pass_total 0" in body
    assert "atlas_judge_fail_total 0" in body


def test_judge_counters_declare_help_and_type():
    body = metrics.render()
    assert "# TYPE atlas_judge_pass_total counter" in body
    assert "# TYPE atlas_judge_fail_total counter" in body


def test_record_judge_pass_increments_only_the_pass_counter():
    metrics.record_judge_pass()
    metrics.record_judge_pass()
    body = metrics.render()
    assert "atlas_judge_pass_total 2" in body
    assert "atlas_judge_fail_total 0" in body


def test_record_judge_fail_increments_only_the_fail_counter():
    metrics.record_judge_fail()
    body = metrics.render()
    assert "atlas_judge_fail_total 1" in body
    assert "atlas_judge_pass_total 0" in body


def test_reset_judge_counts_clears_both_counters():
    metrics.record_judge_pass()
    metrics.record_judge_fail()
    metrics.reset_judge_counts()
    body = metrics.render()
    assert "atlas_judge_pass_total 0" in body
    assert "atlas_judge_fail_total 0" in body
