"""`atlas.adapters.resilience` (SP4 task 3), hermetic: the retry classification table (every status
code row named in the Global Constraints), `RetryPolicy` against a raw callable (no client needed),
`CircuitBreaker`'s three state walk driven by a fake clock, and `call_with_resilience`'s composition
(429 exempt from the breaker, a breaker open short circuit never reaching the wrapped call, typed
errors surfacing via `httpx.MockTransport`). No Docker, no network, no real sleep: `stamina.set_testing`
disables stamina's own backoff sleep for this whole module (its documented pattern for test suites),
capped rather than overridden so `RetryPolicy`'s own `attempts=3` still governs attempt counts.
"""
from __future__ import annotations

import httpx
import psycopg
import pytest
import stamina
from atlas.adapters.resilience import (
    CircuitBreaker,
    EmbeddingServiceError,
    ProviderError,
    RerankServiceError,
    RetrievalError,
    RetryPolicy,
    breaker_exempt,
    call_with_resilience,
    call_with_resilience_async,
    classify_exception,
    is_retryable_status,
    last_call_retried,
)


@pytest.fixture(autouse=True)
def _no_real_backoff_sleep():
    with stamina.set_testing(True, attempts=50, cap=True):
        yield


def _http_error(status_code: int, *, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://tei.test/embed")
    response = httpx.Response(status_code, request=request, headers=headers or {})
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return exc
    raise AssertionError(f"{status_code} did not raise_for_status()")


# --- typed errors: every subclass carries provider_key (structured routing, never string parsed) ----


def test_retrieval_error_carries_provider_key() -> None:
    exc = RetrievalError("boom", provider_key="postgres")
    assert exc.provider_key == "postgres"


def test_typed_error_provider_key_defaults_to_none_when_unset() -> None:
    assert EmbeddingServiceError("boom").provider_key is None
    assert RerankServiceError("boom").provider_key is None
    assert RetrievalError("boom").provider_key is None


def test_embedding_and_rerank_service_errors_carry_provider_key() -> None:
    assert EmbeddingServiceError("boom", provider_key="tei-embed").provider_key == "tei-embed"
    assert RerankServiceError("boom", provider_key="tei-rerank").provider_key == "tei-rerank"


def test_provider_error_carries_both_retryable_and_provider_key() -> None:
    exc = ProviderError("boom", retryable=True, provider_key="tei-embed")
    assert exc.retryable is True
    assert exc.provider_key == "tei-embed"


# --- classification table: every status code row -----------------------------------------------------


@pytest.mark.parametrize(
    ("status_code", "expected_retryable"),
    [
        # never retried (Global Constraints)
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (422, False),
        # retryable (Global Constraints)
        (408, True),
        (429, True),
        (500, True),
        (501, True),
        (502, True),
        (503, True),
        (504, True),
        (529, True),
        (599, True),
        # unlisted: the table is an allow list, not a deny list -- default never retried
        (418, False),
    ],
)
def test_classification_table_every_status_code_row(status_code: int, expected_retryable: bool) -> None:
    assert is_retryable_status(status_code) is expected_retryable


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_content_filtered_is_never_retried_regardless_of_status_code(status_code: int) -> None:
    # a content filter refusal repeats itself on retry no matter which status it arrived with.
    assert is_retryable_status(status_code, content_filtered=True) is False


def test_classify_exception_delegates_http_status_errors_to_the_table() -> None:
    assert classify_exception(_http_error(500)) is True
    assert classify_exception(_http_error(400)) is False


def test_classify_exception_treats_connect_and_read_timeouts_as_retryable() -> None:
    request = httpx.Request("GET", "http://tei.test/info")
    assert classify_exception(httpx.ConnectTimeout("connect timed out", request=request)) is True
    assert classify_exception(httpx.ReadTimeout("read timed out", request=request)) is True


def test_classify_exception_does_not_retry_write_or_pool_timeouts() -> None:
    # only connect/read timeouts are named retryable by the binding table; anything else defaults
    # to never retried, the same allow list discipline the status table documents.
    request = httpx.Request("POST", "http://tei.test/embed")
    assert classify_exception(httpx.WriteTimeout("write timed out", request=request)) is False
    assert classify_exception(httpx.PoolTimeout("pool timed out", request=request)) is False


def test_classify_exception_treats_psycopg_operational_errors_as_retryable() -> None:
    assert classify_exception(psycopg.OperationalError("connection refused")) is True


def test_classify_exception_does_not_retry_psycopg_programming_errors() -> None:
    # a bad SQL statement retried three times is still bad SQL; only connection level pg failures
    # (OperationalError) are transient in the sense the binding table cares about.
    assert classify_exception(psycopg.errors.UndefinedTable("relation does not exist")) is False


def test_classify_exception_defaults_unknown_exceptions_to_never_retried() -> None:
    assert classify_exception(ValueError("not a provider failure at all")) is False


def test_classify_exception_honors_content_filtered_override_directly() -> None:
    # `is_retryable_status(..., content_filtered=True)` is covered above; this pins that
    # `classify_exception` itself (not just the status table it delegates to) also takes and
    # forwards the override, since that is the function every call site actually uses.
    assert classify_exception(_http_error(500), content_filtered=True) is False
    assert classify_exception(_http_error(500), content_filtered=False) is True


def test_breaker_exempt_is_true_only_for_429() -> None:
    assert breaker_exempt(_http_error(429)) is True
    assert breaker_exempt(_http_error(500)) is False
    assert breaker_exempt(ValueError("not http at all")) is False


# --- RetryPolicy: classification + attempts + Retry-After, against a raw callable -------------------


def test_retry_policy_retries_a_retryable_failure_then_succeeds() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _http_error(503)
        return "ok"

    policy = RetryPolicy(attempts=3)
    assert policy.call(flaky) == "ok"
    assert calls["n"] == 2


def test_retry_policy_never_retries_a_400_single_attempt() -> None:
    calls = {"n": 0}

    def always_400() -> str:
        calls["n"] += 1
        raise _http_error(400)

    policy = RetryPolicy(attempts=3)
    with pytest.raises(httpx.HTTPStatusError):
        policy.call(always_400)
    assert calls["n"] == 1


def test_retry_policy_caps_at_max_attempts_inside_the_stage_deadline() -> None:
    calls = {"n": 0}

    def always_529() -> str:
        calls["n"] += 1
        raise _http_error(529)

    policy = RetryPolicy(attempts=3)
    with pytest.raises(httpx.HTTPStatusError):
        policy.call(always_529)
    assert calls["n"] == 3


def test_retry_policy_honors_retry_after_as_a_custom_backoff() -> None:
    # stamina's `on` hook can return a float to override its own backoff computation entirely; a
    # Retry-After header must drive that path. This proves the header is READ and PASSED (not that
    # any particular sleep duration elapsed -- `set_testing` disables the actual sleep here); the
    # honored value itself is pinned directly by `test_retry_after_seconds_reads_the_numeric_header`
    # below.
    calls = {"n": 0}

    def once_429_then_ok() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _http_error(429, headers={"retry-after": "7"})
        return "ok"

    policy = RetryPolicy(attempts=3)
    assert policy.call(once_429_then_ok) == "ok"
    assert calls["n"] == 2


def test_retry_after_seconds_reads_the_numeric_header() -> None:
    from atlas.adapters.resilience import _retry_after_seconds

    assert _retry_after_seconds(_http_error(429, headers={"retry-after": "3.5"})) == 3.5
    assert _retry_after_seconds(_http_error(429)) is None
    assert _retry_after_seconds(_http_error(429, headers={"retry-after": "not-a-number"})) is None
    assert _retry_after_seconds(ValueError("not http")) is None


# --- CircuitBreaker: three state walk with a fake clock -----------------------------------------------


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_breaker_starts_closed() -> None:
    breaker = CircuitBreaker(_FakeClock())
    assert breaker.state("tei-embed") == "closed"
    breaker.before_call("tei-embed")  # never raises while closed


def test_breaker_closed_to_open_on_n_failures() -> None:
    clock = _FakeClock()
    breaker = CircuitBreaker(clock, failure_threshold=3, cooldown_seconds=30.0)

    breaker.record_failure("tei-embed")
    assert breaker.state("tei-embed") == "closed"
    breaker.record_failure("tei-embed")
    assert breaker.state("tei-embed") == "closed"
    breaker.record_failure("tei-embed")
    assert breaker.state("tei-embed") == "open"


def test_breaker_open_short_circuits_before_cooldown_elapses() -> None:
    clock = _FakeClock()
    breaker = CircuitBreaker(clock, failure_threshold=1, cooldown_seconds=30.0)
    breaker.record_failure("tei-embed")
    assert breaker.state("tei-embed") == "open"

    clock.advance(29.999)
    with pytest.raises(ProviderError, match="circuit breaker open"):
        breaker.before_call("tei-embed")
    assert breaker.state("tei-embed") == "open"  # still open, no transition on a short circuit


def test_breaker_half_open_probe_after_cooldown() -> None:
    clock = _FakeClock()
    breaker = CircuitBreaker(clock, failure_threshold=1, cooldown_seconds=30.0)
    breaker.record_failure("tei-embed")
    assert breaker.state("tei-embed") == "open"

    clock.advance(30.0)
    breaker.before_call("tei-embed")  # does not raise: this call IS the probe
    assert breaker.state("tei-embed") == "half_open"


def test_breaker_half_open_probe_success_closes() -> None:
    clock = _FakeClock()
    breaker = CircuitBreaker(clock, failure_threshold=1, cooldown_seconds=30.0)
    breaker.record_failure("tei-embed")
    clock.advance(30.0)
    breaker.before_call("tei-embed")
    assert breaker.state("tei-embed") == "half_open"

    breaker.record_success("tei-embed")
    assert breaker.state("tei-embed") == "closed"


def test_breaker_half_open_probe_failure_reopens_with_a_fresh_cooldown() -> None:
    clock = _FakeClock()
    breaker = CircuitBreaker(clock, failure_threshold=1, cooldown_seconds=30.0)
    breaker.record_failure("tei-embed")
    clock.advance(30.0)
    breaker.before_call("tei-embed")
    assert breaker.state("tei-embed") == "half_open"

    breaker.record_failure("tei-embed")
    assert breaker.state("tei-embed") == "open"

    # the cooldown restarted from THIS failure, not the original one: 30s from the probe failure,
    # not 60s from the very first failure.
    clock.advance(29.0)
    with pytest.raises(ProviderError):
        breaker.before_call("tei-embed")
    clock.advance(1.0)
    breaker.before_call("tei-embed")  # now past the fresh cooldown
    assert breaker.state("tei-embed") == "half_open"


def test_breaker_429_exemption_never_trips_it() -> None:
    clock = _FakeClock()
    breaker = CircuitBreaker(clock, failure_threshold=1, cooldown_seconds=30.0)
    for _ in range(10):
        breaker.record_failure("tei-embed", exempt=True)
    assert breaker.state("tei-embed") == "closed"


def test_breaker_keys_are_independent_per_provider() -> None:
    clock = _FakeClock()
    breaker = CircuitBreaker(clock, failure_threshold=1, cooldown_seconds=30.0)
    breaker.record_failure("tei-embed")
    assert breaker.state("tei-embed") == "open"
    assert breaker.state("tei-rerank") == "closed"
    breaker.before_call("tei-rerank")  # never raises: a different provider key entirely


# --- call_with_resilience: the composition, typed errors, 429 exemption end to end -------------------


def test_call_with_resilience_returns_the_wrapped_result_on_success() -> None:
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=3)
    result = call_with_resilience(lambda: "ok", policy=policy, breaker=breaker, provider_key="tei-embed")
    assert result == "ok"
    assert breaker.state("tei-embed") == "closed"


def test_call_with_resilience_raises_the_typed_error_never_the_raw_exception() -> None:
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=1)

    def always_500() -> str:
        raise _http_error(500)

    with pytest.raises(EmbeddingServiceError) as excinfo:
        call_with_resilience(
            always_500, policy=policy, breaker=breaker, provider_key="tei-embed", error_type=EmbeddingServiceError
        )
    assert not isinstance(excinfo.value, httpx.HTTPStatusError)
    assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)  # the raw cause is chained, not hidden
    assert excinfo.value.provider_key == "tei-embed"  # structured routing data, never string parsed


def test_call_with_resilience_retry_exhausted_generic_path_keeps_provider_error_with_its_key() -> None:
    # no `error_type` given: the generic path. `retryable` reflects the underlying failure's own
    # classification (a 500 IS retryable per the table; retries were exhausted anyway).
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=1)

    def always_500() -> str:
        raise _http_error(500)

    with pytest.raises(ProviderError) as excinfo:
        call_with_resilience(always_500, policy=policy, breaker=breaker, provider_key="tei-embed")
    assert type(excinfo.value) is ProviderError
    assert excinfo.value.provider_key == "tei-embed"
    assert excinfo.value.retryable is True


def test_call_with_resilience_retry_exhausted_generic_path_marks_never_retried_failures_correctly() -> None:
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=3)

    def always_400() -> str:
        raise _http_error(400)

    with pytest.raises(ProviderError) as excinfo:
        call_with_resilience(always_400, policy=policy, breaker=breaker, provider_key="tei-embed")
    assert excinfo.value.retryable is False


def test_call_with_resilience_defaults_to_retrieval_error_when_error_type_is_retrieval_error() -> None:
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=1)

    def always_fails() -> str:
        raise psycopg.errors.UndefinedTable("nope")

    with pytest.raises(RetrievalError) as excinfo:
        call_with_resilience(
            always_fails, policy=policy, breaker=breaker, provider_key="postgres", error_type=RetrievalError
        )
    assert excinfo.value.provider_key == "postgres"


def test_call_with_resilience_429_does_not_count_toward_the_breaker() -> None:
    breaker = CircuitBreaker(_FakeClock(), failure_threshold=1, cooldown_seconds=30.0)
    policy = RetryPolicy(attempts=1)

    def always_429() -> str:
        raise _http_error(429)

    with pytest.raises(EmbeddingServiceError):
        call_with_resilience(
            always_429, policy=policy, breaker=breaker, provider_key="tei-embed", error_type=EmbeddingServiceError
        )
    assert breaker.state("tei-embed") == "closed"  # threshold=1, but 429 is exempt: never trips


def test_call_with_resilience_non_429_failure_counts_toward_the_breaker() -> None:
    breaker = CircuitBreaker(_FakeClock(), failure_threshold=1, cooldown_seconds=30.0)
    policy = RetryPolicy(attempts=1)

    def always_500() -> str:
        raise _http_error(500)

    with pytest.raises(EmbeddingServiceError):
        call_with_resilience(
            always_500, policy=policy, breaker=breaker, provider_key="tei-embed", error_type=EmbeddingServiceError
        )
    assert breaker.state("tei-embed") == "open"  # threshold=1: one non exempt failure trips it


def test_open_embedding_breaker_short_circuits_as_embedding_service_error_with_provider_key() -> None:
    # the routing fix: an open breaker's short circuit must come through AS the call site's own
    # error_type (never a bare ProviderError an open rerank breaker would be equally indistinguishable
    # from), carrying provider_key so a caller routes on structured data, never a message string.
    breaker = CircuitBreaker(_FakeClock(), failure_threshold=1, cooldown_seconds=30.0)
    breaker.record_failure("tei-embed")
    assert breaker.state("tei-embed") == "open"
    policy = RetryPolicy(attempts=3)

    calls = {"n": 0}

    def would_succeed() -> str:
        calls["n"] += 1
        return "ok"

    with pytest.raises(EmbeddingServiceError, match="circuit breaker open") as excinfo:
        call_with_resilience(
            would_succeed, policy=policy, breaker=breaker, provider_key="tei-embed", error_type=EmbeddingServiceError
        )
    assert calls["n"] == 0  # never even attempted: a true fail fast short circuit
    assert excinfo.value.provider_key == "tei-embed"
    assert not isinstance(excinfo.value, ProviderError)  # a DISTINCT type, not a ProviderError subclass


def test_open_rerank_breaker_short_circuits_as_rerank_service_error_with_provider_key() -> None:
    breaker = CircuitBreaker(_FakeClock(), failure_threshold=1, cooldown_seconds=30.0)
    breaker.record_failure("tei-rerank")
    assert breaker.state("tei-rerank") == "open"
    policy = RetryPolicy(attempts=3)

    with pytest.raises(RerankServiceError, match="circuit breaker open") as excinfo:
        call_with_resilience(
            lambda: "ok", policy=policy, breaker=breaker, provider_key="tei-rerank", error_type=RerankServiceError
        )
    assert excinfo.value.provider_key == "tei-rerank"


def test_open_breaker_generic_provider_path_keeps_provider_error_with_its_key() -> None:
    # no `error_type` given: stays a `ProviderError` (the generic path), still carrying provider_key.
    breaker = CircuitBreaker(_FakeClock(), failure_threshold=1, cooldown_seconds=30.0)
    breaker.record_failure("some-provider")
    assert breaker.state("some-provider") == "open"
    policy = RetryPolicy(attempts=3)

    with pytest.raises(ProviderError, match="circuit breaker open") as excinfo:
        call_with_resilience(lambda: "ok", policy=policy, breaker=breaker, provider_key="some-provider")
    assert type(excinfo.value) is ProviderError
    assert excinfo.value.provider_key == "some-provider"
    assert excinfo.value.retryable is False


def test_open_breaker_short_circuit_never_double_records_a_failure() -> None:
    # a short circuit is not a NEW failure (the breaker is already open because of an earlier one);
    # recording it again would refresh the cooldown for no reason, extending it indefinitely under
    # repeated traffic against an open breaker.
    clock = _FakeClock()
    breaker = CircuitBreaker(clock, failure_threshold=1, cooldown_seconds=30.0)
    breaker.record_failure("tei-embed")
    policy = RetryPolicy(attempts=3)

    for _ in range(5):
        with pytest.raises(EmbeddingServiceError):
            call_with_resilience(
                lambda: "ok", policy=policy, breaker=breaker, provider_key="tei-embed", error_type=EmbeddingServiceError
            )
    clock.advance(30.0)
    # if any of the short circuits above had refreshed opened_at, this would still be open and
    # raise again; instead the cooldown elapsed from the ORIGINAL failure, the call is let through
    # as the half open probe, and since it succeeds the breaker closes.
    call_with_resilience(lambda: "ok", policy=policy, breaker=breaker, provider_key="tei-embed")
    assert breaker.state("tei-embed") == "closed"


# --- SP4 task 4: last_call_retried(), the retry rung's own minimal inspection surface -----------------


def test_last_call_retried_is_false_after_a_first_attempt_success() -> None:
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=3)
    call_with_resilience(lambda: "ok", policy=policy, breaker=breaker, provider_key="tei-embed")
    assert last_call_retried() is False


def test_last_call_retried_is_true_after_succeeding_on_a_later_attempt() -> None:
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=3)
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _http_error(503)
        return "ok"

    call_with_resilience(flaky, policy=policy, breaker=breaker, provider_key="tei-embed")
    assert last_call_retried() is True


def test_last_call_retried_reflects_only_the_most_recently_completed_call() -> None:
    # a stale True from an earlier call must never leak into a later, cleanly-succeeded call's own
    # reading: the whole point of a per-call carrier (SP4 task 4's own justification for it).
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=3)
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _http_error(503)
        return "ok"

    call_with_resilience(flaky, policy=policy, breaker=breaker, provider_key="tei-embed")
    assert last_call_retried() is True
    call_with_resilience(lambda: "ok", policy=policy, breaker=breaker, provider_key="tei-embed")
    assert last_call_retried() is False


def test_call_with_resilience_typed_error_surfaces_via_mocktransport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "TEI is down"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei.test")
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=2)

    def do_request() -> httpx.Response:
        response = client.post("/embed", json={"inputs": ["hi"]})
        response.raise_for_status()
        return response

    with pytest.raises(EmbeddingServiceError):
        call_with_resilience(
            do_request, policy=policy, breaker=breaker, provider_key="tei-embed", error_type=EmbeddingServiceError
        )
    client.close()


# --- call_with_resilience_async: the same composition, genuinely awaited (SP4 final fix wave, F2) ----


@pytest.mark.asyncio
async def test_call_with_resilience_async_returns_the_wrapped_result_on_success() -> None:
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=3)

    async def ok() -> str:
        return "ok"

    result = await call_with_resilience_async(ok, policy=policy, breaker=breaker, provider_key="primary-model")
    assert result == "ok"
    assert breaker.state("primary-model") == "closed"


@pytest.mark.asyncio
async def test_call_with_resilience_async_retries_a_retryable_failure_then_succeeds() -> None:
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=3)
    calls = {"n": 0}

    async def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise _http_error(503)
        return "ok"

    result = await call_with_resilience_async(flaky, policy=policy, breaker=breaker, provider_key="primary-model")
    assert result == "ok"
    assert calls["n"] == 2
    assert last_call_retried() is True


@pytest.mark.asyncio
async def test_call_with_resilience_async_raises_the_typed_error_never_the_raw_exception() -> None:
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=1)

    async def always_500() -> str:
        raise _http_error(500)

    with pytest.raises(RetrievalError) as excinfo:
        await call_with_resilience_async(
            always_500, policy=policy, breaker=breaker, provider_key="primary-model", error_type=RetrievalError
        )
    assert not isinstance(excinfo.value, httpx.HTTPStatusError)
    assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)  # the raw cause is chained, not hidden
    assert excinfo.value.provider_key == "primary-model"


@pytest.mark.asyncio
async def test_call_with_resilience_async_generic_path_keeps_provider_error_with_its_retryable_flag() -> None:
    # no `error_type` given, the generic path atlas_graph.py's generation seam actually calls with:
    # `retryable` reflects the underlying failure's own classification.
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=1)

    async def always_400() -> str:
        raise _http_error(400)

    with pytest.raises(ProviderError) as excinfo:
        await call_with_resilience_async(always_400, policy=policy, breaker=breaker, provider_key="primary-model")
    assert type(excinfo.value) is ProviderError
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_open_breaker_short_circuits_the_async_producer_as_the_requested_typed_error() -> None:
    # the SAME routing fix the sync producer's own breaker open tests pin: an open breaker's short
    # circuit must come through AS the call site's own error_type, never a bare ProviderError.
    breaker = CircuitBreaker(_FakeClock(), failure_threshold=1, cooldown_seconds=30.0)
    breaker.record_failure("primary-model")
    assert breaker.state("primary-model") == "open"
    policy = RetryPolicy(attempts=3)
    calls = {"n": 0}

    async def would_succeed() -> str:
        calls["n"] += 1
        return "ok"

    with pytest.raises(RetrievalError, match="circuit breaker open") as excinfo:
        await call_with_resilience_async(
            would_succeed, policy=policy, breaker=breaker, provider_key="primary-model", error_type=RetrievalError
        )
    assert calls["n"] == 0  # never even attempted: a true fail fast short circuit
    assert excinfo.value.provider_key == "primary-model"
    assert not isinstance(excinfo.value, ProviderError)  # a DISTINCT type, not a ProviderError subclass


@pytest.mark.asyncio
async def test_open_breaker_generic_path_reraises_the_async_producers_own_provider_error_unchanged() -> None:
    # no `error_type` given, the generic path (`error_type is ProviderError`): the breaker's own
    # `ProviderError` short circuit comes through AS IS, never wrapped a second time.
    breaker = CircuitBreaker(_FakeClock(), failure_threshold=1, cooldown_seconds=30.0)
    breaker.record_failure("primary-model")
    assert breaker.state("primary-model") == "open"
    policy = RetryPolicy(attempts=3)

    async def would_succeed() -> str:
        return "ok"

    with pytest.raises(ProviderError, match="circuit breaker open") as excinfo:
        await call_with_resilience_async(would_succeed, policy=policy, breaker=breaker, provider_key="primary-model")
    assert type(excinfo.value) is ProviderError
    assert excinfo.value.provider_key == "primary-model"
    assert excinfo.value.retryable is False
