"""The resilience module (SP4 task 3): retry classification, a stamina wrapped retry policy, a per
provider three state circuit breaker, and the typed error boundary every adapter raises instead of
letting a raw client exception (`httpx`, `psycopg`) leak past the port it sits behind.

Composition, not inheritance: `RetryPolicy` only knows how to retry one call (classification,
exponential backoff with jitter, Retry-After honored, capped attempts inside a stage deadline);
`CircuitBreaker` only knows per provider key state (closed, open, half open); `call_with_resilience`
wires the two together around one call and translates EVERY outcome, the breaker's own open circuit
short circuit included, into the call site's own typed error, carrying `provider_key` on the way
out. An adapter (today, `pgvector_retriever.py`) owns the provider keys and which typed error each
call site raises.

Determinism note: the breaker's clock is an injected `Callable[[], float]` (`time.monotonic`'s own
shape), never read directly here, so a test walks the state machine with a fake clock and gets a
byte reproducible answer. `RetryPolicy`'s own backoff sleep is real wall clock time in production
(retries only happen against a live TEI/Postgres, never inside the hermetic replay lane), and tests
disable it via `stamina.set_testing(True)` rather than injecting a fake sleep, stamina's own
documented pattern for exactly this.
"""
from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import TypeVar

import httpx
import psycopg
import stamina

T = TypeVar("T")


# --- typed errors: never let a raw httpx/psycopg exception past an adapter boundary -----------------


class RetrievalError(Exception):
    """The base typed error every retrieval adapter raises instead of a raw client exception.
    `provider_key` names which provider key (the same string the breaker and policy are scoped by,
    e.g. "tei-embed") produced this failure, when known, carried on every subclass and set on every
    raise inside `call_with_resilience`, so a caller (the degradation ladder, Task 4) routes on
    structured data instead of parsing the message string."""

    def __init__(self, message: str, *, provider_key: str | None = None) -> None:
        super().__init__(message)
        self.provider_key = provider_key


class EmbeddingServiceError(RetrievalError):
    """The TEI embed service (or its shared `/info` endpoint) failed: retries were exhausted, the
    call was classified never retried, or the breaker is open for the `tei-embed` provider key."""


class RerankServiceError(RetrievalError):
    """The TEI rerank service failed: retries were exhausted, the call was classified never
    retried, or the breaker is open for the `tei-rerank` provider key."""


class ProviderError(RetrievalError):
    """A provider level failure not tied to one call site's own typed error: the generic path
    `call_with_resilience` raises through when no call site specific `error_type` was requested.
    Today that is the breaker's own fail fast short circuit (`CircuitBreaker.before_call` always
    raises this, whatever `provider_key`) and any retry exhausted failure at a call site that never
    asked for a more specific type; later also Task 4's generation provider. `retryable` records
    whether the triggering condition was itself retryable, so a caller (the degradation ladder,
    Task 4) can tell "this provider is circuit broken, do not bother retrying" apart from "this one
    call was a validation error, do not bother retrying" without re deriving the classification
    itself."""

    def __init__(self, message: str, *, retryable: bool, provider_key: str | None = None) -> None:
        super().__init__(message, provider_key=provider_key)
        self.retryable = retryable


# --- retry classification (the binding table, Global Constraints / the HLD section 4.7) ------------

# Never retried regardless of anything else below: the request itself was wrong (auth, not found,
# a validation failure a retry cannot fix), so repeating it only repeats the same outcome.
_NEVER_RETRIED_STATUS_CODES = frozenset({400, 401, 403, 404, 422})

# 408 (request timeout) and 429 (rate limited) are 4xx codes that ARE transient, named explicitly
# since they fall outside the 5xx range check below. 529 (overloaded) is already inside that range
# numerically; it is named here too only because the binding table calls it out by number, so a
# reader scanning this set does not have to do the arithmetic to confirm it is covered.
_EXPLICIT_RETRYABLE_STATUS_CODES = frozenset({408, 429, 529})


def is_retryable_status(status_code: int, *, content_filtered: bool = False) -> bool:
    """The binding retry classification table: retryable = 408, 429, 5xx (529 included); never
    retried = 400, 401, 403, 404, 422, or any response a caller marks `content_filtered` (a
    provider content filter refusal repeats itself on retry no matter which status code it arrived
    with). This table is an allow list, not a deny list: an unrecognized status code defaults to
    never retried rather than retried, so a code nobody explicitly classified never gets retried by
    accident."""
    if content_filtered:
        return False
    if status_code in _NEVER_RETRIED_STATUS_CODES:
        return False
    if status_code in _EXPLICIT_RETRYABLE_STATUS_CODES:
        return True
    return 500 <= status_code < 600


def classify_exception(exc: BaseException, *, content_filtered: bool = False) -> bool:
    """Retry classification for whatever exception a wrapped call actually raised. HTTP status
    errors defer to `is_retryable_status`; connect and read timeouts are retryable per the binding
    table; a Postgres `OperationalError` (a connection level failure, the closest pg equivalent to
    a connect timeout) is retryable; every other exception type defaults to never retried, the same
    allow list discipline `is_retryable_status` documents."""
    if isinstance(exc, httpx.HTTPStatusError):
        return is_retryable_status(exc.response.status_code, content_filtered=content_filtered)
    if isinstance(exc, (httpx.ConnectTimeout, httpx.ReadTimeout)):
        return True
    if isinstance(exc, psycopg.OperationalError):
        return True
    return False


def breaker_exempt(exc: BaseException) -> bool:
    """429 never counts toward the breaker (Global Constraints): a rate limit is the provider
    asking the caller to slow down, not evidence the provider itself is unhealthy, so it must never
    trip a breaker meant to catch genuine outages."""
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Retry-After honored (Global Constraints) when the failing exception carries the header,
    numeric seconds form only (the HTTP date form is a known, documented simplification: none of
    this adapter's providers send it). None means there is nothing to honor, so the caller falls
    back to the policy's own exponential backoff with jitter (see `RetryPolicy`'s own docstring for
    the exact formula; it is not the unrelated "full jitter" algorithm, which has no deterministic
    term at all)."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    value = exc.response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# --- RetryPolicy: stamina wrapped, classified, exponential backoff with jitter, Retry-After honored --


class RetryPolicy:
    """One call, retried: classification per the table above, capped at `attempts` tries (default
    3, the Global Constraints ceiling) inside `stage_deadline_seconds` total, Retry-After honored
    when the failing exception carries it (overrides the backoff below entirely for that attempt).

    Backoff shape: stamina's own documented formula (`stamina.retry`'s docstring), unmodified. The
    wait before attempt n (1 based, n > 1) is

        min(wait_max_seconds, wait_initial_seconds * wait_exp_base ** (n - 1) + random(0, wait_jitter_seconds))

    With the defaults below that is `0.1 + U(0, 0.1)` seconds before attempt 2 and
    `0.2 + U(0, 0.1)` seconds before attempt 3: real attempt scaled exponential growth (`wait_exp_base
    = 2.0` doubles the deterministic term each attempt), plus a bounded random component so two
    callers retrying the same provider at the same moment do not stay in lockstep.
    `wait_max_seconds=2.0` only exists as a safety ceiling; at `attempts=3` the deterministic term
    never gets close to it, so the cap never actually binds here, and the worst case total backoff
    across all 3 attempts stays under a second, well inside every `stage_deadline_seconds` this
    module is constructed with (30 to 120)."""

    def __init__(
        self,
        *,
        attempts: int = 3,
        stage_deadline_seconds: float = 30.0,
        wait_initial_seconds: float = 0.1,
        wait_exp_base: float = 2.0,
        wait_jitter_seconds: float = 0.1,
        wait_max_seconds: float = 2.0,
    ) -> None:
        self._caller = stamina.RetryingCaller(
            attempts=attempts,
            timeout=stage_deadline_seconds,
            wait_initial=wait_initial_seconds,
            wait_max=wait_max_seconds,
            wait_jitter=wait_jitter_seconds,
            wait_exp_base=wait_exp_base,
        )
        # SP4 final fix wave (F2): the async twin of `_caller` above, same shape, same params --
        # `stamina.AsyncRetryingCaller` is the library's own async counterpart, needed for a call
        # site whose `fn` is a real coroutine (the live generation seam's `model.ainvoke(...)`, a
        # genuine network call, never a `fn()` that merely raises synchronously the way every
        # existing retrieval seam's stub does). Built once here so both callers share one set of
        # retry/backoff parameters, never two independently tuned copies.
        self._async_caller = stamina.AsyncRetryingCaller(
            attempts=attempts,
            timeout=stage_deadline_seconds,
            wait_initial=wait_initial_seconds,
            wait_max=wait_max_seconds,
            wait_jitter=wait_jitter_seconds,
            wait_exp_base=wait_exp_base,
        )

    @staticmethod
    def _hook(exc: Exception) -> bool | float:
        if not classify_exception(exc):
            return False
        retry_after = _retry_after_seconds(exc)
        return retry_after if retry_after is not None else True

    def call(self, fn: Callable[[], T]) -> T:
        return self._caller(self._hook, fn)

    async def call_async(self, fn: Callable[[], Awaitable[T]]) -> T:
        """Async twin of `call` (SP4 final fix wave, F2): the SAME classification hook, awaited
        through `stamina.AsyncRetryingCaller` instead of run synchronously, for a call site whose
        `fn` genuinely needs to be awaited (a real network call, not a stub that raises inline)."""
        return await self._async_caller(self._hook, fn)


# --- CircuitBreaker: per provider key, three state, injected clock -----------------------------------

_FAILURE_THRESHOLD = 3
# Three consecutive call level failures trip the breaker. Each of those failures already exhausted
# its own `RetryPolicy` (up to 3 attempts on its own), so this is not "3 raw errors": it is 3 whole
# calls that failed even after retrying. Low enough that a genuinely down provider stops receiving
# traffic quickly; high enough that one call unlucky enough to fail after its own retries never
# trips the breaker alone.

_COOLDOWN_SECONDS = 30.0
# How long an open breaker refuses calls before letting exactly one half open probe through.
# Matches this adapter's own request timeout order of magnitude (`_TIMEOUT_SECONDS` in
# pgvector_retriever.py), so a recovered provider rejoins within about one request's worth of
# patience, not minutes, while a still down provider is not hammered on every retry either.


@dataclass(frozen=True)
class _BreakerState:
    status: str = "closed"  # closed | open | half_open
    failures: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    """Per provider three state breaker (closed, open, half open). State lives in one dict keyed by
    `provider_key`, each value a frozen `_BreakerState` replaced wholesale on every transition (the
    dict is the one mutable boundary here, mirroring `domain/accounts.py`'s own single mutable
    store). `clock` is an injected callable (`time.monotonic`'s own shape); this class never reads
    a wall clock itself, only whatever the caller passed in, so a test can walk the state machine
    with a fake clock deterministically. Calling `record_failure` directly while a key is already
    "open" (bypassing `before_call`) refreshes the cooldown; `call_with_resilience` below never does
    this, since it always calls `before_call` first and that raises instead of falling through."""

    def __init__(
        self,
        clock: Callable[[], float],
        *,
        failure_threshold: int = _FAILURE_THRESHOLD,
        cooldown_seconds: float = _COOLDOWN_SECONDS,
    ) -> None:
        self._clock = clock
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._states: dict[str, _BreakerState] = {}

    def before_call(self, provider_key: str) -> None:
        """Fail fast: raise `ProviderError` if `provider_key` is open and still cooling down. An
        open breaker past its cooldown transitions to half open and lets exactly this one call
        through as the probe."""
        state = self._states.get(provider_key, _BreakerState())
        if state.status != "open":
            return
        if self._clock() - (state.opened_at or 0.0) < self._cooldown_seconds:
            raise ProviderError(
                f"circuit breaker open for provider {provider_key!r}", retryable=False, provider_key=provider_key
            )
        self._states[provider_key] = replace(state, status="half_open")

    def record_success(self, provider_key: str) -> None:
        self._states[provider_key] = _BreakerState()

    def record_failure(self, provider_key: str, *, exempt: bool = False) -> None:
        """`exempt` is True for 429s (Global Constraints: 429 never counts toward the breaker)."""
        if exempt:
            return
        state = self._states.get(provider_key, _BreakerState())
        if state.status == "half_open":
            # The probe failed: back to open immediately, a fresh cooldown, no partial credit.
            self._states[provider_key] = replace(state, status="open", opened_at=self._clock())
            return
        failures = state.failures + 1
        opened = failures >= self._failure_threshold
        self._states[provider_key] = replace(
            state,
            failures=failures,
            status="open" if opened else "closed",
            opened_at=self._clock() if opened else None,
        )

    def state(self, provider_key: str) -> str:
        """Inspection only (tests, SP6 tracing later): current status, no side effects."""
        return self._states.get(provider_key, _BreakerState()).status


# --- composition: one call, through the policy and the breaker, typed on the way out -----------------

# The retry rung's own carrier (SP4 task 4): the minimal resilience inspection surface extension
# the degradation ladder needs, contextvar based, the same isolation pattern
# `pgvector_retriever._search_result` already uses (per thread by default, per asyncio task once a
# task copies its context at creation), so interleaved concurrent calls each see only their own
# outcome. Additive only: `call_with_resilience`'s own signature and every existing call site are
# unchanged.
_last_attempt_count: contextvars.ContextVar[int] = contextvars.ContextVar(
    "atlas_resilience_last_attempt_count", default=0
)


def last_call_retried() -> bool:
    """Inspection only (SP4 task 4): whether the most recently COMPLETED `call_with_resilience`
    call in THIS execution context needed more than one attempt to succeed (stamina retried at
    least once and the retry succeeded). Read this immediately after the call you care about: a
    later `call_with_resilience` call in the same context overwrites it, and a call that ultimately
    raised never sets it (there is no "succeeded after N attempts" to report for a failure), so a
    stale value from an earlier, unrelated call is never mistaken for this one's outcome as long as
    the caller reads it right after the call it describes."""
    return _last_attempt_count.get() > 1


def call_with_resilience(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    breaker: CircuitBreaker,
    provider_key: str,
    error_type: type[RetrievalError] = ProviderError,
) -> T:
    """Compose one `RetryPolicy` and one `CircuitBreaker` around a single call: fail fast if the
    breaker is open for `provider_key`, otherwise run `fn` through the retry policy. EVERY failure,
    the breaker's own open circuit short circuit included, is translated into `error_type` (never a
    raw httpx or psycopg exception, and never a bare `ProviderError` when a call site asked for
    something more specific): an open embedding breaker raises `EmbeddingServiceError`, an open
    rerank breaker raises `RerankServiceError`, so a caller routes on the exception TYPE, never a
    string parsed message (the routing collapse a bare `ProviderError` for every provider caused).
    Every raised error carries `provider_key`. The untyped default (`error_type=ProviderError`) is
    the generic path: no call site specific type was requested, so the breaker's own `ProviderError`
    (open circuit, re raised as is) or a freshly built one (retry exhausted, `retryable` set from
    the underlying failure's own classification) comes through unchanged in kind. The outcome is
    also recorded back onto the breaker (429 exempt, see `breaker_exempt`); a breaker open short
    circuit is NOT a new failure (the breaker is already open because of an earlier one) and is
    never recorded again, which would only refresh its cooldown for no reason. On success, the
    attempt count this call needed is recorded for `last_call_retried` (SP4 task 4)."""
    attempts = {"n": 0}

    def counted_fn() -> T:
        attempts["n"] += 1
        return fn()

    try:
        breaker.before_call(provider_key)
        result = policy.call(counted_fn)
    except ProviderError as exc:
        if error_type is ProviderError:
            raise
        raise error_type(str(exc), provider_key=provider_key) from exc
    except Exception as exc:
        breaker.record_failure(provider_key, exempt=breaker_exempt(exc))
        message = f"{provider_key} call failed: {exc}"
        if error_type is ProviderError:
            raise ProviderError(message, retryable=classify_exception(exc), provider_key=provider_key) from exc
        raise error_type(message, provider_key=provider_key) from exc
    _last_attempt_count.set(attempts["n"])
    breaker.record_success(provider_key)
    return result


async def call_with_resilience_async(
    fn: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy,
    breaker: CircuitBreaker,
    provider_key: str,
    error_type: type[RetrievalError] = ProviderError,
) -> T:
    """The async twin of `call_with_resilience` above (SP4 final fix wave, F2): identical
    composition (breaker fail fast, the SAME classification table, translated to `error_type` on
    the way out, `last_call_retried` recorded on success), the only difference is `fn` is awaited
    through `RetryPolicy.call_async` (`stamina.AsyncRetryingCaller`) instead of run synchronously.
    Exists for a call site that needs a genuinely awaited retry -- today, the live generation seam
    (`atlas_graph.py`'s `_generate_message`, wrapping `model.ainvoke(...)`), the one seam
    `resilience.py`'s own module docstring named as a later addition ("later also Task 4's
    generation provider"). Every retrieval seam's own `fn` merely raises synchronously (an httpx
    call made through a sync `httpx.Client`), so `call_with_resilience` stays their producer
    unchanged; this is additive, not a replacement."""
    attempts = {"n": 0}

    async def counted_fn() -> T:
        attempts["n"] += 1
        return await fn()

    try:
        breaker.before_call(provider_key)
        result = await policy.call_async(counted_fn)
    except ProviderError as exc:
        if error_type is ProviderError:
            raise
        raise error_type(str(exc), provider_key=provider_key) from exc
    except Exception as exc:
        breaker.record_failure(provider_key, exempt=breaker_exempt(exc))
        message = f"{provider_key} call failed: {exc}"
        if error_type is ProviderError:
            raise ProviderError(message, retryable=classify_exception(exc), provider_key=provider_key) from exc
        raise error_type(message, provider_key=provider_key) from exc
    _last_attempt_count.set(attempts["n"])
    breaker.record_success(provider_key)
    return result
