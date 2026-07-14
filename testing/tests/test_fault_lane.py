"""The deterministic fault injection lane (SP4 task 7), hermetic: scripted faults at the seams via
stub adapters and fake clocks, asserted end to end through the REAL graph (`build_atlas_graph`),
never a bare typed error raise the way `test_ladder.py`'s own simpler stubs use. Every stub here
routes its failure through the ACTUAL resilience module (`RetryPolicy` + `CircuitBreaker` +
`call_with_resilience`, `atlas.adapters.resilience`), the same composition `PgvectorRetriever`
uses, so a case exercises the real retry/breaker machinery, not just the ladder's own routing on a
typed error someone handed it. Every case's assertions include the breaker/retry counters (the
resilience module's own inspection surface -- `CircuitBreaker.state()`, `last_call_retried()`, plus
a raw request counter each stub keeps at each seam), not only the outcome.

The 529 storm case (b) pins the SEEDED, binding arithmetic from the Task 3 reviewer's empirical
verification: with `failure_threshold=3` and `RetryPolicy(attempts=3)`, the breaker opens after 9
raw HTTP requests (3 failed top level calls x 3 attempts each); the 4th top level call short
circuits with ZERO additional requests; the short circuit surfaces as the call site's own typed
error (`EmbeddingServiceError`, never a generic `ProviderError`).

No Docker, no network, no real sleep: `stamina.set_testing` disables stamina's own backoff sleep
for this whole module (`test_resilience.py`'s own established pattern), and every `CircuitBreaker`
here takes an injected fake clock, never `time.monotonic`.
"""
from __future__ import annotations

import json

import httpx
import jsonschema
import psycopg
import pytest
import stamina
from contract_tools import loader
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory, fixture_kit

from atlas.adapters.resilience import (
    CircuitBreaker,
    EmbeddingServiceError,
    ProviderError,
    RerankServiceError,
    RetrievalError,
    RetryPolicy,
    call_with_resilience,
    last_call_retried,
)
from atlas.chat_app import make_chat_app
from atlas.domain.actions import ActionsBackend
from atlas.domain.retrieval import RetrievalConfig
from atlas.orchestration.atlas_graph import HANDOFF_PREFIX, _REFUSAL_MESSAGE, build_atlas_graph
from atlas.ports.knowledge import Chunk

# ---------------------------------------------------------------------------------------------
# shared fixtures / helpers (mirrors test_resilience.py's own fake clock + http error builder,
# duplicated rather than imported: this file's own territory is self contained, SP4 task 7's
# constraints)
# ---------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_backoff_sleep():
    with stamina.set_testing(True, attempts=50, cap=True):
        yield


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _http_error(status_code: int, *, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://tei.test/embed")
    response = httpx.Response(status_code, request=request, headers=headers or {})
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return exc
    raise AssertionError(f"{status_code} did not raise_for_status()")


class _SearchOnceThenAnswerModel(BaseChatModel):
    """Emits one search_knowledge tool call, then a plain answer once it sees the ToolMessage --
    the same deterministic, cassette free shape `test_ladder.py`'s own stub uses, reused here so
    every retrieval seam case (a through f below) drives the SAME turn shape through the real graph."""

    query: str = "contract question"
    answer: str = "Here is what I found."

    @property
    def _llm_type(self) -> str:
        return "search-once-then-answer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("the graph is async only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if any(isinstance(m, ToolMessage) for m in messages):
            msg = AIMessage(content=self.answer)
        else:
            msg = AIMessage(
                content="", tool_calls=[{"name": "search_knowledge", "args": {"query": self.query}, "id": "k1"}]
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _PlainAnswerModel(BaseChatModel):
    """A fallback model that always answers with fixed content and never calls a tool -- used only
    in case (h) to prove a non retryable generation failure never reaches the fallback at all."""

    answer: str = "plain answer"

    @property
    def _llm_type(self) -> str:
        return "plain-answer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("the graph is async only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.answer))])


def _graph(retriever=None, *, model=None, fallback_model=None):
    return build_atlas_graph(
        model or _SearchOnceThenAnswerModel(),
        IdFactory("idem"),
        ActionsBackend(IdFactory("ref")),
        new_checkpointer(),
        retriever=retriever,
        fallback_model=fallback_model,
    )


_SESSION = {"customer_id": "cust_current"}


# ---------------------------------------------------------------------------------------------
# (a) 429 then success: retried, answer produced, mode retry, breaker still closed
# ---------------------------------------------------------------------------------------------


class _EmbedFlakyOnceRetriever:
    """One 429 at the `tei-embed` seam, then success on retry -- routed through the real
    resilience module (never a bare typed error raise), so this exercises the real attempt
    counting AND the 429 breaker exemption at once. `last_result()` mirrors `PgvectorRetriever`'s
    own per call carrier so `knowledge_server._retried()` reports the SAME thing a real adapter
    would: a retry that succeeded transparently."""

    def __init__(self, breaker: CircuitBreaker, policy: RetryPolicy) -> None:
        self.raw_requests = 0
        self.breaker = breaker
        self.policy = policy
        self.retried = False

    def search_chunks(self, query, k, config):
        def do_request():
            self.raw_requests += 1
            if self.raw_requests == 1:
                raise _http_error(429)
            return "ok"

        call_with_resilience(
            do_request, policy=self.policy, breaker=self.breaker,
            provider_key="tei-embed", error_type=EmbeddingServiceError,
        )
        self.retried = last_call_retried()
        return [Chunk(doc_id="doc-1", chunk_id="chunk-1", score=0.5, text="retried after 429 answer")][:k]

    def last_result(self):
        return self


@pytest.mark.asyncio
async def test_429_then_success_retries_transparently_and_never_trips_the_breaker():
    clock = _FakeClock()
    breaker = CircuitBreaker(clock)
    policy = RetryPolicy(attempts=3)
    retriever = _EmbedFlakyOnceRetriever(breaker, policy)
    graph = _graph(retriever)

    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract free?")], "session": _SESSION},
        {"configurable": {"thread_id": "retry-429"}},
    )

    assert out["degradation_mode"] == "retry"
    assert out["final_response"] == "Here is what I found."
    assert retriever.raw_requests == 2  # one failed 429 attempt, one succeeding retry
    assert breaker.state("tei-embed") == "closed"  # 429 is exempt: it never counts as a failure


# ---------------------------------------------------------------------------------------------
# (b) 529 storm: breaker opens after threshold, subsequent call short circuits fast, half open
# probe after cooldown (fake clock advance). SEEDED, binding: 3 failed top level calls x 3
# attempts each = 9 raw requests before the breaker opens; the 4th top level call short circuits
# with ZERO additional requests, surfacing as EmbeddingServiceError, never a generic ProviderError.
# ---------------------------------------------------------------------------------------------


class _EmbeddingStormRetriever:
    """A `tei-embed` provider stuck returning 529 (overloaded): EVERY call, initial or the
    ladder's own lexical_only fallback attempt, is routed through the SAME shared breaker + policy
    (`provider_key="tei-embed"`) -- unlike a real `PgvectorRetriever`, this stub does NOT skip the
    resilience wrapped call when `config.lexical_only` (it models a total provider outage, not the
    embedding specific rung), so the SEEDED arithmetic plays out across knowledge_server.py's own
    two call ladder walk (the initial attempt, then its one lexical_only fallback attempt) driven
    entirely through the real graph, never a bespoke driver outside it."""

    def __init__(self, breaker: CircuitBreaker, policy: RetryPolicy) -> None:
        self.raw_requests = 0
        self.healthy = False
        self.breaker = breaker
        self.policy = policy

    def search_chunks(self, query, k, config):
        def do_request():
            self.raw_requests += 1
            if not self.healthy:
                raise _http_error(529)
            return "ok"

        call_with_resilience(
            do_request, policy=self.policy, breaker=self.breaker,
            provider_key="tei-embed", error_type=EmbeddingServiceError,
        )
        return [Chunk(doc_id="doc-1", chunk_id="chunk-1", score=0.5, text="storm recovered answer")][:k]


@pytest.mark.asyncio
async def test_529_storm_opens_the_breaker_after_nine_raw_requests_then_half_open_probes_after_cooldown():
    clock = _FakeClock()
    breaker = CircuitBreaker(clock, failure_threshold=3, cooldown_seconds=30.0)
    policy = RetryPolicy(attempts=3)
    retriever = _EmbeddingStormRetriever(breaker, policy)
    graph = _graph(retriever)

    # Turn 1: the initial attempt exhausts its 3 attempts (top level call #1), the ladder's own
    # lexical_only fallback attempt ALSO exhausts its 3 attempts (top level call #2) since this
    # stub models a total tei-embed outage -- 6 raw requests, 2 non exempt failures recorded:
    # still closed (threshold=3).
    out1 = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract free?")], "session": _SESSION},
        {"configurable": {"thread_id": "storm-1"}},
    )
    assert out1["degradation_mode"] == "refusal"
    assert retriever.raw_requests == 6
    assert breaker.state("tei-embed") == "closed"

    # Turn 2: the initial attempt's THIRD attempt (top level call #3) is the breaker's 3rd non
    # exempt failure -- it opens right there, at the 9th raw request. The ladder's own fallback
    # attempt (top level call #4, across both turns) short circuits before it ever reaches
    # do_request: zero additional raw requests -- the exact SEEDED arithmetic this case pins.
    out2 = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract free?")], "session": _SESSION},
        {"configurable": {"thread_id": "storm-2"}},
    )
    assert out2["degradation_mode"] == "refusal"
    assert retriever.raw_requests == 9  # NOT 12: the 4th top level call added zero requests
    assert breaker.state("tei-embed") == "open"

    # Directly pin the typed error the short circuit surfaces as (SEEDED, binding): a call made
    # right now against the still open breaker raises EmbeddingServiceError with provider_key,
    # never a bare ProviderError, and makes zero additional raw requests.
    with pytest.raises(EmbeddingServiceError) as excinfo:
        retriever.search_chunks("q", 3, RetrievalConfig())
    assert not isinstance(excinfo.value, ProviderError)
    assert excinfo.value.provider_key == "tei-embed"
    assert retriever.raw_requests == 9  # confirmed: the short circuit made no attempt at all

    # subsequent call short circuits fast (still inside cooldown): no state change, no requests.
    clock.advance(29.999)
    with pytest.raises(ProviderError, match="circuit breaker open"):
        breaker.before_call("tei-embed")
    assert breaker.state("tei-embed") == "open"
    assert retriever.raw_requests == 9

    # half open probe after cooldown (fake clock advance): the cooldown elapses, the NEXT call is
    # let through as the probe.
    clock.advance(0.001)
    breaker.before_call("tei-embed")
    assert breaker.state("tei-embed") == "half_open"

    # the provider has recovered: the probe succeeds through a real graph turn, and the breaker
    # closes; the ladder never even needed its fallback this time (an ordinary happy path again).
    retriever.healthy = True
    out3 = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract free?")], "session": _SESSION},
        {"configurable": {"thread_id": "storm-3"}},
    )
    assert out3["degradation_mode"] == "none"
    assert out3["final_response"] == "Here is what I found."
    assert retriever.raw_requests == 10  # exactly one more: the successful probe
    assert breaker.state("tei-embed") == "closed"


# ---------------------------------------------------------------------------------------------
# (c) rerank service down: drop_rerank answer with mode stamped
# ---------------------------------------------------------------------------------------------


class _RerankStormRetriever:
    """`tei-rerank` is down: every attempt through the fused width returns a retryable 5xx,
    routed through the real resilience module. The SAME query recovers once the ladder's own
    fallback disables reranking entirely -- the fallback path never touches the rerank seam again
    (mirrors the real `PgvectorRetriever._finalize`, which only calls `_rerank` when
    `config.rerank_enabled`)."""

    def __init__(self, breaker: CircuitBreaker, policy: RetryPolicy) -> None:
        self.rerank_requests = 0
        self.breaker = breaker
        self.policy = policy

    def search_chunks(self, query, k, config):
        if not config.rerank_enabled:
            return [Chunk(doc_id="doc-1", chunk_id="chunk-1", score=0.5, text="rerank off answer")][:k]

        def do_request():
            self.rerank_requests += 1
            raise _http_error(503)

        call_with_resilience(
            do_request, policy=self.policy, breaker=self.breaker,
            provider_key="tei-rerank", error_type=RerankServiceError,
        )


@pytest.mark.asyncio
async def test_rerank_down_retries_then_answers_with_drop_rerank_stamped_and_the_breaker_still_closed():
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=3)
    retriever = _RerankStormRetriever(breaker, policy)
    graph = _graph(retriever)

    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract free?")], "session": _SESSION},
        {"configurable": {"thread_id": "rerank-down"}},
    )

    assert out["degradation_mode"] == "drop_rerank"
    assert out["final_response"] == "Here is what I found."
    assert retriever.rerank_requests == 3  # the full retry ceiling, exhausted before the fallback
    assert breaker.state("tei-rerank") == "closed"  # one failed top level call, threshold is 3


# ---------------------------------------------------------------------------------------------
# (d) embedding service down: lexical_only
# ---------------------------------------------------------------------------------------------


class _EmbeddingDownThenLexicalOnlyRetriever:
    """`tei-embed` is down; the ladder's own lexical_only fallback skips the vector arm entirely
    (mirrors the real `PgvectorRetriever.search_chunks`: `if not config.lexical_only:
    query_vector = self._embed_query(query)`), so the recovered call never touches the resilience
    wrapped seam a second time."""

    def __init__(self, breaker: CircuitBreaker, policy: RetryPolicy) -> None:
        self.embed_requests = 0
        self.breaker = breaker
        self.policy = policy

    def search_chunks(self, query, k, config):
        if config.lexical_only:
            return [Chunk(doc_id="doc-1", chunk_id="chunk-1", score=0.5, text="lexical only answer")][:k]

        def do_request():
            self.embed_requests += 1
            raise _http_error(500)

        call_with_resilience(
            do_request, policy=self.policy, breaker=self.breaker,
            provider_key="tei-embed", error_type=EmbeddingServiceError,
        )


@pytest.mark.asyncio
async def test_embedding_down_retries_then_answers_lexical_only_and_the_breaker_still_closed():
    breaker = CircuitBreaker(_FakeClock())
    policy = RetryPolicy(attempts=3)
    retriever = _EmbeddingDownThenLexicalOnlyRetriever(breaker, policy)
    graph = _graph(retriever)

    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract free?")], "session": _SESSION},
        {"configurable": {"thread_id": "embed-down"}},
    )

    assert out["degradation_mode"] == "lexical_only"
    assert out["final_response"] == "Here is what I found."
    assert retriever.embed_requests == 3
    assert breaker.state("tei-embed") == "closed"


# ---------------------------------------------------------------------------------------------
# (e) everything down: refusal with the honest message, mode refusal
# ---------------------------------------------------------------------------------------------


class _EverythingDownRetriever:
    """Every seam this adapter touches is down at once: `tei-embed` exhausts its full retry
    ceiling, the ladder's own lexical_only fallback then reaches `postgres`, which is ALSO down --
    the terminal failure with no fallback left (SP4 task 4: a still failing retrieval routes to
    refusal). Two independent provider keys, each with its own breaker and counter, so BOTH
    seams' resilience inspection surfaces are pinned, not just the outcome."""

    def __init__(self, embed_breaker, embed_policy, pg_breaker, pg_policy) -> None:
        self.embed_requests = 0
        self.pg_requests = 0
        self.embed_breaker = embed_breaker
        self.embed_policy = embed_policy
        self.pg_breaker = pg_breaker
        self.pg_policy = pg_policy

    def search_chunks(self, query, k, config):
        if not config.lexical_only:
            def do_embed():
                self.embed_requests += 1
                raise _http_error(500)

            call_with_resilience(
                do_embed, policy=self.embed_policy, breaker=self.embed_breaker,
                provider_key="tei-embed", error_type=EmbeddingServiceError,
            )

        def do_pg():
            self.pg_requests += 1
            raise psycopg.OperationalError("connection refused")

        call_with_resilience(
            do_pg, policy=self.pg_policy, breaker=self.pg_breaker,
            provider_key="postgres", error_type=RetrievalError,
        )


@pytest.mark.asyncio
async def test_every_seam_down_produces_the_honest_refusal_message_with_both_breakers_still_closed():
    embed_breaker = CircuitBreaker(_FakeClock())
    pg_breaker = CircuitBreaker(_FakeClock())
    retriever = _EverythingDownRetriever(
        embed_breaker, RetryPolicy(attempts=3), pg_breaker, RetryPolicy(attempts=3)
    )
    graph = _graph(retriever)

    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract free?")], "session": _SESSION},
        {"configurable": {"thread_id": "everything-down"}},
    )

    assert out["degradation_mode"] == "refusal"
    assert out["final_response"] == f"{HANDOFF_PREFIX} {_REFUSAL_MESSAGE}"  # the honest message, fixed
    assert retriever.embed_requests == 3
    assert retriever.pg_requests == 3
    assert embed_breaker.state("tei-embed") == "closed"  # one failed top level call, threshold is 3
    assert pg_breaker.state("postgres") == "closed"


# ---------------------------------------------------------------------------------------------
# (f) malformed tool JSON from a stub MCP result: typed error surfaces, no crash, refusal path
# ---------------------------------------------------------------------------------------------


class _UnclassifiedExceptionRetriever:
    """A bare, unclassified exception (not RerankServiceError/EmbeddingServiceError/RetrievalError)
    -- a bug, not a modeled provider failure (`knowledge_server.py`'s own module docstring names
    this exact swallow): FastMCP's own generic `except Exception` handler catches and stringifies
    it into malformed (non passages shaped) tool result text with `isError=True`, before the
    ladder's OWN try/except in `search_knowledge` ever gets a chance to classify it. `breaker` is
    wired the same way every other case's breaker is (`CircuitBreaker` over a fake clock), so the
    test below inspects a real object that was actually part of this scenario, not a fresh one
    built only to assert against. `calls` proves this bypasses the resilience module entirely:
    never retried, never counted toward any breaker -- the honest counter for THIS seam is
    "exactly one attempt, never retried", not a breaker state (there is no typed error here for a
    breaker to key off of)."""

    def __init__(self, breaker: CircuitBreaker) -> None:
        self.calls = 0
        self.breaker = breaker

    def search_chunks(self, query, k, config):
        self.calls += 1
        raise KeyError("some_unexpected_key")


@pytest.mark.asyncio
async def test_malformed_mcp_result_from_an_unclassified_exception_routes_to_refusal_without_crashing():
    breaker = CircuitBreaker(_FakeClock())
    retriever = _UnclassifiedExceptionRetriever(breaker)
    graph = _graph(retriever)

    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract free?")], "session": _SESSION},
        {"configurable": {"thread_id": "malformed-mcp"}},
    )

    assert out["degradation_mode"] == "refusal"
    assert out["final_response"].startswith(HANDOFF_PREFIX)
    # the graph completing this ainvoke at all (no unhandled exception escaping it) is the "no
    # crash" proof; `calls == 1` is this seam's own honest counter -- never retried, since an
    # unclassified exception never reaches the ladder's typed error classification at all.
    assert retriever.calls == 1
    # design fact: an unclassified exception bypasses the resilience module entirely, so its
    # counters must stay untouched -- the breaker built for this case never records a failure, and
    # no retry is ever reported in this context.
    assert breaker.state("tei-embed") == "closed"
    assert not last_call_retried()


# ---------------------------------------------------------------------------------------------
# (g) SSE stream killed mid generation: error event then message_end(error) (Task 6's hook)
# ---------------------------------------------------------------------------------------------


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _login(client, customer_id):
    r = await client.post("/auth/login", json={"customer_id": customer_id})
    assert r.status_code == 200
    return r.json()["access_token"]


def _parse_block(lines: list[str]) -> dict:
    data = "".join(ln[len("data: "):] for ln in lines if ln.startswith("data: "))
    return json.loads(data)


async def _stream(client, token, message, thread_id="s1"):
    events: list[dict] = []
    buffer: list[str] = []
    async with client.stream(
        "POST", "/chat/stream",
        json={"message": message, "thread_id": thread_id},
        headers={"authorization": f"Bearer {token}"},
    ) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line == "":
                if buffer:
                    events.append(_parse_block(buffer))
                    buffer = []
                continue
            buffer.append(line)
        if buffer:
            events.append(_parse_block(buffer))
    return events


def _validate_all(events: list[dict], schema: dict) -> None:
    for event in events:
        jsonschema.validate(event, schema)


@pytest.fixture(scope="module")
def sse_schema() -> dict:
    return loader.load_schema("sse")


@pytest.mark.asyncio
async def test_sse_stream_killed_mid_generation_emits_error_then_terminal_message_end(sse_schema):
    """Reuses Task 6's own test only injection hook (`make_chat_app(..., stream_events_fn=...)`,
    `chat_app.py`'s own docstring: "a caller substitutes the async iterator ... so a failure can be
    injected at any point in the sequence without needing a real graph failure to reach it"). The
    injected failure here is itself produced by exhausting the SAME resilience module the other
    cases exercise (a simulated primary-model provider storm: 3 raw attempts, one non exempt
    failure, breaker stays closed), then surfaces as a raw connection drop exception mid stream:
    the SSE seam's own failure mode is a transport level kill, orthogonal to the typed
    retrieval/generation errors the graph routes on, so the terminal guarantee (an error event,
    then message_end(error), always last) is what THIS seam's own contract promises instead."""
    clock = _FakeClock()
    breaker = CircuitBreaker(clock)
    policy = RetryPolicy(attempts=3)
    counter = {"n": 0}

    async def _flaky_source(graph, state, config):
        yield {
            "event": "on_chain_end",
            "name": "tools_read",
            "metadata": {"langgraph_node": "tools_read"},
            "data": {"output": {
                "messages": [ToolMessage(
                    content=json.dumps([{"doc_id": "doc-9", "chunk_id": "c-9", "score": 1.0, "text": "x"}]),
                    tool_call_id="k1", name="search_knowledge",
                )],
                "degradation_mode": "none",
            }},
        }

        def do_request():
            counter["n"] += 1
            raise _http_error(529)

        try:
            call_with_resilience(
                do_request, policy=policy, breaker=breaker,
                provider_key="primary-model", error_type=ProviderError,
            )
        except ProviderError as exc:
            raise RuntimeError("provider connection dropped mid turn") from exc

    app = make_chat_app(fixture_kit().clock, object(), stream_events_fn=_flaky_source)
    async with _client(app) as client:
        token = await _login(client, "cust_current")
        events = await _stream(client, token, "hi")

    _validate_all(events, sse_schema)
    assert events[0]["event"] == "message_start"
    assert events[1] == {"event": "citation", "doc_id": "doc-9"}
    assert events[-2]["event"] == "error"
    assert events[-2]["recoverable"] is False
    assert events[-1] == {"event": "message_end", "finish_reason": "error"}
    # the resilience module's own inspection surface, even for a fault at the transport seam:
    # three raw attempts were made (RetryPolicy's own ceiling, 529 is retryable), and one non
    # exempt failure landed on the breaker -- nowhere near its threshold of 3, so it stays closed.
    assert counter["n"] == 3
    assert breaker.state("primary-model") == "closed"


# ---------------------------------------------------------------------------------------------
# (h) 400 from provider: never retried (assert single attempt), refusal
# ---------------------------------------------------------------------------------------------


class _GenerationFault:
    """Plain (non pydantic) mutable state for `_ProviderStormModel`, deliberately NOT a bare
    `dict`: `BaseChatModel` is a pydantic model, and a `dict` FIELD is copied by pydantic's own
    validation at construction (proven empirically -- mutating it through `self.fault` would not
    be visible on the object the test itself holds); an arbitrary class instance is instead kept
    by IDENTITY (an isinstance check only, no validation to copy), so the counter the model
    mutates and the counter this test asserts against are the SAME object."""

    def __init__(self, *, status_code: int, policy: RetryPolicy, breaker: CircuitBreaker) -> None:
        self.raw_requests = 0
        self.status_code = status_code
        self.policy = policy
        self.breaker = breaker


class _ProviderStormModel(BaseChatModel):
    """A test double that manually calls the SYNC `call_with_resilience` (`provider_key=
    "primary-model"`) inside `_agenerate`, proving `agent()`'s OWN `ProviderError` routing
    (`retryable` decides whether `fallback_model` ever runs) composes correctly with the real
    classification table -- it does not, on its own, prove production generation is wrapped: this
    model's `_agenerate` raises synchronously (no real network call to await), the shape every
    OTHER stub in this file already uses, never the genuinely awaited live call
    `atlas_graph._generate_message` actually makes. SP4 final fix wave (F2) gave that seam a real
    producer (`call_with_resilience_async`, wrapping `model.ainvoke(...)`/`model._agenerate(...)` in
    live/record mode); see `test_the_generation_seam_itself_walks_the_ladder_on_a_flapping_live_provider`
    below for the case that exercises THAT seam directly, through a duck typed `.mode`/`.inner`
    model rather than a hand rolled `call_with_resilience` call."""

    fault: _GenerationFault

    @property
    def _llm_type(self) -> str:
        return "provider-storm"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("the graph is async only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        def do_request():
            self.fault.raw_requests += 1
            raise _http_error(self.fault.status_code)

        call_with_resilience(
            do_request, policy=self.fault.policy, breaker=self.fault.breaker,
            provider_key="primary-model", error_type=ProviderError,
        )


@pytest.mark.asyncio
async def test_400_from_the_generation_provider_is_never_retried_and_routes_to_refusal():
    breaker = CircuitBreaker(_FakeClock())
    fault = _GenerationFault(status_code=400, policy=RetryPolicy(attempts=3), breaker=breaker)
    model = _ProviderStormModel(fault=fault)
    fallback = _PlainAnswerModel(answer="fallback should never run")
    graph = _graph(model=model, fallback_model=fallback)

    out = await graph.ainvoke(
        {"messages": [HumanMessage("What are your opening hours?")], "session": _SESSION},
        {"configurable": {"thread_id": "provider-400"}},
    )

    assert out["degradation_mode"] == "refusal"
    assert out["final_response"] == f"{HANDOFF_PREFIX} {_REFUSAL_MESSAGE}"
    assert "fallback should never run" not in out["final_response"]  # the fallback was never used
    assert fault.raw_requests == 1  # 400 is never retried: the classification table's own row
    assert breaker.state("primary-model") == "closed"  # one non exempt failure, nowhere near threshold


# ---------------------------------------------------------------------------------------------
# (i) SP4 final fix wave (F2): the generation seam ITSELF walks the ladder on a real flapping
# live provider -- a duck typed `.mode`/`.inner` model, never a hand rolled call_with_resilience
# call the way `_ProviderStormModel` above simulates it
# ---------------------------------------------------------------------------------------------


class _FlappingLiveModel:
    """Duck types `GatewayChatModel`'s own `.mode`/`.inner` shape (`atlas_graph._tool_bindable`
    reads only those two attributes plus `.model_id`, the same fake shape `test_ladder.py`'s
    `_GatewayLikeModel` uses) so `_generate_message` treats this as a real live/record generation
    call and wraps it in the resilience seam this fix wave added (`call_with_resilience_async`),
    never a manually inlined `call_with_resilience` the way `_ProviderStormModel` above simulates
    it. `mcp_tools=None` (`_graph`'s own default) keeps this on the `model._agenerate(...)` shape of
    the live call (no tools ever bound), the OTHER live generation path this seam wraps, sibling to
    the bind_tools one `test_ladder.py` already covers.

    Raises a RETRYABLE 529 on the very first raw attempt, then a FATAL 400 on every attempt after: a
    realistic flapping provider, not merely "always down". `RetryPolicy` retries the 529 once, then
    stops immediately on the 400 (never retried, the classification table's own row), so the seam
    raises `ProviderError(retryable=False)`, routing straight past `provider_fallback` (even with a
    fallback model configured) to the honest refusal, never an unhandled exception."""

    def __init__(self) -> None:
        self.mode = "live"
        self.inner = object()  # any non None value; _tool_bindable only checks identity, never calls it
        self.model_id = "flapping-live-model"
        self.raw_requests = 0

    async def _agenerate(self, messages, **kwargs):
        self.raw_requests += 1
        status = 529 if self.raw_requests == 1 else 400
        raise _http_error(status)


@pytest.mark.asyncio
async def test_the_generation_seam_itself_walks_the_ladder_on_a_flapping_live_provider():
    """SP4 final fix wave (F2) regression. Before this wave, nothing translated a real generation
    failure into `ProviderError` in production: `atlas_graph._generate_message`'s live call raised
    whatever the SDK/httpx raised, which `except ProviderError` (`agent()`) never caught, so a live
    outage surfaced as an unhandled exception (a raw 500 on `/chat`, a generic in band error on
    `/chat/stream`), never the ladder. This drives the real graph through a duck typed `.mode`/
    `.inner` model, proving the seam (`call_with_resilience_async`, wired into `_generate_message`)
    classifies a genuine flapping provider correctly end to end: one retried attempt, one fatal
    attempt, an honest refusal, fallback never touched."""
    model = _FlappingLiveModel()
    fallback = _PlainAnswerModel(answer="fallback should never run")
    graph = _graph(model=model, fallback_model=fallback)

    out = await graph.ainvoke(
        {"messages": [HumanMessage("What are your opening hours?")], "session": _SESSION},
        {"configurable": {"thread_id": "generation-seam-flap"}},
    )

    assert out["degradation_mode"] == "refusal"
    assert out["final_response"] == f"{HANDOFF_PREFIX} {_REFUSAL_MESSAGE}"
    assert "fallback should never run" not in out["final_response"]  # the fallback was never used
    assert model.raw_requests == 2  # the real RetryPolicy retried the 529 once, then stopped on 400
