"""The degradation ladder (SP4 task 4), hermetic: the pure domain constant + ordering property, the
MCP knowledge server's own ladder walk (a stub retriever per typed error from `resilience.py`), and
the graph's end to end wiring -- state carried mode, the `refusal` node, `provider_fallback` for
generation, and the ordering invariant (a lower rung never overwrites a higher one already recorded
this turn). No Docker, no network, no real Postgres/TEI/provider anywhere in this file.
"""
from __future__ import annotations

import itertools
import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from mcp.shared.memory import create_connected_server_and_client_session

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory

from atlas.adapters.resilience import EmbeddingServiceError, ProviderError, RerankServiceError, RetrievalError
from atlas.domain.actions import ActionsBackend
from atlas.domain.degradation import DEGRADATION_LADDER, DEGRADATION_MODE_NONE, DEGRADED_RESULT_KEY, escalate
from atlas.domain.retrieval import RetrievalConfig
from atlas.mcp_servers.knowledge_server import build_knowledge_server
from atlas.orchestration.atlas_graph import HANDOFF_PREFIX, build_atlas_graph
from atlas.ports.knowledge import Chunk

# ---------------------------------------------------------------------------------------------
# pure domain: the ladder constant + escalate() ordering property
# ---------------------------------------------------------------------------------------------


def test_ladder_matches_the_trace_contract_order():
    # contracts/trace/schema.json's atlas.degradation.mode / contracts/sse/schema.json's
    # degradation event: "none, retry, provider_fallback, drop_rerank, lexical_only, refusal",
    # "none" dropped since a transition never escalates TO it.
    assert DEGRADATION_LADDER == ("retry", "provider_fallback", "drop_rerank", "lexical_only", "refusal")


def test_escalate_from_none_always_adopts_the_candidate_rung():
    for rung in DEGRADATION_LADDER:
        assert escalate(DEGRADATION_MODE_NONE, rung) == rung


def test_escalate_never_lets_a_lower_rung_overwrite_a_higher_one():
    # property: for every ordered pair on the ladder, escalating a higher rung with a lower
    # candidate stays put; escalating a lower rung with a higher candidate moves.
    for lower, higher in itertools.combinations(DEGRADATION_LADDER, 2):
        assert escalate(higher, lower) == higher
        assert escalate(lower, higher) == higher


def test_escalate_is_idempotent_on_the_same_rung():
    for rung in DEGRADATION_LADDER:
        assert escalate(rung, rung) == rung


def test_escalate_treats_an_unranked_string_as_ranking_below_none():
    # fail closed the same way an unrecognized status code defaults to never retried: a typo'd
    # mode can neither win against a real rung nor silently downgrade one.
    assert escalate("drop_rerank", "not-a-real-rung") == "drop_rerank"
    assert escalate("not-a-real-rung", "retry") == "retry"


# ---------------------------------------------------------------------------------------------
# the knowledge MCP server's own ladder walk (stub retrievers, one typed error per rung)
# ---------------------------------------------------------------------------------------------


class _RerankDownRetriever:
    """Rerank is down; the SAME query succeeds once the caller disables it."""

    def search_chunks(self, query, k, config):
        if config.rerank_enabled:
            raise RerankServiceError("tei-rerank down", provider_key="tei-rerank")
        return [Chunk(doc_id="doc-1", chunk_id="chunk-1", score=0.5, text="rerank-off answer")][:k]


class _EmbeddingDownRetriever:
    """Embedding is down; the same query succeeds once the caller asks lexical only."""

    def search_chunks(self, query, k, config):
        if not config.lexical_only:
            raise EmbeddingServiceError("tei-embed down", provider_key="tei-embed")
        return [Chunk(doc_id="doc-1", chunk_id="chunk-1", score=0.5, text="lexical-only answer")][:k]


class _AlwaysDownRetriever:
    """Every ladder rung exhausted: postgres itself is down, no config change saves it."""

    def search_chunks(self, query, k, config):
        raise RetrievalError("postgres down", provider_key="postgres")


class _UnclassifiedExceptionRetriever:
    """Raises something that is NOT one of the three typed errors `search_knowledge` recognizes
    (`RerankServiceError`/`EmbeddingServiceError`/`RetrievalError`) -- a bug, not a modeled failure.
    knowledge_server.py's own module docstring names what happens next: FastMCP's `call_tool`
    handler catches it, stringifies it, and sets `isError=True`; nothing about the exception's TYPE
    survives that trip. This is the isError backstop's own test case (SP4 task 5 ride along)."""

    def search_chunks(self, query, k, config):
        raise KeyError("some_unexpected_key")


class _RerankThenAlsoDownRetriever:
    """The fallback attempt ALSO fails (a second, distinct typed error): the ladder has nothing
    left to try, so this must land on refusal too, not loop forever."""

    def search_chunks(self, query, k, config):
        if config.rerank_enabled:
            raise RerankServiceError("tei-rerank down", provider_key="tei-rerank")
        raise EmbeddingServiceError("tei-embed also down", provider_key="tei-embed")


class _FakeSearchResult:
    def __init__(self, *, retried: bool) -> None:
        self.retried = retried


class _RetriedRetriever:
    """Never raises; reports (via the SAME duck typed `last_result()` accessor
    `PgvectorRetriever` exposes) that the resilience layer needed a retry to succeed."""

    def __init__(self) -> None:
        self._result: _FakeSearchResult | None = None

    def search_chunks(self, query, k, config):
        chunks = [Chunk(doc_id="doc-1", chunk_id="chunk-1", score=0.5, text="retried answer")][:k]
        self._result = _FakeSearchResult(retried=True)
        return chunks

    def last_result(self):
        return self._result


async def _search(retriever, query: str = "q") -> str:
    server = build_knowledge_server(retriever)
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        res = await client.call_tool("search_knowledge", {"query": query})
        return res.content[0].text


@pytest.mark.asyncio
async def test_rerank_down_retries_with_rerank_disabled_and_reports_drop_rerank():
    parsed = json.loads(await _search(_RerankDownRetriever()))
    assert parsed[DEGRADED_RESULT_KEY] is True
    assert parsed["degradation_mode"] == "drop_rerank"
    # SP4 task 5: chunk_id/score are additive on every passage (knowledge_server._passages())
    assert parsed["passages"] == [{"doc_id": "doc-1", "chunk_id": "chunk-1", "score": "F:0.5", "text": "rerank-off answer"}]


@pytest.mark.asyncio
async def test_embedding_down_retries_lexical_only_and_reports_lexical_only():
    parsed = json.loads(await _search(_EmbeddingDownRetriever()))
    assert parsed["degradation_mode"] == "lexical_only"
    assert parsed["passages"] == [{"doc_id": "doc-1", "chunk_id": "chunk-1", "score": "F:0.5", "text": "lexical-only answer"}]


@pytest.mark.asyncio
async def test_retrieval_that_never_recovers_reports_refusal_with_no_passages():
    parsed = json.loads(await _search(_AlwaysDownRetriever()))
    assert parsed["degradation_mode"] == "refusal"
    assert parsed["passages"] == []


@pytest.mark.asyncio
async def test_a_second_distinct_failure_on_the_fallback_attempt_also_reports_refusal():
    parsed = json.loads(await _search(_RerankThenAlsoDownRetriever()))
    assert parsed["degradation_mode"] == "refusal"


@pytest.mark.asyncio
async def test_a_transparent_retry_reports_the_retry_rung():
    parsed = json.loads(await _search(_RetriedRetriever()))
    assert parsed["degradation_mode"] == "retry"
    assert parsed["passages"] == [{"doc_id": "doc-1", "chunk_id": "chunk-1", "score": "F:0.5", "text": "retried answer"}]


@pytest.mark.asyncio
async def test_knowledge_call_iserror_backstop_routes_to_refusal_never_composes_over_error_text():
    """SP4 task 5 ride along: `atlas_graph._knowledge_call`'s isError backstop. An unclassified
    exception (here, a bare KeyError -- not RerankServiceError/EmbeddingServiceError/RetrievalError)
    is stringified by FastMCP's own generic catch, isError=True, BEFORE `_knowledge_call` ever sees
    it; the fix checks `res.isError` first and routes straight to refusal, so the model is never
    handed the stringified exception text disguised as an ordinary (or even degraded) tool result."""
    from atlas.orchestration.atlas_graph import _knowledge_call

    text, mode = await _knowledge_call(_UnclassifiedExceptionRetriever(), "q")
    assert mode == "refusal"
    assert text == ""  # never the stringified KeyError treated as content


@pytest.mark.asyncio
async def test_account_call_iserror_backstop_routes_to_refusal_never_composes_over_error_text(monkeypatch):
    """SP4 task 5 fix round 1 (reviewer finding): the isError backstop generalized past
    `_knowledge_call` to `_account_call` too -- the exact failure mode the catalog routing gap this
    same task fixed elsewhere demonstrated live (a tool level exception silently becoming "Unknown
    tool" style text, treated as an ordinary tool result). `get_account` is monkeypatched on the
    NAME as imported into `account_server` (`from atlas.domain.accounts import get_account`), the
    same "patch where the name is bound, not where it is defined" discipline this repo's own test
    suite already follows elsewhere (`test_persistence.py`'s `open_postgres_checkpointer` patch, and
    its own comment on exactly this point)."""
    import atlas.mcp_servers.account_server as account_server_module
    from atlas.orchestration.atlas_graph import _account_call

    def _boom(customer_id: str):
        raise KeyError("unexpected")

    monkeypatch.setattr(account_server_module, "get_account", _boom)
    text, mode = await _account_call("cust_current", "get_account_summary", {})
    assert mode == "refusal"
    assert text == ""  # never the stringified KeyError treated as content


@pytest.mark.asyncio
async def test_catalog_call_iserror_backstop_routes_to_refusal_never_composes_over_error_text(monkeypatch):
    """SP4 task 5 fix round 1: `_catalog_call`'s own isError backstop. `catalog_server.py` accesses
    `catalog.get_plan` as a module attribute (`from atlas.domain import catalog`, not a
    `from ... import get_plan` name), so patching `atlas.domain.catalog.get_plan` directly is
    enough -- no "patch where imported" indirection needed here, unlike the account/actions cases."""
    from atlas.domain import catalog
    from atlas.orchestration.atlas_graph import _catalog_call

    def _boom(plan_id: str):
        raise KeyError("unexpected")

    monkeypatch.setattr(catalog, "get_plan", _boom)
    text, mode = await _catalog_call("get_plan", {"plan_id": "plan_current_fast"})
    assert mode == "refusal"
    assert text == ""  # never the stringified KeyError treated as content


@pytest.mark.asyncio
async def test_actions_call_iserror_backstop_routes_to_refusal_never_composes_over_error_text(monkeypatch):
    """SP4 task 5 fix round 1: `_actions_call`'s own isError backstop -- the write surface. Every
    `actions_server.py` tool ends with `serialize_tool_result(...)`; patching that name as imported
    into `actions_server` makes every tool raise, `reset_modem` (no args) picked here since it needs
    no extra setup."""
    import atlas.mcp_servers.actions_server as actions_server_module
    from atlas.orchestration.atlas_graph import _actions_call

    def _boom(payload):
        raise KeyError("unexpected")

    monkeypatch.setattr(actions_server_module, "serialize_tool_result", _boom)
    text, mode = await _actions_call("cust_current", "reset_modem", {})
    assert mode == "refusal"
    assert text == ""  # never the stringified KeyError treated as content


@pytest.mark.asyncio
async def test_the_ordinary_happy_path_is_byte_identical_to_before_the_ladder_existed():
    from atlas.adapters.inmemory_retriever import InMemoryRetriever
    from determinism.canonical import serialize_tool_result

    query = "contract-free plan"
    raw = await _search(InMemoryRetriever(), query)
    chunks = InMemoryRetriever().search_chunks(query, config=RetrievalConfig())
    # SP4 task 5: chunk_id/score are additive; still a bare passages array, no envelope.
    expected = serialize_tool_result(
        [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": c.score, "text": c.text} for c in chunks]
    )
    assert raw == expected  # a bare passages array, no envelope: untouched by the ladder


# ---------------------------------------------------------------------------------------------
# the graph, end to end: state carried degradation_mode, the refusal node, provider_fallback
# ---------------------------------------------------------------------------------------------


class _SearchOnceThenAnswerModel(BaseChatModel):
    """Emits one search_knowledge tool call, then a plain answer once it sees the ToolMessage --
    the same deterministic, cassette free pattern test_atlas_graph.py's `_StubModel` uses."""

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
            msg = AIMessage(content="", tool_calls=[{"name": "search_knowledge", "args": {"query": self.query}, "id": "k1"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _SearchThenProviderErrorModel(BaseChatModel):
    """Emits one search_knowledge tool call, then a RETRYABLE ProviderError once it sees the
    ToolMessage: used to prove the ordering property (a retrieval rung already recorded must
    survive a later, lower generation rung)."""

    query: str = "contract question"

    @property
    def _llm_type(self) -> str:
        return "search-then-provider-error"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("the graph is async only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if any(isinstance(m, ToolMessage) for m in messages):
            raise ProviderError("primary down after retrieval", retryable=True, provider_key="primary-model")
        msg = AIMessage(content="", tool_calls=[{"name": "search_knowledge", "args": {"query": self.query}, "id": "k1"}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _AlwaysProviderErrorModel(BaseChatModel):
    """A permanently down generation provider, retryable per the constructor argument."""

    retryable: bool = False

    @property
    def _llm_type(self) -> str:
        return "always-provider-error"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("the graph is async only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise ProviderError("primary model down", retryable=self.retryable, provider_key="primary-model")


class _PlainAnswerModel(BaseChatModel):
    """A model (primary or fallback) that always answers with fixed content and never calls a
    tool -- used as the fallback in `provider_fallback` tests."""

    answer: str = "plain answer"

    @property
    def _llm_type(self) -> str:
        return "plain-answer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("the graph is async only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.answer))])


class _RecordingBindToolsInner:
    """A fake mimicking a real provider model's `.bind_tools(...)` -> Runnable chain (SP4 task 5):
    `.bind_tools()` records exactly what it was called with (so a test can assert the intent scoped
    surface `_bound_tool_specs` computed) and returns itself, whose `.ainvoke()` answers like
    `_SearchOnceThenAnswerModel` above: one search_knowledge tool call on the first turn, a plain
    answer once a ToolMessage is present in the history."""

    def __init__(self, *, answer: str = "bound answer") -> None:
        self.bound_tools: list[dict] | None = None
        self._answer = answer

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    async def ainvoke(self, messages):
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content=self._answer)
        return AIMessage(content="", tool_calls=[{"name": "search_knowledge", "args": {"query": "q"}, "id": "k1"}])


class _ExplodingBindToolsInner:
    """`.bind_tools()` explodes if ever called: the explode pattern proof that replay/hermetic mode
    never reaches real tool binding, even when a live `.inner` happens to be present (SP4 task 5)."""

    def bind_tools(self, tools):
        raise AssertionError("bind_tools must never be called in replay/hermetic mode")


class _ExplodingBindToolsChatModel(BaseChatModel):
    """SP4 task 5 fix round 1: a REAL `BaseChatModel` (unlike `_ExplodingBindToolsInner`, a plain
    duck typed stand in) whose `bind_tools` explodes if ever called. Used as a REAL
    `GatewayChatModel`'s own `.inner`, so the replay mode gate is proven against the actual
    `GatewayMode` enum `atlas_graph._tool_bindable` reads in production, not only against the
    fakes' plain string `.mode` attribute above (which happened to hide a real bug: see
    `test_replay_mode_gate_is_load_bearing_against_a_real_gateway_chat_model`)."""

    @property
    def _llm_type(self) -> str:
        return "exploding-bind-tools"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("the graph is async only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise NotImplementedError("bind_tools should have exploded before this was ever reached")

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        raise AssertionError("bind_tools must never be called in replay/hermetic mode")


class _GatewayLikeModel:
    """Mimics `GatewayChatModel`'s own duck typed shape (`.mode`, `.inner`,
    `testing/harness/replay/gateway.py`) WITHOUT importing that harness class here, and WITHOUT
    subclassing `BaseChatModel`: `atlas_graph._generate_message` only ever reads `.mode`/`.inner`
    off `model`, and calls either `.inner.bind_tools(...).ainvoke(...)` (tool bound path) or
    `model._agenerate(...)` (the unbound, byte identical fallback) -- nothing else on `model`
    itself, so a plain object suffices."""

    def __init__(self, *, mode: str, inner=None, answer: str = "hermetic answer, no tools bound") -> None:
        self.mode = mode
        self.inner = inner
        self.model_id = "gateway like"
        self._answer = answer

    async def _agenerate(self, messages, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._answer))])


def _graph(retriever=None, *, model=None, fallback_model=None, tracer=None, mcp_tools=None):
    return build_atlas_graph(
        model or _SearchOnceThenAnswerModel(),
        IdFactory("idem"),
        ActionsBackend(IdFactory("ref")),
        new_checkpointer(),
        retriever=retriever,
        tracer=tracer,
        fallback_model=fallback_model,
        mcp_tools=mcp_tools,
    )


_SESSION = {"customer_id": "cust_current"}


@pytest.mark.asyncio
async def test_graph_stamps_drop_rerank_when_rerank_is_down_and_still_answers():
    graph = _graph(_RerankDownRetriever())
    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract-free?")], "session": _SESSION},
        {"configurable": {"thread_id": "ladder-drop-rerank"}},
    )
    assert out["degradation_mode"] == "drop_rerank"
    assert out["final_response"] == "Here is what I found."


@pytest.mark.asyncio
async def test_graph_stamps_lexical_only_when_embedding_is_down_and_still_answers():
    graph = _graph(_EmbeddingDownRetriever())
    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract-free?")], "session": _SESSION},
        {"configurable": {"thread_id": "ladder-lexical-only"}},
    )
    assert out["degradation_mode"] == "lexical_only"
    assert out["final_response"] == "Here is what I found."


@pytest.mark.asyncio
async def test_graph_stamps_retry_when_the_resilience_layer_recovered_transparently():
    graph = _graph(_RetriedRetriever())
    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract-free?")], "session": _SESSION},
        {"configurable": {"thread_id": "ladder-retry"}},
    )
    assert out["degradation_mode"] == "retry"
    assert out["final_response"] == "Here is what I found."


@pytest.mark.asyncio
async def test_graph_routes_to_refusal_when_retrieval_is_exhausted():
    graph = _graph(_AlwaysDownRetriever())
    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract-free?")], "session": _SESSION},
        {"configurable": {"thread_id": "ladder-refusal"}},
    )
    assert out["degradation_mode"] == "refusal"
    assert out["final_response"].startswith(HANDOFF_PREFIX)


@pytest.mark.asyncio
async def test_a_refusal_still_answers_every_pending_tool_call_so_history_stays_well_formed():
    graph = _graph(_AlwaysDownRetriever())
    cfg = {"configurable": {"thread_id": "ladder-history"}}
    await graph.ainvoke({"messages": [HumanMessage("Is my plan contract-free?")], "session": _SESSION}, cfg)

    snapshot = await graph.aget_state(cfg)
    messages = snapshot.values["messages"]
    last_ai = next(m for m in reversed(messages) if isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    assert {tc["id"] for tc in last_ai.tool_calls} <= answered


@pytest.mark.asyncio
async def test_the_undegraded_happy_path_never_sets_needs_refusal_or_touches_the_mode():
    """Regression pin: InMemoryRetriever (the hermetic default) never raises a typed error and
    never reports a retry, so a RAG turn over it must finish with mode "none", exactly the
    pre ladder behaviour this task must not disturb."""
    from atlas.adapters.inmemory_retriever import InMemoryRetriever

    graph = _graph(InMemoryRetriever())
    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract-free?")], "session": _SESSION},
        {"configurable": {"thread_id": "ladder-happy"}},
    )
    assert out["degradation_mode"] == "none"
    assert out["final_response"] == "Here is what I found."


# --- generation side: ProviderError, provider_fallback -------------------------------------------


@pytest.mark.asyncio
async def test_generation_retryable_provider_error_falls_over_to_the_fallback_model():
    graph = _graph(
        model=_AlwaysProviderErrorModel(retryable=True),
        fallback_model=_PlainAnswerModel(answer="fallback answered this turn"),
    )
    out = await graph.ainvoke(
        {"messages": [HumanMessage("What are your opening hours?")], "session": _SESSION},
        {"configurable": {"thread_id": "ladder-provider-fallback"}},
    )
    assert out["degradation_mode"] == "provider_fallback"
    assert out["final_response"] == "fallback answered this turn"


@pytest.mark.asyncio
async def test_generation_non_retryable_provider_error_routes_to_refusal_even_with_a_fallback_configured():
    graph = _graph(
        model=_AlwaysProviderErrorModel(retryable=False),
        fallback_model=_PlainAnswerModel(answer="should never be used"),
    )
    out = await graph.ainvoke(
        {"messages": [HumanMessage("What are your opening hours?")], "session": _SESSION},
        {"configurable": {"thread_id": "ladder-non-retryable"}},
    )
    assert out["degradation_mode"] == "refusal"
    assert out["final_response"].startswith(HANDOFF_PREFIX)
    assert "should never be used" not in out["final_response"]  # single attempt: never retried


@pytest.mark.asyncio
async def test_generation_retryable_provider_error_with_no_fallback_configured_routes_to_refusal():
    graph = _graph(model=_AlwaysProviderErrorModel(retryable=True), fallback_model=None)
    out = await graph.ainvoke(
        {"messages": [HumanMessage("What are your opening hours?")], "session": _SESSION},
        {"configurable": {"thread_id": "ladder-retryable-no-fallback"}},
    )
    assert out["degradation_mode"] == "refusal"
    assert out["final_response"].startswith(HANDOFF_PREFIX)


@pytest.mark.asyncio
async def test_generation_fallback_that_also_fails_routes_to_refusal():
    graph = _graph(
        model=_AlwaysProviderErrorModel(retryable=True),
        fallback_model=_AlwaysProviderErrorModel(retryable=True),
    )
    out = await graph.ainvoke(
        {"messages": [HumanMessage("What are your opening hours?")], "session": _SESSION},
        {"configurable": {"thread_id": "ladder-fallback-also-fails"}},
    )
    assert out["degradation_mode"] == "refusal"
    assert out["final_response"].startswith(HANDOFF_PREFIX)


# --- the ordering property, end to end: a higher rung already recorded is never downgraded ------


@pytest.mark.asyncio
async def test_a_higher_rung_already_recorded_this_turn_is_never_overwritten_by_a_later_lower_one():
    """drop_rerank (retrieval) outranks provider_fallback (generation): this turn hits drop_rerank
    FIRST (the search), then provider_fallback fires SECOND (composing the answer). The final mode
    must stay drop_rerank -- a chronologically later but lower ranked event must not downgrade it."""
    graph = _graph(
        _RerankDownRetriever(),
        model=_SearchThenProviderErrorModel(),
        fallback_model=_PlainAnswerModel(answer="fallback text, benign"),
    )
    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract-free?")], "session": _SESSION},
        {"configurable": {"thread_id": "ladder-ordering"}},
    )
    assert out["degradation_mode"] == "drop_rerank"


# --- SP4 task 5: bind_tools wiring, hermetic (no live provider anywhere in this file either) -----


@pytest.mark.asyncio
async def test_replay_mode_never_calls_bind_tools_even_with_mcp_tools_and_a_live_inner_present():
    """The explode pattern (mirrors SP3/SP4's own "monkeypatch that explodes" idiom): `mode="replay"`
    must gate off real tool binding even when a live `.inner` happens to be present (not merely when
    `.inner is None`) and `mcp_tools` was supplied -- the MODE check, not just the `.inner is None`
    check, is what keeps the hermetic path byte identical to before this task."""
    from atlas.mcp_servers.tool_surface import mcp_tool_surface

    model = _GatewayLikeModel(mode="replay", inner=_ExplodingBindToolsInner())
    graph = _graph(model=model, mcp_tools=mcp_tool_surface())
    out = await graph.ainvoke(
        {"messages": [HumanMessage("What are your opening hours?")], "session": _SESSION},
        {"configurable": {"thread_id": "bind-tools-replay-gate"}},
    )
    assert out["final_response"] == "hermetic answer, no tools bound"  # _agenerate's own path, never bound


@pytest.mark.asyncio
async def test_replay_mode_gate_is_load_bearing_against_a_real_gateway_chat_model(tmp_path, seed_cassette):
    """SP4 task 5 fix round 1 (reviewer finding): the test above uses `_GatewayLikeModel`, a plain
    object whose `.mode` is a literal Python string ("replay"), so it could not have caught a bug in
    how `_tool_bindable` reads a REAL `GatewayMode` enum. This test constructs an ACTUAL
    `GatewayChatModel` (`replay/gateway.py`) with `mode="replay"` (pydantic coerces this to
    `GatewayMode.REPLAY`) AND a live `.inner` explicitly set -- nothing in `GatewayChatModel`'s own
    `_check_wiring` forbids that combination (it only REQUIRES `.inner` for RECORD/LIVE, it never
    forbids it for REPLAY); production `_gateway()` never constructs one this way, but the type
    system allows it, and `.inner`'s own `bind_tools` must still never be called. This is the ONLY
    thing that makes the mode check (as opposed to the `.inner is None` check) load bearing: before
    the fix, `_tool_bindable` called `str(GatewayMode.REPLAY)`, which is `"GatewayMode.REPLAY"`
    (Enum's own `__str__` wins over the `str` mixin for `str()`, a well known enum gotcha), never
    equal to `"replay"`, so the mode gate silently never fired here."""
    from atlas.mcp_servers.tool_surface import mcp_tool_surface
    from replay.gateway import GatewayChatModel

    seed_cassette(
        tmp_path, [HumanMessage("What are your opening hours?")],
        {"content": "hours answer", "tool_calls": []},
    )
    model = GatewayChatModel(
        model_id="claude-test", cassette_dir=tmp_path, mode="replay", inner=_ExplodingBindToolsChatModel()
    )
    graph = _graph(model=model, mcp_tools=mcp_tool_surface())
    out = await graph.ainvoke(
        {"messages": [HumanMessage("What are your opening hours?")], "session": _SESSION},
        {"configurable": {"thread_id": "bind-tools-real-gateway-replay-gate"}},
    )
    assert out["final_response"] == "hours answer"


@pytest.mark.asyncio
async def test_no_mcp_tools_never_calls_bind_tools_even_in_live_mode():
    """Symmetric to the test above: a live `.inner` in a non replay mode, but `mcp_tools=None` (every
    hermetic caller's own default) -- still never binds. `build_atlas_graph`'s default keeps every
    EXISTING caller unaffected, this is that default's own explode proof."""
    model = _GatewayLikeModel(mode="live", inner=_ExplodingBindToolsInner())
    graph = _graph(model=model, mcp_tools=None)
    out = await graph.ainvoke(
        {"messages": [HumanMessage("What are your opening hours?")], "session": _SESSION},
        {"configurable": {"thread_id": "bind-tools-no-surface"}},
    )
    assert out["final_response"] == "hermetic answer, no tools bound"


@pytest.mark.asyncio
async def test_live_mode_binds_the_intent_scoped_mcp_tool_surface_and_uses_its_real_tool_calls():
    """mode != "replay" with a live `.inner` and a non empty tool surface DOES bind real tools,
    scoped to exactly what `domain.binding.bound_tools(intent)` already declares reachable for THIS
    turn's intent -- proving `domain/binding.py`'s own claim ("a dev/prod build would also hand the
    model only the bound tools, so the capability is simply absent") is now code, not just a comment
    left for a later task."""
    from atlas.domain.binding import bound_tools
    from atlas.mcp_servers.tool_surface import mcp_tool_surface

    inner = _RecordingBindToolsInner(answer="live answer, real tool call happened")
    model = _GatewayLikeModel(mode="live", inner=inner)
    graph = _graph(model=model, mcp_tools=mcp_tool_surface())
    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract-free?")], "session": _SESSION},
        {"configurable": {"thread_id": "bind-tools-live"}},
    )
    assert out["final_response"] == "live answer, real tool call happened"
    assert inner.bound_tools is not None  # bind_tools really was called
    # a plain question with no action cue classifies "troubleshooting" (classify_intent's own binary
    # split): only knowledge + read tools were ever handed to the model, catalog and every write
    # tool withheld -- least agency enforced at BIND time, not only by the downstream intercept.
    bound_names = {t["name"] for t in inner.bound_tools}
    assert bound_names == bound_tools("troubleshooting")
    assert "list_plans" not in bound_names
    assert "change_plan" not in bound_names


@pytest.mark.asyncio
async def test_record_mode_also_binds_tools_not_only_live():
    """`_tool_bindable` gates on "not replay", not on a specific non replay value: record mode binds
    exactly like live mode (D19 defers only whether a record mode cassette CAPTURES the tool bound
    exchange, never whether the model gets real tools to call)."""
    inner = _RecordingBindToolsInner(answer="recorded answer")
    model = _GatewayLikeModel(mode="record", inner=inner)
    from atlas.mcp_servers.tool_surface import mcp_tool_surface

    graph = _graph(model=model, mcp_tools=mcp_tool_surface())
    out = await graph.ainvoke(
        {"messages": [HumanMessage("Is my plan contract-free?")], "session": _SESSION},
        {"configurable": {"thread_id": "bind-tools-record"}},
    )
    assert out["final_response"] == "recorded answer"
    assert inner.bound_tools is not None
