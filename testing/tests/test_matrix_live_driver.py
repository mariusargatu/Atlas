"""`matrix.live_driver`, hermetic (SP9 task 4's deferred live caller): every ASSEMBLY function here
is proven with an injected stub retriever/reranker/embedding client/chat model -- never a real
network call, never a real key. The live construction helpers (`construct_live_pgvector_retriever`,
`construct_live_tei_reranker`, `construct_live_openai_embedding_client`, and the private
`_live_anthropic_model`/`_live_gpt_model`/`_live_ollama_model`) are reached ONLY when a caller omits
its own injected dependency; every test below always injects one, so none of those helpers' own
live bodies ever run here -- except the worded-failure path, which is itself pure (raises before
any import/network attempt).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from atlas.domain.retrieval import RetrievalConfig
from atlas.ports.knowledge import Chunk
from replay.gateway import GatewayMode

from matrix.cases import MatrixCase
from matrix.embedders import BASELINE_COMPONENT_IDS, EmbedderComponent
from matrix.generators import GeneratorComponent
from matrix.rerankers import RerankerComponent
from matrix.spend_gate import ALWAYS_RUNS, SpendGate, check_spend

from matrix.live_driver import (
    MissingEnvVarError,
    build_baseline_embedder_components,
    build_bge_m3_embedder_component,
    build_claude_generator_component,
    build_gpt_generator_component,
    build_judge_gateway,
    build_openai_embedder_component,
    build_reranker_components,
    build_variants_config,
    estimate_generation_cost_usd,
    require_env,
)


class _StubProvider(BaseChatModel):
    """The smallest real `BaseChatModel` a gateway will accept (mirrors `test_matrix_ollama_
    generator.py`'s/`test_gateway.py`'s own `_StubProvider`)."""

    reply: str = "stub reply"

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


_CASE_A = MatrixCase("case-a", "how much is plan a", frozenset({"d1"}), ({"fact_id": "a:price", "value": "10"},))
_CASE_B = MatrixCase("case-b", "how much is plan b", frozenset({"d2"}), ({"fact_id": "b:price", "value": "20"},))
_CASES = (_CASE_A, _CASE_B)


# ---- require_env ----------------------------------------------------------------------------------


def test_require_env_returns_the_value_when_set(monkeypatch):
    monkeypatch.setenv("ATLAS_TEST_VAR", "hello")
    assert require_env("ATLAS_TEST_VAR", purpose="testing") == "hello"


def test_require_env_raises_a_worded_failure_when_unset(monkeypatch):
    monkeypatch.delenv("ATLAS_TEST_VAR", raising=False)
    with pytest.raises(MissingEnvVarError, match="ATLAS_TEST_VAR"):
        require_env("ATLAS_TEST_VAR", purpose="a very specific live call")


def test_require_env_message_names_the_purpose(monkeypatch):
    monkeypatch.delenv("ATLAS_TEST_VAR", raising=False)
    with pytest.raises(MissingEnvVarError, match="a very specific live call"):
        require_env("ATLAS_TEST_VAR", purpose="a very specific live call")


def test_require_env_treats_an_empty_string_as_unset(monkeypatch):
    monkeypatch.setenv("ATLAS_TEST_VAR", "")
    with pytest.raises(MissingEnvVarError):
        require_env("ATLAS_TEST_VAR", purpose="testing")


# ---- bge-m3 embedder component: dry assembly, no live call ----------------------------------------


class _StubRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, int, RetrievalConfig]] = []

    def search_chunks(self, query: str, k: int, config: RetrievalConfig) -> list[Chunk]:
        self.calls.append((query, k, config))
        return self._chunks[:k]


def test_build_bge_m3_embedder_component_wires_the_injected_retriever_with_rerank_disabled():
    chunks = [Chunk(chunk_id="d1", doc_id="doc-1", text="plan a costs 10")]
    retriever = _StubRetriever(chunks)
    component = build_bge_m3_embedder_component(pool_size=50, retriever=retriever)

    assert isinstance(component, EmbedderComponent)
    assert component.component_id == "bge-m3-local"
    assert component.embedding_model == {"id": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181"}

    result = component.search(_CASE_A)
    assert [c.chunk_id for c in result] == ["d1"]
    (query, k, config) = retriever.calls[0]
    assert query == _CASE_A.query
    assert k == 50
    assert config.rerank_enabled is False
    assert config.k_fused == 50
    assert config.k_final == 50


def test_build_bge_m3_embedder_component_reads_index_build_id_from_the_committed_index():
    component = build_bge_m3_embedder_component(pool_size=20, retriever=_StubRetriever([]))
    assert component.index_build_id == "a86bc176d5bf7d04"  # the real committed build_manifest.json


def test_build_bge_m3_embedder_component_never_touches_network_when_a_retriever_is_injected(monkeypatch):
    # every env var the live path would need is deliberately absent; construction must still succeed
    for var in ("ATLAS_PG_DSN", "ATLAS_TEI_EMBED_URL", "ATLAS_TEI_RERANK_URL", "ATLAS_INDEX_DIR"):
        monkeypatch.delenv(var, raising=False)
    component = build_bge_m3_embedder_component(pool_size=20, retriever=_StubRetriever([]))
    assert component.component_id == "bge-m3-local"


# ---- baseline embedder components (BM25 + exact_scan): dry assembly, no live call -----------------


def test_build_baseline_embedder_components_returns_bm25_and_exact_scan():
    retriever = _StubRetriever([Chunk(chunk_id="d1", doc_id="doc-1", text="plan a costs 10")])
    bm25, exact_scan = build_baseline_embedder_components(pool_size=50, retriever=retriever)

    assert {bm25.component_id, exact_scan.component_id} == BASELINE_COMPONENT_IDS
    assert bm25.is_baseline is True
    assert exact_scan.is_baseline is True
    assert bm25.embedding_model is None  # lexical: no real embedder at all
    assert exact_scan.embedding_model == {"id": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181"}


def test_build_baseline_embedder_components_use_lexical_only_and_exact_scan_configs_respectively():
    retriever = _StubRetriever([])
    bm25, exact_scan = build_baseline_embedder_components(pool_size=30, retriever=retriever)

    bm25.search(_CASE_A)
    exact_scan.search(_CASE_A)
    (_, _, bm25_config), (_, _, exact_config) = retriever.calls[0], retriever.calls[1]
    assert bm25_config.lexical_only is True
    assert bm25_config.rerank_enabled is False
    assert exact_config.exact_scan is True
    assert exact_config.rerank_enabled is False


def test_build_baseline_embedder_components_never_touches_network_when_a_retriever_is_injected(monkeypatch):
    for var in ("ATLAS_PG_DSN", "ATLAS_TEI_EMBED_URL"):
        monkeypatch.delenv(var, raising=False)
    bm25, exact_scan = build_baseline_embedder_components(pool_size=20, retriever=_StubRetriever([]))
    assert bm25.component_id and exact_scan.component_id


# ---- openai embedder component: dry assembly, no live call -----------------------------------------


class _StubSearcher:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.calls: list[tuple[str, int]] = []

    def search_chunks(self, query: str, k: int) -> list[Chunk]:
        self.calls.append((query, k))
        return self._chunks[:k]


def test_build_openai_embedder_component_wires_the_injected_retriever():
    chunks = [Chunk(chunk_id="o1", doc_id="doc-1", text="plan a costs 10")]
    searcher = _StubSearcher(chunks)
    component = build_openai_embedder_component(pool_size=50, retriever=searcher, index_build_id="build-test-openai")

    assert component.component_id == "openai-text-embedding-3-small"
    assert component.embedding_model == {"id": "text-embedding-3-small", "revision": "text-embedding-3-small"}
    assert component.index_build_id == "build-test-openai"

    result = component.search(_CASE_A)
    assert [c.chunk_id for c in result] == ["o1"]
    assert searcher.calls == [(_CASE_A.query, 50)]


def test_build_openai_embedder_component_raises_a_worded_failure_when_the_index_dir_is_missing(monkeypatch):
    monkeypatch.delenv("ATLAS_OPENAI_INDEX_DIR", raising=False)
    with pytest.raises(MissingEnvVarError, match="ATLAS_OPENAI_INDEX_DIR"):
        build_openai_embedder_component(pool_size=50)


def test_build_openai_embedder_component_never_touches_network_when_a_retriever_is_injected(monkeypatch):
    for var in ("ATLAS_OPENAI_INDEX_DIR", "ATLAS_PG_DSN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    component = build_openai_embedder_component(pool_size=10, retriever=_StubSearcher([]), index_build_id="b")
    assert component.component_id == "openai-text-embedding-3-small"


class _StubEmbeddingClient:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls: list[list[str]] = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [self._vector for _ in texts]


class _OpenAiEmbedderFakeCursor:
    """The same recording fake idiom `test_matrix_live_search.py`'s own `_FakeCursor` uses,
    restated here rather than imported, matching this repo's own established
    deliberate duplication over cross test import convention."""

    def __init__(self, *, vector_ids: list[str], tsv_ids: list[str], rows_by_id: dict[str, tuple]) -> None:
        self._vector_ids = vector_ids
        self._tsv_ids = tsv_ids
        self._rows_by_id = rows_by_id
        self.executed: list[tuple[str, dict]] = []
        self._last_sql = ""

    def __enter__(self) -> "_OpenAiEmbedderFakeCursor":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def execute(self, sql: str, params: dict | None = None) -> None:
        self._last_sql = sql
        self.executed.append((sql, params or {}))

    def fetchall(self):
        sql = self._last_sql
        if "ORDER BY embedding" in sql:
            return [(cid,) for cid in self._vector_ids]
        if "websearch_to_tsquery" in sql:
            return [(cid,) for cid in self._tsv_ids]
        if "chunk_id = ANY" in sql:
            _, params = self.executed[-1]
            return [self._rows_by_id[cid] for cid in params["chunk_ids"] if cid in self._rows_by_id]
        raise AssertionError(f"fetchall() called after an unexpected statement: {sql!r}")


class _OpenAiEmbedderFakeConnection:
    def __init__(self, cursor: _OpenAiEmbedderFakeCursor) -> None:
        self._cursor = cursor
        self.committed = 0
        self.closed = False

    def cursor(self) -> _OpenAiEmbedderFakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed += 1

    def close(self) -> None:
        self.closed = True


def test_build_openai_embedder_component_falls_back_to_live_construction_when_retriever_is_omitted(monkeypatch, tmp_path):
    """Closes the review's secondary coverage gap note: `retriever is None` (`live_driver.py:289-313`)
    was only ever proven to raise `MissingEnvVarError` on the missing index dir path before, never
    proven to actually assemble a working `OpenAiEmbeddedRetriever` from a real index build's own
    `build_manifest.json`/`fingerprint.json` -- unlike
    `test_build_bge_m3_embedder_component_reads_index_build_id_from_the_committed_index`, which
    proves the equivalent bge-m3 path against the real committed manifest. `index_dir`/`pg_dsn` are
    passed explicitly (bypassing the env var reads) and `embedding_client` is injected, so no live
    call is attempted anywhere in this test except `psycopg.connect`, monkeypatched here to a fake
    connection -- `OpenAiEmbeddedRetriever`'s own constructor never connects eagerly, only
    `search_chunks` does, so this proves the FULL assembly wired end to end, not just construction."""
    import psycopg

    index_dir = tmp_path / "openai-index"
    index_dir.mkdir()
    (index_dir / "build_manifest.json").write_text(json.dumps({"index_build_id": "openai-build-42"}))
    (index_dir / "fingerprint.json").write_text(json.dumps({"normalize": False, "query_prefix": "query: "}))

    rows = {"c1": ("c1", "doc-1", "doc-1", "v1", "plan_page", [], 0, 16, "plan a costs 10", [])}
    cursor = _OpenAiEmbedderFakeCursor(vector_ids=["c1"], tsv_ids=["c1"], rows_by_id=rows)
    conn = _OpenAiEmbedderFakeConnection(cursor)
    monkeypatch.setattr(psycopg, "connect", lambda dsn: conn)  # never a real connection attempt

    embedding_client = _StubEmbeddingClient([3.0, 4.0])
    component = build_openai_embedder_component(
        pool_size=5, index_dir=index_dir, pg_dsn="postgresql://fake-host/fake-db", embedding_client=embedding_client,
    )

    assert component.index_build_id == "openai-build-42"  # read from build_manifest.json, not guessed

    results = component.search(_CASE_A)

    assert [c.chunk_id for c in results] == ["c1"]
    build_ids = {params["build_id"] for _, params in cursor.executed if "build_id" in params}
    assert build_ids == {"openai-build-42"}  # scoped to the manifest's own index_build_id
    assert embedding_client.calls == [["query: how much is plan a"]]  # fingerprint's query_prefix applied
    vector_call = next(params for _, params in cursor.executed if "vector" in params)
    assert vector_call["vector"] == "[3.0,4.0]"  # fingerprint's normalize=False left the vector as is
    assert conn.closed is True


# ---- reranker components: dry assembly, no live call -----------------------------------------------


class _StubReranker:
    def rerank(self, query, chunks):
        return list(reversed(chunks))


def test_build_reranker_components_returns_bge_and_none_axes():
    bge, none = build_reranker_components(reranker=_StubReranker())
    assert isinstance(bge, RerankerComponent)
    assert isinstance(none, RerankerComponent)
    assert bge.component_id == "bge-reranker-v2-m3"
    assert bge.reranker is not None
    assert none.component_id == "none"
    assert none.reranker is None


def test_build_reranker_components_never_touches_network_when_a_reranker_is_injected(monkeypatch):
    monkeypatch.delenv("ATLAS_TEI_RERANK_URL", raising=False)
    bge, _none = build_reranker_components(reranker=_StubReranker())
    assert bge.component_id == "bge-reranker-v2-m3"


def test_build_reranker_components_raises_a_worded_failure_when_no_reranker_injected_and_url_missing(monkeypatch):
    monkeypatch.delenv("ATLAS_TEI_RERANK_URL", raising=False)
    with pytest.raises(MissingEnvVarError, match="ATLAS_TEI_RERANK_URL"):
        build_reranker_components()


# ---- generator components: dry assembly, no live call -----------------------------------------------


def test_build_claude_generator_component_reads_the_real_models_lock_entry(tmp_path):
    component = build_claude_generator_component(cases=_CASES, cassette_dir=tmp_path, inner=_StubProvider())
    assert isinstance(component, GeneratorComponent)
    assert component.component_id == "anthropic-claude-sonnet-5"
    assert component.model_snapshot == {
        "provider": "anthropic", "model_id": "claude-sonnet-5", "revision": "claude-sonnet-5",
    }
    assert component.gateway.mode is GatewayMode.RECORD
    assert component.gateway.model_id == "anthropic:claude-sonnet-5"


def test_build_claude_generator_component_computes_a_real_positive_estimate(tmp_path):
    component = build_claude_generator_component(cases=_CASES, cassette_dir=tmp_path, inner=_StubProvider())
    assert component.estimated_usd > 0.0


def test_build_gpt_generator_component_reads_the_real_models_lock_entry(tmp_path):
    component = build_gpt_generator_component(cases=_CASES, cassette_dir=tmp_path, inner=_StubProvider())
    assert component.component_id == "openai-gpt-5.6-sol"
    assert component.model_snapshot == {
        "provider": "openai", "model_id": "gpt-5.6-sol", "revision": "gpt-5.6-sol",
    }
    assert component.estimated_usd > 0.0


def test_generator_components_never_touch_network_when_inner_is_injected(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    claude = build_claude_generator_component(cases=_CASES, cassette_dir=tmp_path, inner=_StubProvider())
    gpt = build_gpt_generator_component(cases=_CASES, cassette_dir=tmp_path, inner=_StubProvider())
    assert claude.component_id and gpt.component_id


# ---- estimate_generation_cost_usd: pure math over real prompt shape -------------------------------


def test_estimate_generation_cost_usd_is_zero_for_no_cases():
    assert estimate_generation_cost_usd((), "anthropic") == 0.0


def test_estimate_generation_cost_usd_is_positive_for_a_paid_provider():
    assert estimate_generation_cost_usd(_CASES, "anthropic") > 0.0
    assert estimate_generation_cost_usd(_CASES, "openai") > 0.0


def test_estimate_generation_cost_usd_is_zero_for_ollama():
    assert estimate_generation_cost_usd(_CASES, "ollama") == 0.0


def test_estimate_generation_cost_usd_scales_with_case_count():
    # `rel=0.01`, not an exact doubling: `int(total_chars / chars_per_token)` floors per call, so a
    # single extra identical case can shift the floored token count by one token's worth of cost --
    # negligible at this scale, but real, not a bug this test should chase to exact equality.
    single = estimate_generation_cost_usd((_CASE_A,), "anthropic")
    double = estimate_generation_cost_usd((_CASE_A, _CASE_A), "anthropic")
    assert double == pytest.approx(single * 2, rel=0.01)


# ---- judge gateway: dry assembly, no live call -----------------------------------------------------


def test_build_judge_gateway_reads_the_real_generator_lock_entry_and_tags_it_distinctly(tmp_path):
    gateway = build_judge_gateway("anthropic", cassette_dir=tmp_path, inner=_StubProvider())
    assert gateway.mode is GatewayMode.RECORD
    assert gateway.model_id == "anthropic:judge-claude-sonnet-5"


def test_build_judge_gateway_never_touches_network_when_inner_is_injected(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for provider in ("anthropic", "openai", "ollama"):
        gateway = build_judge_gateway(provider, cassette_dir=tmp_path, inner=_StubProvider())
        assert gateway.mode is GatewayMode.RECORD


def test_build_judge_gateway_is_distinct_from_that_providers_own_generator_cassette_key(tmp_path):
    claude_generator = build_claude_generator_component(cases=_CASES, cassette_dir=tmp_path, inner=_StubProvider())
    judge_gateway = build_judge_gateway("anthropic", cassette_dir=tmp_path, inner=_StubProvider())
    assert claude_generator.gateway.model_id != judge_gateway.model_id


# ---- variants config: pre spend-checked, dry assembly, no live call --------------------------------


def test_build_variants_config_ollama_always_admits_with_no_spend_check_needed():
    gate = SpendGate()
    config, gate_out, reason = build_variants_config(
        cases=_CASES, gate=gate, provider="ollama", cassette_dir="unused",
        retriever=_StubRetriever([]), reranker=_StubReranker(), graph=object(), inner=_StubProvider(),
    )
    assert reason is None
    assert config is not None
    assert config.gateway.mode is GatewayMode.RECORD
    assert gate_out == gate  # ollama never spends against any provider's ceiling


def test_build_variants_config_refuses_a_paid_provider_that_would_exceed_the_ceiling(tmp_path):
    tiny_gate = SpendGate(ceilings={"anthropic": 0.0001, "openai": 20.0, "ollama": 0.0})
    config, gate_out, reason = build_variants_config(
        cases=_CASES, gate=tiny_gate, provider="anthropic", cassette_dir=tmp_path,
        retriever=_StubRetriever([]), reranker=_StubReranker(), graph=object(), inner=_StubProvider(),
    )
    assert config is None
    assert reason is not None and "anthropic" in reason
    assert gate_out == tiny_gate  # a refusal spends nothing, the gate comes back unchanged


def test_build_variants_config_admits_a_paid_provider_within_budget(tmp_path):
    gate = SpendGate()
    config, gate_out, reason = build_variants_config(
        cases=_CASES, gate=gate, provider="anthropic", cassette_dir=tmp_path,
        retriever=_StubRetriever([]), reranker=_StubReranker(), graph=object(), inner=_StubProvider(),
    )
    assert reason is None
    assert config is not None
    assert config.gateway.model_id == "anthropic:claude-sonnet-5"
    assert gate_out is not gate  # a NEW gate, never the same instance mutated in place


def test_build_variants_config_records_the_admitted_estimate_into_the_returned_gate(tmp_path):
    """The Important review finding: `build_variants_config` only ever READ the gate via
    `check_spend`, never recorded the admitted estimate back into it, so the SAME pristine gate
    handed to `run_matrix` for stage 3 was unaware the variant stage had already committed real
    spend against the identical starting budget. The returned gate must carry that spend."""
    gate = SpendGate()
    estimate = 3 * estimate_generation_cost_usd(_CASES, "anthropic")
    _config, gate_out, reason = build_variants_config(
        cases=_CASES, gate=gate, provider="anthropic", cassette_dir=tmp_path,
        retriever=_StubRetriever([]), reranker=_StubReranker(), graph=object(), inner=_StubProvider(),
    )
    assert reason is None
    assert gate_out.spent_usd("anthropic") == pytest.approx(estimate)
    assert gate.spent_usd("anthropic") == 0.0  # the ORIGINAL gate stays untouched (immutable)


def test_build_variants_config_reconciles_against_stage_3s_own_admission_closing_the_cross_stage_gap(tmp_path):
    """Reproduces the reviewer's exact scenario at unit scale: a ceiling that fits the variant
    stage's own 3x estimate AND stage 3's own 1x estimate individually against the PRISTINE gate,
    but not their sum. Before the fix, `main()` fed stage 3 the same pristine gate the variant stage
    was checked against, so both checks independently reported "fits" and the combined real spend
    silently exceeded the ceiling. After the fix, the caller threads the RETURNED gate (with the
    variant stage's admitted spend already recorded) into stage 3's own `check_spend` call, so the
    combined spend is correctly bounded."""
    single = estimate_generation_cost_usd(_CASES, "anthropic")
    variant_estimate = 3 * single
    stage3_estimate = single
    ceiling = variant_estimate + (0.5 * stage3_estimate)  # room for the variant stage, not for both
    gate = SpendGate(ceilings={"anthropic": ceiling, "openai": 20.0, "ollama": 0.0})

    # Sanity: both checks independently "fit" against the PRISTINE gate -- this is exactly the
    # illusion the Important finding names; the fix is in what gate stage 3 is checked against next.
    assert check_spend(gate, "anthropic", variant_estimate).allowed is True
    assert check_spend(gate, "anthropic", stage3_estimate).allowed is True

    _config, gate_out, reason = build_variants_config(
        cases=_CASES, gate=gate, provider="anthropic", cassette_dir=tmp_path,
        retriever=_StubRetriever([]), reranker=_StubReranker(), graph=object(), inner=_StubProvider(),
    )
    assert reason is None  # the variant stage itself is admitted, same as before the fix

    # The fix: checked against the RETURNED (reconciled) gate, stage 3's own admission now correctly
    # refuses -- the combined spend can never silently exceed the ceiling.
    stage3_decision = check_spend(gate_out, "anthropic", stage3_estimate)
    assert stage3_decision.allowed is False


def test_build_variants_config_never_touches_network_when_every_dependency_is_injected(monkeypatch, tmp_path):
    for var in ("ATLAS_PG_DSN", "ATLAS_TEI_EMBED_URL", "ATLAS_TEI_RERANK_URL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    gate = SpendGate()
    config, _gate_out, reason = build_variants_config(
        cases=_CASES, gate=gate, provider="anthropic", cassette_dir=tmp_path,
        retriever=_StubRetriever([]), reranker=_StubReranker(), graph=object(), inner=_StubProvider(),
    )
    assert reason is None and config is not None


def test_build_variants_config_falls_back_to_live_construction_when_nothing_is_injected(monkeypatch, tmp_path):
    """Never actually reaches the network: with `retriever`/`reranker`/`graph` all omitted AND the
    env vars they would need all deleted, `construct_live_pgvector_retriever`'s own fail-closed
    check fires first -- proving `build_variants_config` really does delegate to the live
    construction helpers on the omitted-dependency path, rather than that branch being declared but
    never actually exercised."""
    for var in ("ATLAS_PG_DSN", "ATLAS_TEI_EMBED_URL"):
        monkeypatch.delenv(var, raising=False)
    gate = SpendGate()
    with pytest.raises(MissingEnvVarError, match="ATLAS_PG_DSN"):
        build_variants_config(
            cases=_CASES, gate=gate, provider="ollama", cassette_dir=tmp_path, inner=_StubProvider(),
        )


def test_build_variants_config_falls_back_to_live_reranker_construction_when_reranker_is_omitted(monkeypatch, tmp_path):
    """Closes the review's secondary coverage gap note: only the `retriever is None` delegation line
    was ever independently proven to fall through to live construction; `reranker is None`
    (`live_driver.py:525`) was declared but never exercised on its own. `retriever` is injected here
    so the retriever branch never fires first and masks this one."""
    monkeypatch.delenv("ATLAS_TEI_RERANK_URL", raising=False)
    gate = SpendGate()
    with pytest.raises(MissingEnvVarError, match="ATLAS_TEI_RERANK_URL"):
        build_variants_config(
            cases=_CASES, gate=gate, provider="ollama", cassette_dir=tmp_path,
            retriever=_StubRetriever([]), inner=_StubProvider(),
        )


def test_build_variants_config_falls_back_to_live_graph_construction_when_graph_is_omitted(monkeypatch, tmp_path):
    """Closes the review's secondary coverage gap note: `graph is None` (`live_driver.py:528`) was
    declared but never independently exercised either. `retriever`/`reranker` are both injected here
    so neither of their own branches fires first and masks this one."""
    monkeypatch.delenv("ATLAS_PG_DSN", raising=False)
    gate = SpendGate()
    with pytest.raises(MissingEnvVarError, match="ATLAS_PG_DSN"):
        build_variants_config(
            cases=_CASES, gate=gate, provider="ollama", cassette_dir=tmp_path,
            retriever=_StubRetriever([]), reranker=_StubReranker(), inner=_StubProvider(),
        )


def test_ollama_is_still_the_one_always_runs_provider_this_module_relies_on():
    # a cheap tripwire: if matrix.spend_gate.ALWAYS_RUNS ever changes, build_variants_config's own
    # "ollama never needs a spend pre-check" branch needs to be revisited right alongside it.
    assert ALWAYS_RUNS == frozenset({"ollama"})


# ---- a full, real models.lock sanity check (guards every hardcoded id/revision string above) -----


def test_the_real_models_lock_has_every_entry_this_module_reads_by_provider():
    data = json.loads((Path(__file__).resolve().parents[2] / "models.lock").read_text())
    embedding_providers = {e["provider"] for e in data["embedding"]}
    generator_providers = {e["provider"] for e in data["generator"]}
    assert {"local-tei", "openai"} <= embedding_providers
    assert {"anthropic", "openai", "ollama"} <= generator_providers
    assert len(data["reranker"]) == 1
