"""The matrix LIVE DRIVER (SP9 task 4's own deferred "live caller"): constructs the REAL component
sets `matrix.runner.run_matrix` expects -- the real `EmbedderComponent` axis (`bge-m3` via a live
TEI call through `PgvectorRetriever`, `text-embedding-3-small` via a real `OpenAiEmbeddingClient`
bound to `matrix.live_search.OpenAiEmbeddedRetriever`), the real `RerankerComponent` axis (BGE
reranker v2 m3 via a live TEI `/rerank` call, `matrix.live_search.TeiReranker`, plus `none`), the
real `GeneratorComponent` axis (Claude and GPT via `replay.gateway.GatewayChatModel` RECORD mode;
`qwen2.5:7b` on Ollama is already real, `matrix.ollama_generator.build_ollama_generator_component`,
reused here rather than re-built), and the naive/agentic/graph `VariantsConfig` `matrix.variants.
run_variant_comparison` needs.

DEPENDENCY INJECTION, not a live/hermetic branch inside each function: every `build_*` function
below accepts its own real collaborator as an optional keyword (`retriever`, `reranker`,
`embedding_client`, `inner`, ...). Given one, construction is 100% hermetic -- no import of
`httpx`/`psycopg`/a provider SDK ever executes, no env var is even read for that collaborator.
Omitted, the SAME function builds the real thing from the process environment, WORDED-FAILING
(`MissingEnvVarError`) on a missing required var before any live call is attempted, mirroring
`atlas.adapters.pgvector_retriever.PgvectorRetriever`'s own fail-closed construction discipline and
`matrix.ollama_generator.build_ollama_generator_component`'s own `inner=None` seam. This is the SAME
"seeded fixture now, live swap deferred, no change to the contract" shape `matrix.embedders`/
`matrix.rerankers`/`matrix.variants` already establish; this module is simply the caller that
finally exercises the live swap those modules' own docstrings named as deferred.

ALL configuration is read from the environment, never hardcoded (no port number, no localhost URL
baked into this file): `ATLAS_PG_DSN`, `ATLAS_TEI_EMBED_URL`, `ATLAS_TEI_RERANK_URL`,
`ATLAS_OPENAI_INDEX_DIR`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (Ollama's own base URL is read by
`replay.providers.build_chat_model` itself, `OLLAMA_BASE_URL`, already env driven and already
defaulted -- Ollama is the one `matrix.spend_gate.ALWAYS_RUNS` axis, local and free, so it alone
never needs a fail-closed check here). `ATLAS_INDEX_DIR` is the one deliberate exception: the
bge-m3 index has a real, committed default (`indexes/corpus-0.1.1-bge-m3-03f983e0`), the SAME
default `atlas.config.AtlasSettings`/`PgvectorRetriever` already fall back to, so requiring it here
too would only force an operator to redundantly restate a path this repo already ships.

`text-embedding-3-small`'s own index directory (`ATLAS_OPENAI_INDEX_DIR`) has NO such default: no
OpenAI-embedded index build is committed to this repo (only the bge-m3 one is), and building one is
out of this task's scope (`rag_tools.ingest.build_index` is itself hardcoded to a live TEI embed
call, not yet pluggable to a second embedding client) -- a real, disclosed narrowness, not a bug:
an operator must build and load that index first (mirroring `task rag:ingest`'s own bge-m3 recipe,
adapted to call `OpenAiEmbeddingClient` instead of TEI) before this axis can run for real. Until
then, `build_openai_embedder_component` fails closed with a worded message naming the missing var,
never a silent skip.

Real per-cell cost ESTIMATES (`estimate_generation_cost_usd`) are computed from the ACTUAL
`matrix.generators.build_generate_prompt` shape over the real `cases` a caller passes in, fed to
every paid `GeneratorComponent.estimated_usd` -- so `matrix.spend_gate.check_spend`'s own
LIVE-MONEY SAFETY refusal (a zero-or-unknown estimate can never be admitted against a paid
provider) never silently defeats itself here. `matrix.runner.run_matrix`'s own variant-comparison
stage has NO equivalent spend gate at all (it runs unconditionally whenever a caller supplies a
`VariantsConfig`); `build_variants_config` is this driver's own pre-check, refusing to build a
paid variant gateway that would exceed its provider's remaining budget, gated the SAME way stage 3's
generator cells already are. Because that pre-check and stage 3's own admission both check the SAME
provider's remaining budget, `build_variants_config` returns an UPDATED gate (the admitted estimate
recorded via `matrix.spend_gate.record_spend`), never just a decision -- the caller (`matrix.
__main__.main`) threads that returned gate, not the one it passed in, into `run_matrix`'s own
`spend_gate` argument, so stage 3's admission sees the variant stage's real spend already reflected
in its remaining budget. Skipping this reconciliation (checking stage 3 against the SAME pristine
gate the variant stage was checked against) would let both checks independently report "fits" while
their combined real spend silently exceeds the hard ceiling -- exactly the failure mode `matrix.
spend_gate.SpendGate`'s own "never silently overspend" contract exists to prevent.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from atlas.domain.retrieval import RetrievalConfig
from atlas.ports.embedding import EmbeddingClient
from atlas.ports.knowledge import Chunk, Retriever
from atlas.ports.knowledge_graph import KnowledgeGraph
from atlas.ports.reranker import Reranker

from matrix.cases import MatrixCase
from matrix.embedders import BM25_COMPONENT_ID, EXACT_SCAN_COMPONENT_ID, EmbedderComponent
from matrix.generators import GeneratorComponent, build_generate_prompt
from matrix.live_search import OpenAiEmbeddedRetriever, TeiReranker
from matrix.ollama_generator import MODELS_LOCK_PATH
from matrix.rerankers import NONE_RERANKER_ID, RerankerComponent
from matrix.spend_gate import (
    ALWAYS_RUNS,
    SpendGate,
    build_generator_gateway,
    check_spend,
    generation_cost_usd,
    record_spend,
)
from matrix.variants import VariantsConfig

_BGE_M3_COMPONENT_ID = "bge-m3-local"
_OPENAI_EMBEDDER_COMPONENT_ID = "openai-text-embedding-3-small"
_BGE_RERANKER_COMPONENT_ID = "bge-reranker-v2-m3"

_BGE_M3_EMBEDDING_PROVIDER = "local-tei"
_OPENAI_EMBEDDING_PROVIDER = "openai"

# The one committed real index this repo ships (SP3 task 5), the SAME default `atlas.config.
# AtlasSettings`/`PgvectorRetriever` themselves fall back to. See the module docstring on why this
# is the one deliberate env-optional exception.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_BGE_M3_INDEX_DIR = _REPO_ROOT / "indexes" / "corpus-0.1.1-bge-m3-03f983e0"

# A conservative, tunable PRE-CALL cost estimate (see `estimate_generation_cost_usd`'s own
# docstring): recheck these against the real corpus/model before trusting a large sweep's budget,
# the same "recheck before trusting" note `matrix.spend_gate.GENERATION_PRICE_PER_1M` already carries.
_DEFAULT_AVG_OUTPUT_TOKENS = 300
_DEFAULT_K_CHUNKS_ESTIMATE = 5
_DEFAULT_AVG_CHUNK_CHARS_ESTIMATE = 600
_CHARS_PER_TOKEN_ESTIMATE = 4.0


class MissingEnvVarError(RuntimeError):
    """Raised BEFORE any live call when a required environment variable is unset or empty."""


def require_env(name: str, *, purpose: str) -> str:
    """Read `name` from the process environment. Worded-fails immediately (never a bare `KeyError`
    or a silently-`None` value surfacing deep inside a provider SDK far from the real cause) when
    `name` is unset or set to an empty string."""
    value = os.environ.get(name)
    if not value:
        raise MissingEnvVarError(
            f"{name} is not set, needed for {purpose}. Set it (e.g. in .env / .env.fastlane, or "
            "export it directly) before running the matrix live driver -- refusing to attempt any "
            "live call with it missing."
        )
    return value


def _embedding_lock_entry(models_lock_path: Path, *, provider: str) -> dict:
    """`models.lock`'s own `embedding` entry for `provider`, in `EmbedderComponent.embedding_model`'s
    own `{"id", "revision"}` shape (see `matrix.embedders.EmbedderComponent`'s own docstring)."""
    data = json.loads(Path(models_lock_path).read_text())
    entry = next((e for e in data.get("embedding", []) if e["provider"] == provider), None)
    if entry is None:
        raise ValueError(f"models.lock ({models_lock_path}) has no embedding entry for provider={provider!r}")
    return {"id": entry["model_id"], "revision": entry["revision"]}


def _reranker_lock_entry(models_lock_path: Path) -> dict:
    data = json.loads(Path(models_lock_path).read_text())
    rerankers = data.get("reranker", [])
    if not rerankers:
        raise ValueError(f"models.lock ({models_lock_path}) has no reranker entry")
    return rerankers[0]


def _generator_lock_entry(models_lock_path: Path, *, provider: str) -> dict:
    data = json.loads(Path(models_lock_path).read_text())
    entry = next((e for e in data.get("generator", []) if e["provider"] == provider), None)
    if entry is None:
        raise ValueError(f"models.lock ({models_lock_path}) has no generator entry for provider={provider!r}")
    return entry


# ---- stage 1: embedders --------------------------------------------------------------------------


def construct_live_pgvector_retriever(
    *,
    pg_dsn: Optional[str],
    tei_embed_url: Optional[str],
    tei_rerank_url: Optional[str],
    index_dir: Optional[str | Path],
) -> Retriever:  # pragma: no cover - live only, needs Postgres + a reachable TEI server
    resolved_pg_dsn = pg_dsn or require_env(
        "ATLAS_PG_DSN", purpose="constructing the live PgvectorRetriever (bge-m3 embedder axis)"
    )
    resolved_embed_url = tei_embed_url or require_env(
        "ATLAS_TEI_EMBED_URL", purpose="embedding the live bge-m3 query vector"
    )
    resolved_rerank_url = tei_rerank_url or os.environ.get("ATLAS_TEI_RERANK_URL")
    resolved_index_dir = Path(index_dir) if index_dir else Path(os.environ.get("ATLAS_INDEX_DIR", _DEFAULT_BGE_M3_INDEX_DIR))

    from atlas.adapters.pgvector_retriever import PgvectorRetriever

    kwargs: dict[str, Any] = {"pg_dsn": resolved_pg_dsn, "tei_embed_url": resolved_embed_url, "index_dir": resolved_index_dir}
    if resolved_rerank_url:
        kwargs["tei_rerank_url"] = resolved_rerank_url
    return PgvectorRetriever(**kwargs)


def build_bge_m3_embedder_component(
    *,
    pool_size: int,
    retriever: Optional[Retriever] = None,
    embedding_model: Optional[dict] = None,
    index_build_id: Optional[str] = None,
    pg_dsn: Optional[str] = None,
    tei_embed_url: Optional[str] = None,
    tei_rerank_url: Optional[str] = None,
    index_dir: Optional[str | Path] = None,
    models_lock_path: Path = MODELS_LOCK_PATH,
) -> EmbedderComponent:
    """The real `bge-m3` embedder axis (SP9 task 3's local axis). `retriever` (any `Retriever`,
    e.g. a hermetic stub) skips ALL live construction; omitted, builds a real `PgvectorRetriever`
    from env (worded failures before any live call). `embedding_model`/`index_build_id` default to
    `models.lock`'s own `local-tei` entry and the active index's `build_manifest.json` -- both plain
    committed-file reads, safe and deterministic in every context, hermetic or live alike."""
    resolved_index_dir = Path(index_dir) if index_dir else _DEFAULT_BGE_M3_INDEX_DIR
    if embedding_model is None:
        embedding_model = _embedding_lock_entry(models_lock_path, provider=_BGE_M3_EMBEDDING_PROVIDER)
    if index_build_id is None:
        manifest_path = resolved_index_dir / "build_manifest.json"
        if manifest_path.is_file():
            index_build_id = json.loads(manifest_path.read_text())["index_build_id"]
    if retriever is None:
        retriever = construct_live_pgvector_retriever(
            pg_dsn=pg_dsn, tei_embed_url=tei_embed_url, tei_rerank_url=tei_rerank_url, index_dir=resolved_index_dir,
        )

    def _search(case: MatrixCase) -> Sequence[Chunk]:
        config = RetrievalConfig(rerank_enabled=False, k_fused=pool_size, k_final=pool_size)
        return retriever.search_chunks(case.query, k=pool_size, config=config)

    return EmbedderComponent(
        component_id=_BGE_M3_COMPONENT_ID, search=_search, embedding_model=embedding_model, index_build_id=index_build_id,
    )


def build_baseline_embedder_components(
    *,
    pool_size: int,
    retriever: Optional[Retriever] = None,
    embedding_model: Optional[dict] = None,
    pg_dsn: Optional[str] = None,
    tei_embed_url: Optional[str] = None,
    tei_rerank_url: Optional[str] = None,
    index_dir: Optional[str | Path] = None,
    models_lock_path: Path = MODELS_LOCK_PATH,
) -> tuple[EmbedderComponent, EmbedderComponent]:
    """The two named baseline rows every stage 1 run carries (D8/research 14, `matrix.embedders`'s
    own module docstring: "never omitted"): BM25 (lexical, no reranker,
    `RetrievalConfig(lexical_only=True)` -- no embed call at all, no `embedding_model`) and
    `exact_scan` (the recall ground truth row, HNSW bypassed, `RetrievalConfig(exact_scan=True)`,
    still bge-m3 embedded). Neither `matrix.embedders` nor any other module in this package
    constructs these two components itself (see `EmbedderComponent`'s own docstring): every caller,
    hermetic test and live driver alike, builds them. `retriever` is injected the SAME way
    `build_bge_m3_embedder_component`'s own seam works, so a live caller can share ONE real
    `PgvectorRetriever` (one live fingerprint check, not three) across bge-m3 AND both baselines --
    pass the SAME already-constructed retriever to all three calls."""
    resolved_index_dir = Path(index_dir) if index_dir else _DEFAULT_BGE_M3_INDEX_DIR
    if embedding_model is None:
        embedding_model = _embedding_lock_entry(models_lock_path, provider=_BGE_M3_EMBEDDING_PROVIDER)
    if retriever is None:
        retriever = construct_live_pgvector_retriever(
            pg_dsn=pg_dsn, tei_embed_url=tei_embed_url, tei_rerank_url=tei_rerank_url, index_dir=resolved_index_dir,
        )

    def _bm25_search(case: MatrixCase) -> Sequence[Chunk]:
        config = RetrievalConfig(lexical_only=True, rerank_enabled=False, k_fused=pool_size, k_final=pool_size)
        return retriever.search_chunks(case.query, k=pool_size, config=config)

    def _exact_scan_search(case: MatrixCase) -> Sequence[Chunk]:
        config = RetrievalConfig(exact_scan=True, rerank_enabled=False, k_fused=pool_size, k_final=pool_size)
        return retriever.search_chunks(case.query, k=pool_size, config=config)

    bm25 = EmbedderComponent(
        component_id=BM25_COMPONENT_ID, search=_bm25_search, embedding_model=None, is_baseline=True,
    )
    exact_scan = EmbedderComponent(
        component_id=EXACT_SCAN_COMPONENT_ID, search=_exact_scan_search, embedding_model=embedding_model, is_baseline=True,
    )
    return bm25, exact_scan


def construct_live_openai_embedding_client(model_id: str) -> EmbeddingClient:  # pragma: no cover - live only, needs OPENAI_API_KEY
    require_env("OPENAI_API_KEY", purpose="embedding live queries for the text-embedding-3-small axis")
    from atlas.adapters.openai_embedding import OpenAiEmbeddingClient

    return OpenAiEmbeddingClient(model_id)


def build_openai_embedder_component(
    *,
    pool_size: int,
    retriever: Optional[Any] = None,
    embedding_client: Optional[EmbeddingClient] = None,
    embedding_model: Optional[dict] = None,
    index_build_id: Optional[str] = None,
    pg_dsn: Optional[str] = None,
    index_dir: Optional[str | Path] = None,
    models_lock_path: Path = MODELS_LOCK_PATH,
) -> EmbedderComponent:
    """The real `text-embedding-3-small` embedder axis (SP9 task 3's OpenAI axis). `retriever` (any
    object exposing `search_chunks(query, k) -> list[Chunk]`, e.g. `matrix.live_search.
    OpenAiEmbeddedRetriever` or a hermetic stub) skips ALL live construction -- pass `index_build_id`
    explicitly alongside it (there is no live index directory to read it from in that case).
    Omitted, requires `ATLAS_OPENAI_INDEX_DIR` (no default; see the module docstring on why this
    axis, unlike bge-m3, has none) and `ATLAS_PG_DSN`, worded-failing before any live call if either
    is missing, then builds the real `OpenAiEmbeddingClient` + `OpenAiEmbeddedRetriever` from that
    index's own `fingerprint.json`/`build_manifest.json`."""
    if embedding_model is None:
        embedding_model = _embedding_lock_entry(models_lock_path, provider=_OPENAI_EMBEDDING_PROVIDER)

    if retriever is None:
        resolved_index_dir = Path(index_dir) if index_dir else Path(require_env(
            "ATLAS_OPENAI_INDEX_DIR",
            purpose="locating the OpenAI-embedded index build (fingerprint.json/build_manifest.json) "
            "for the text-embedding-3-small embedder axis",
        ))
        resolved_pg_dsn = pg_dsn or require_env(
            "ATLAS_PG_DSN", purpose="querying the OpenAI-embedded index's rows in Postgres"
        )
        manifest = json.loads((resolved_index_dir / "build_manifest.json").read_text())
        fingerprint = json.loads((resolved_index_dir / "fingerprint.json").read_text())
        if embedding_client is None:
            embedding_client = construct_live_openai_embedding_client(embedding_model["id"])

        import psycopg

        retriever = OpenAiEmbeddedRetriever(
            embedding_client=embedding_client,
            index_build_id=manifest["index_build_id"],
            connect=lambda dsn=resolved_pg_dsn: psycopg.connect(dsn),
            normalize=fingerprint.get("normalize", True),
            query_prefix=fingerprint.get("query_prefix", ""),
        )
        if index_build_id is None:
            index_build_id = manifest["index_build_id"]

    def _search(case: MatrixCase) -> Sequence[Chunk]:
        return retriever.search_chunks(case.query, k=pool_size)

    return EmbedderComponent(
        component_id=_OPENAI_EMBEDDER_COMPONENT_ID, search=_search,
        embedding_model=embedding_model, index_build_id=index_build_id,
    )


# ---- stage 2: rerankers --------------------------------------------------------------------------


def construct_live_tei_reranker(*, tei_rerank_url: Optional[str]) -> Reranker:  # pragma: no cover - live only, needs a reachable TEI server
    resolved = tei_rerank_url or require_env(
        "ATLAS_TEI_RERANK_URL", purpose="the live BGE reranker component / variant comparison reranker"
    )
    return TeiReranker(base_url=resolved)


def build_reranker_components(
    *,
    reranker: Optional[Reranker] = None,
    tei_rerank_url: Optional[str] = None,
    models_lock_path: Path = MODELS_LOCK_PATH,
) -> tuple[RerankerComponent, RerankerComponent]:
    """`{BGE reranker v2 m3, none}` (SP9's own thin, documented reranker axis). `reranker` injects a
    stub for hermetic tests, skipping all live construction; omitted, requires `ATLAS_TEI_RERANK_URL`
    (worded failure before any live call) and builds a real `matrix.live_search.TeiReranker`."""
    _reranker_lock_entry(models_lock_path)  # fail closed on a models.lock missing the reranker axis
    if reranker is None:
        reranker = construct_live_tei_reranker(tei_rerank_url=tei_rerank_url)
    return (
        RerankerComponent(component_id=_BGE_RERANKER_COMPONENT_ID, reranker=reranker),
        RerankerComponent(component_id=NONE_RERANKER_ID, reranker=None),
    )


# ---- stage 3: generators -------------------------------------------------------------------------


def estimate_generation_cost_usd(
    cases: Sequence[MatrixCase],
    provider: str,
    *,
    avg_output_tokens: int = _DEFAULT_AVG_OUTPUT_TOKENS,
    k_chunks: int = _DEFAULT_K_CHUNKS_ESTIMATE,
    avg_chunk_chars: int = _DEFAULT_AVG_CHUNK_CHARS_ESTIMATE,
    chars_per_token: float = _CHARS_PER_TOKEN_ESTIMATE,
) -> float:
    """A conservative PRE-CALL dollar estimate for one (retrieval config, generator) cell's worth of
    real generate calls over `cases` -- the honest, positive upfront number `matrix.spend_gate.
    check_spend`'s own LIVE-MONEY SAFETY contract requires before a paid cell may be admitted at all
    (a real per-token cost is only known AFTER the call returns `usage_metadata`). Reuses the REAL
    `matrix.generators.build_generate_prompt` shape (never re-derives a second prompt-size guess)
    over `k_chunks` filler passages of `avg_chunk_chars` each -- `avg_chunk_chars`/`k_chunks`/
    `avg_output_tokens`/`chars_per_token` are named, tunable assumptions, not measured actuals:
    recheck them against the real corpus/model before trusting this number for a large sweep, the
    same "recheck before trusting" discipline `matrix.spend_gate.GENERATION_PRICE_PER_1M`'s own
    module docstring already asks of its pricing table. Zero for an empty `cases` (nothing to
    estimate) and for `ollama` (always free, `matrix.spend_gate.ALWAYS_RUNS`, priced at zero by
    `matrix.spend_gate.GENERATION_PRICE_PER_1M` regardless)."""
    if not cases:
        return 0.0

    class _FillerChunk:
        def __init__(self, text: str) -> None:
            self.text = text

    filler_chunks = [_FillerChunk("x" * avg_chunk_chars) for _ in range(k_chunks)]
    total_chars = sum(len(build_generate_prompt(case.query, filler_chunks)) for case in cases)
    input_tokens = int(total_chars / chars_per_token)
    output_tokens = len(cases) * avg_output_tokens
    return generation_cost_usd(provider, input_tokens, output_tokens)


def _live_anthropic_model(model_id: str) -> BaseChatModel:  # pragma: no cover - live only, needs ANTHROPIC_API_KEY
    require_env("ANTHROPIC_API_KEY", purpose="constructing the live Claude generator (ChatAnthropic)")
    from replay.providers import build_chat_model

    return build_chat_model("anthropic", model_id)


def build_claude_generator_component(
    *,
    cases: Sequence[MatrixCase],
    cassette_dir: Path,
    inner: Optional[BaseChatModel] = None,
    models_lock_path: Path = MODELS_LOCK_PATH,
    avg_output_tokens: int = _DEFAULT_AVG_OUTPUT_TOKENS,
) -> GeneratorComponent:
    """The real Claude generator cell. `inner` injects a stub `BaseChatModel` for hermetic tests
    (skips the live Anthropic SDK entirely); omitted, requires `ANTHROPIC_API_KEY` (worded failure
    before any live call) and builds the real `ChatAnthropic`. Always RECORD mode (`matrix.
    spend_gate.build_generator_gateway`), matching every other generator cell in this package."""
    entry = _generator_lock_entry(models_lock_path, provider="anthropic")
    model_id = entry["model_id"]
    live_inner = inner if inner is not None else _live_anthropic_model(model_id)
    gateway = build_generator_gateway(provider="anthropic", model_id=model_id, inner=live_inner, cassette_dir=Path(cassette_dir))
    estimated_usd = estimate_generation_cost_usd(cases, "anthropic", avg_output_tokens=avg_output_tokens)
    return GeneratorComponent(
        component_id=f"anthropic-{model_id}",
        model_snapshot={"provider": entry["provider"], "model_id": entry["model_id"], "revision": entry["revision"]},
        gateway=gateway,
        estimated_usd=estimated_usd,
    )


def _live_gpt_model(model_id: str) -> BaseChatModel:  # pragma: no cover - live only, needs OPENAI_API_KEY
    require_env("OPENAI_API_KEY", purpose="constructing the live GPT generator (ChatOpenAI)")
    from replay.providers import build_chat_model

    return build_chat_model("openai", model_id)


def build_gpt_generator_component(
    *,
    cases: Sequence[MatrixCase],
    cassette_dir: Path,
    inner: Optional[BaseChatModel] = None,
    models_lock_path: Path = MODELS_LOCK_PATH,
    avg_output_tokens: int = _DEFAULT_AVG_OUTPUT_TOKENS,
) -> GeneratorComponent:
    """The real GPT generator cell, the OpenAI mirror of `build_claude_generator_component` (see its
    own docstring for the shared discipline: `inner` injection, RECORD mode, a real upfront
    estimate)."""
    entry = _generator_lock_entry(models_lock_path, provider="openai")
    model_id = entry["model_id"]
    live_inner = inner if inner is not None else _live_gpt_model(model_id)
    gateway = build_generator_gateway(provider="openai", model_id=model_id, inner=live_inner, cassette_dir=Path(cassette_dir))
    estimated_usd = estimate_generation_cost_usd(cases, "openai", avg_output_tokens=avg_output_tokens)
    return GeneratorComponent(
        component_id=f"openai-{model_id}",
        model_snapshot={"provider": entry["provider"], "model_id": entry["model_id"], "revision": entry["revision"]},
        gateway=gateway,
        estimated_usd=estimated_usd,
    )


# ---- variant comparison stage (naive vs agentic vs graph) ----------------------------------------


def _live_ollama_model(model_id: str) -> BaseChatModel:  # pragma: no cover - live only, needs a running Ollama daemon
    from replay.providers import build_chat_model

    return build_chat_model("ollama", model_id)


def _live_chat_model_for(provider: str, model_id: str) -> BaseChatModel:  # pragma: no cover - live only
    if provider == "anthropic":
        return _live_anthropic_model(model_id)
    if provider == "openai":
        return _live_gpt_model(model_id)
    if provider == "ollama":
        return _live_ollama_model(model_id)
    raise ValueError(f"unknown provider {provider!r} (use anthropic | openai | ollama)")


def construct_live_pg_knowledge_graph(*, pg_dsn: Optional[str]) -> KnowledgeGraph:  # pragma: no cover - live only
    resolved_dsn = pg_dsn or require_env(
        "ATLAS_PG_DSN", purpose="constructing the live PgKnowledgeGraph for the variant comparison stage"
    )
    from atlas.adapters.pg_knowledge_graph import PgKnowledgeGraph

    return PgKnowledgeGraph(pg_dsn=resolved_dsn)


def build_variants_config(
    *,
    cases: Sequence[MatrixCase],
    gate: SpendGate,
    provider: str,
    cassette_dir: str | Path,
    retriever: Optional[Retriever] = None,
    reranker: Optional[Reranker] = None,
    graph: Optional[KnowledgeGraph] = None,
    inner: Optional[BaseChatModel] = None,
    k: int = 3,
    avg_output_tokens: int = _DEFAULT_AVG_OUTPUT_TOKENS,
    pg_dsn: Optional[str] = None,
    tei_embed_url: Optional[str] = None,
    tei_rerank_url: Optional[str] = None,
    index_dir: Optional[str | Path] = None,
    models_lock_path: Path = MODELS_LOCK_PATH,
) -> tuple[Optional[VariantsConfig], SpendGate, Optional[str]]:
    """Build the naive/agentic/graph comparison stage's shared fixtures for `provider`, PRE spend
    checked: `matrix.runner.run_matrix`'s own variant stage never gates this stage's generator calls
    at all (only stage 3's generator axis goes through `spend_gate` there), so left unchecked, 3
    variants times `len(cases)` real generate calls would run with NO ceiling protection. This
    function is the driver's own pre-check, refusing to build a paid `VariantsConfig` at all when
    the estimated 3-variant cost would exceed `provider`'s remaining budget in `gate` -- the same
    refusal `matrix.spend_gate.check_spend` already renders for stage 3's own cells. `ollama`
    (`matrix.spend_gate.ALWAYS_RUNS`) skips the pre-check entirely: it always runs, free, by
    construction.

    CROSS STAGE RECONCILIATION (closes the review's Important finding): an admitted estimate is
    folded into the returned gate via `matrix.spend_gate.record_spend` before this function returns
    -- never left as a read only `check_spend` against the pristine gate a caller happened to still
    be holding. `matrix.runner.run_matrix`'s own stage 3 admission checks the SAME provider's
    remaining budget independently; if it were handed the untouched gate this function was given
    (not the one it returns), the variant stage's own real spend would be invisible to that later
    check, and the two admissions could each independently report "fits" while their SUM silently
    exceeds the hard ceiling. The caller (`matrix.__main__.main`) MUST thread the returned gate --
    not the one it passed in -- into `run_matrix(..., spend_gate=...)` for the ceiling to hold across
    both stages. Returns `(VariantsConfig, updated_gate, None)` when admitted (`updated_gate` carries
    the recorded spend), `(None, gate, reason)` when refused (`gate` returned unchanged: a refusal
    spends nothing) -- never raises for a refusal, since "this provider's budget would not cover the
    variant stage" is an ordinary, expected outcome of a live run, not a caller error. `ollama` also
    returns `gate` unchanged: it is priced at zero by construction, so there is nothing to record."""
    if provider not in ALWAYS_RUNS:
        estimate = 3 * estimate_generation_cost_usd(cases, provider, avg_output_tokens=avg_output_tokens)
        decision = check_spend(gate, provider, estimate)
        if not decision.allowed:
            return None, gate, decision.reason
        gate = record_spend(gate, provider, estimate)

    entry = _generator_lock_entry(models_lock_path, provider=provider)
    model_id = entry["model_id"]
    live_inner = inner if inner is not None else _live_chat_model_for(provider, model_id)
    gateway = build_generator_gateway(provider=provider, model_id=model_id, inner=live_inner, cassette_dir=Path(cassette_dir))

    if retriever is None:
        retriever = construct_live_pgvector_retriever(
            pg_dsn=pg_dsn, tei_embed_url=tei_embed_url, tei_rerank_url=tei_rerank_url, index_dir=index_dir,
        )
    if reranker is None:
        reranker = construct_live_tei_reranker(tei_rerank_url=tei_rerank_url)
    if graph is None:
        graph = construct_live_pg_knowledge_graph(pg_dsn=pg_dsn)

    return VariantsConfig(retriever=retriever, reranker=reranker, graph=graph, gateway=gateway, k=k), gate, None


def build_judge_gateway(
    provider: str,
    *,
    cassette_dir: str | Path,
    inner: Optional[BaseChatModel] = None,
    models_lock_path: Path = MODELS_LOCK_PATH,
) -> BaseChatModel:
    """One judge panel member (D15's cross-provider jury, `judge.panel.panel_vote`'s own module
    docstring: "D15's own 3 model cross provider vote"). Reuses `models.lock`'s own generator entry
    for `provider` -- never a fourth model family this repo has no key for -- tagged `judge-<model>`
    (`matrix.spend_gate.build_generator_gateway`'s own `provider:model_id` shape becomes
    `provider:judge-model_id`) so its cassette key is always distinct from that SAME model's own
    generator cell. `inner` injects a stub `BaseChatModel` for hermetic tests, skipping all live
    construction; omitted, builds the real chat model the SAME way that provider's own generator
    builder does. Always RECORD mode, matching every other gateway this package builds."""
    entry = _generator_lock_entry(models_lock_path, provider=provider)
    model_id = entry["model_id"]
    live_inner = inner if inner is not None else _live_chat_model_for(provider, model_id)
    return build_generator_gateway(
        provider=provider, model_id=f"judge-{model_id}", inner=live_inner, cassette_dir=Path(cassette_dir),
    )


__all__ = [
    "MissingEnvVarError",
    "build_baseline_embedder_components",
    "build_bge_m3_embedder_component",
    "build_claude_generator_component",
    "build_gpt_generator_component",
    "build_judge_gateway",
    "build_openai_embedder_component",
    "build_reranker_components",
    "build_variants_config",
    "construct_live_openai_embedding_client",
    "construct_live_pg_knowledge_graph",
    "construct_live_pgvector_retriever",
    "construct_live_tei_reranker",
    "estimate_generation_cost_usd",
    "require_env",
]
