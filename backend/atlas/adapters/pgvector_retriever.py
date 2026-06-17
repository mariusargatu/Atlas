"""The hybrid pgvector adapter (D8/D9): the real vector adapter that sits behind the knowledge port
in dev/prod, once wired (SP3 task 7). Two client boundaries only, matching `rag_tools.ingest`'s
discipline: `httpx` to TEI (embed + rerank) and `psycopg` to Postgres (pgvector HNSW + generated
tsvector). `InMemoryRetriever` stays the hermetic CI adapter; this one is exercised live, against
the compose stack, and through the hermetic tests in this module's own test file (a recording fake
connection + `httpx.MockTransport`, no Docker).

Per search: embed the query via TEI, run the pgvector HNSW arm and the tsvector `websearch_to_tsquery`
arm to `config.k_fused` each in ONE Postgres connection, fuse the two ranked id lists with
`domain.retrieval.rrf_fuse`, then either rerank the fused top `config.k_fused` via TEI (scores land on
`Chunk.score`) or return the fused top `config.k_final` with the RRF scores. `config.exact_scan`
swaps the HNSW `ef_search` knob for a forced sequential scan on the vector arm (`enable_indexscan =
off`), the recall ground truth mode. Both SQL arms filter `WHERE index_build_id = %(build_id)s` (the
`chunks` table can hold more than one build's rows; see `rag_tools.ingest.create_schema`) and order by
`, chunk_id` last, a deterministic tie break (SP3 final review: a live probe found 13 way ties at
identical `ts_rank` on this corpus).

Fail closed at construction (D9): the adapter reads `fingerprint.json` and `build_manifest.json` from
the active index dir and refuses to build if the fingerprint does not match what the live TEI server
reports over `/info`, so a misconfigured deployment (wrong model, wrong revision) never serves
silently-wrong embeddings; `build_manifest.json`'s `index_build_id` becomes the `build_id` bound into
every search's SQL, scoping this adapter to exactly one index build.

NOTE on the import lint (`testing/tests/test_import_lint.py`): backend must never import harness code
(`rag_tools`/`corpus_tools`/`quality`/`matrix`/...), one way only, harness -> backend. The fingerprint
read below (`_load_fingerprint`) therefore restates a few lines of `rag_tools.fingerprint` rather than
importing it. The L2 normalize / pgvector literal helpers no longer duplicate anything: they live in
`atlas.domain.retrieval` (pure, importable from BOTH directions) and both this adapter and
`rag_tools.ingest` import them from there. The three SQL arms and `row_to_chunk` are public on this
module for the same reason: `matrix.live_search` imports them rather than carrying a second copy.
"""
from __future__ import annotations

import contextvars
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
import psycopg

from atlas.adapters.resilience import (
    CircuitBreaker,
    EmbeddingServiceError,
    RerankServiceError,
    RetrievalError,
    RetryPolicy,
    call_with_resilience,
    last_call_retried,
)
from atlas.config import DEFAULT_INDEX_DIR, DEFAULT_PG_DSN, DEFAULT_TEI_EMBED_URL, DEFAULT_TEI_RERANK_URL
from atlas.domain.retrieval import RRF_K, RetrievalConfig, l2_normalize, rrf_fuse, vector_literal
from atlas.ports.knowledge import Chunk

# The local development defaults come from `atlas.config`, which names each of them once (this
# module and `pg_knowledge_graph.py` both used to carry their own copy of the same DSN literal, and
# `config.AtlasSettings` already declared the two TEI urls and the index dir a third time while its
# own docstring described them as "captured but not threaded"). The env var each maps to and this
# module's own fail fast validation are unchanged.

_TIMEOUT_SECONDS = 30.0  # /info and /embed: one short text, fast even on the Candle CPU backend
# /rerank measured ~22s for one 32-text batch on this machine's Rosetta emulated Candle CPU backend
# (docker-compose.yml documents the same backend as slow under emulation); a generous ceiling well
# above that, not the shared 30s used for /info and /embed's much smaller payloads.
_RERANK_TIMEOUT_SECONDS = 120.0
# TEI enforces a `max_client_batch_size` per `/rerank` request; this is the value BAAI/bge-reranker
# -v2-m3 advertises. A server enforcing a different ceiling is served by passing `rerank_batch_size`
# at construction (the same injection seam `connect`/`clock`/the httpx clients already use), which
# is how `matrix.live_search.TeiReranker` has always taken it. This replaced a construction time
# `/info` probe that fell back to this same literal on any failure, and could block boot for up to
# the rerank client's 120s timeout against a reachable but hanging server, to learn a number that
# only ever costs a few extra requests when wrong.
_RERANK_BATCH_SIZE = 32

# Both arms carry two WHERE/ORDER BY concerns fixed by the SP3 final review:
# (1) `index_build_id = %(build_id)s` scopes every query to ONE index build. `chunks` is a single
#     physical table that can hold rows from more than one build (`rag_tools.ingest.create_schema`'s
#     own comment has the schema side of this), so without this filter a rebuild that leaves an
#     older build's rows in place would silently blend two builds' candidates into one ranking.
#     `chunk_id` is only unique ACROSS builds that differ in corpus_version, since that is one of
#     the values hashed into it (`rag_tools.chunker._compute_chunk_id`); two builds of the SAME
#     corpus_version that differ only in something the hash does not see, an embedding model change
#     being the real example, collide on every chunk_id at load time. `rag_tools.ingest.load_parquet`
#     runs a post load row count check for exactly this reason (SP4 task 3): a build whose rows all
#     collided with an earlier build's raises loud instead of quietly serving an empty index under
#     this build_id.
# (2) `, chunk_id` is a deterministic tie break appended after the real ranking key. A live probe
#     against this corpus found 13 way ties at identical `ts_rank` (many chunks share the exact same
#     lexical overlap against a query) and vector distance ties are just as possible; `LIMIT` over an
#     unstable order can silently return a different row set across otherwise identical runs. The
#     vector arm's `<=>` distance is already ascending by default (smaller distance ranks first), so
#     its tie break needs no explicit `ASC`; the tsv arm's `ts_rank DESC` does.
VECTOR_ARM_SQL = """
    SELECT chunk_id
    FROM chunks
    WHERE index_build_id = %(build_id)s
    ORDER BY embedding <=> %(vector)s::vector ASC, chunk_id
    LIMIT %(k)s;
"""

TSV_ARM_SQL = """
    SELECT chunk_id
    FROM chunks
    WHERE index_build_id = %(build_id)s AND tsv @@ websearch_to_tsquery('english', %(query)s)
    ORDER BY ts_rank(tsv, websearch_to_tsquery('english', %(query)s)) DESC, chunk_id
    LIMIT %(k)s;
"""

HYDRATE_SQL = """
    SELECT chunk_id, parent_id, doc_id, doc_version, doc_type, heading_path,
           char_span_start, char_span_end, text, entity_ids
    FROM chunks
    WHERE chunk_id = ANY(%(chunk_ids)s);
"""


@dataclass(frozen=True)
class SearchResult:
    """Per call inspection carrier (SP4 task 3), replacing the old module level `last_scores`
    global's single caller assumption. `chunks` is what the most recent `search_chunks` call in
    THIS execution context returned; `fused_scores` are the pre rerank fused (chunk_id, score)
    pairs for the up to `k_fused` candidates fusion produced, best first, regardless of whether
    reranking then ran; `rerank_scores` are the reranker's own (chunk_id, score) pairs, best first,
    or None when `config.rerank_enabled` was False. Adapter level, not the `Retriever` port's own
    contract (`search_chunks` still returns `list[Chunk]`); SP6 will read this for tracing.

    `retried` (SP4 task 4, the degradation ladder's retry rung carrier) is True when at least one
    of this call's underlying provider calls (embed, the pg arms, rerank) needed more than one
    attempt to succeed: `resilience.last_call_retried()`, read immediately after each such call so
    it reflects THAT call, never a stale value from an earlier one. Defaults False so every
    existing construction (a bare `SearchResult()`) is unaffected."""

    chunks: tuple[Chunk, ...] = ()
    fused_scores: tuple[tuple[str, float], ...] = ()
    rerank_scores: tuple[tuple[str, float], ...] | None = None
    retried: bool = False


# Contextvar, not a plain module global (the bug the old `last_scores` global had: one caller's
# result could stomp another's mid flight). A `contextvars.ContextVar` is isolated per thread by
# default and per `asyncio` task once a task copies its context at creation, so two `search_chunks`
# calls interleaved across threads or tasks each see only their own result here.
_search_result: contextvars.ContextVar[SearchResult | None] = contextvars.ContextVar(
    "atlas_pgvector_search_result", default=None
)


def last_result() -> SearchResult | None:
    """Inspection only accessor for the most recent `search_chunks` call in THIS execution context.
    Returns None before any call has run in this context. Never read this to make a request scoped
    decision inside the adapter itself; it exists for tracing (SP6) and tests, not application
    logic."""
    return _search_result.get()


class FingerprintMismatchError(RuntimeError):
    """Raised at construction when the active index's `fingerprint.json` does not match what the
    live TEI embedding server reports over `/info` (D9 fail closed discipline)."""


class PgvectorRetriever:
    """The `Retriever` port (`atlas.ports.knowledge.Retriever`) over real Postgres (pgvector HNSW +
    generated tsvector) and real TEI (embed + rerank). `k` (the port's own parameter) caps the final
    returned count; `config.k_final` is the internal width the adapter aims to fill (after optional
    reranking) before that cap applies, so a caller like the knowledge MCP server that asks for
    `k=3` against a wider `config.k_final=5` gets at most 3 back, best first.
    """

    def __init__(
        self,
        *,
        pg_dsn: str | None = None,
        tei_embed_url: str | None = None,
        tei_rerank_url: str | None = None,
        index_dir: str | Path | None = None,
        embed_client: httpx.Client | None = None,
        rerank_client: httpx.Client | None = None,
        connect: Callable[[], Any] | None = None,
        clock: Callable[[], float] | None = None,
        rerank_batch_size: int = _RERANK_BATCH_SIZE,
    ) -> None:
        self._pg_dsn = pg_dsn or os.environ.get("ATLAS_PG_DSN", DEFAULT_PG_DSN)
        tei_embed_url = tei_embed_url or os.environ.get("ATLAS_TEI_EMBED_URL", DEFAULT_TEI_EMBED_URL)
        tei_rerank_url = tei_rerank_url or os.environ.get("ATLAS_TEI_RERANK_URL", DEFAULT_TEI_RERANK_URL)
        index_dir = index_dir or os.environ.get("ATLAS_INDEX_DIR", DEFAULT_INDEX_DIR)
        self._index_dir = Path(index_dir)

        self._embed_client = embed_client or httpx.Client(base_url=tei_embed_url, timeout=_TIMEOUT_SECONDS)
        self._rerank_client = rerank_client or httpx.Client(base_url=tei_rerank_url, timeout=_RERANK_TIMEOUT_SECONDS)
        self._connect = connect or (lambda: psycopg.connect(self._pg_dsn))

        # Resilience wiring (SP4 task 3): one breaker shared across every provider key this adapter
        # calls through ("tei-embed", "tei-rerank", "postgres"), a policy per call shape (rerank's
        # own client timeout is 4x the others', so its retry stage deadline follows suit). `clock`
        # defaults to `time.monotonic` ONLY here, the live wiring edge, the same pattern `connect`
        # follows just above (Global Constraints: the breaker clock comes from an injected callable,
        # defaulting only at the live wiring edge, never inside `resilience.CircuitBreaker` itself).
        self._clock = clock or time.monotonic
        self._breaker = CircuitBreaker(self._clock)
        self._embed_retry_policy = RetryPolicy(stage_deadline_seconds=_TIMEOUT_SECONDS)
        self._rerank_retry_policy = RetryPolicy(stage_deadline_seconds=_RERANK_TIMEOUT_SECONDS)
        self._pg_retry_policy = RetryPolicy(stage_deadline_seconds=_TIMEOUT_SECONDS)

        fp = self._load_fingerprint()
        manifest = self._load_build_manifest()
        self._verify_fingerprint(fp)
        self._normalize: bool = fp["normalize"]
        self._query_prefix: str = fp["query_prefix"]
        self._build_id: str = manifest["index_build_id"]
        self._rerank_batch_size: int = rerank_batch_size

    @property
    def breaker(self) -> CircuitBreaker:
        """Inspection only (SP6 task 5): the ONE `CircuitBreaker` this adapter shares across every
        provider key it calls through ("tei-embed", "tei-rerank", "postgres"). `server.py`'s own
        `/metrics` route reads `.state(provider_key)` off this at scrape time to render
        `atlas_circuit_breaker_state` -- never re derived, per `CircuitBreaker.state()`'s own
        docstring ("Inspection only (tests, SP6 tracing later)") and
        `adapters/trace_translation.py`'s own module docstring, which names this exact metrics
        surface as where breaker state belongs (not a span attribute). Read only: nothing outside
        this adapter's own call sites ever calls `before_call`/`record_success`/`record_failure`."""
        return self._breaker

    # --- construction: fail closed fingerprint + build id checks ------------------------------------

    def _load_fingerprint(self) -> dict:
        # Inline JSON read, not `rag_tools.fingerprint.from_models_lock`/`EmbeddingFingerprint`: see
        # this module's docstring on the import lint boundary.
        path = self._index_dir / "fingerprint.json"
        if not path.is_file():
            # Fail closed BEFORE any network call (SP3 task 7 ride along): a typo'd or unmounted
            # ATLAS_INDEX_DIR must raise one worded, actionable error, not a raw FileNotFoundError
            # or (worse) a later KeyError from _verify_fingerprint reading a TEI response nobody sent.
            raise FingerprintMismatchError(
                f"No fingerprint.json found at {path} (ATLAS_INDEX_DIR={self._index_dir}). Point "
                "ATLAS_INDEX_DIR at a built index directory (containing chunks.parquet + "
                "fingerprint.json) -- e.g. the committed indexes/<name>/ tree at the repo root, or "
                "build one with `task rag:ingest`."
            )
        return json.loads(path.read_text())

    def _load_build_manifest(self) -> dict:
        # Read alongside fingerprint.json, filesystem only, before the one network call
        # (_verify_fingerprint) -- same fail closed ordering as the fingerprint check itself (SP3
        # final review, table scoping): `index_build_id` is the WHERE filter both SQL arms bind so
        # this adapter's queries only ever see rows from the build ATLAS_INDEX_DIR actually names,
        # never a stale or differently-built row left behind in the same physical `chunks` table.
        path = self._index_dir / "build_manifest.json"
        if not path.is_file():
            raise FingerprintMismatchError(
                f"No build_manifest.json found at {path} (ATLAS_INDEX_DIR={self._index_dir}). Point "
                "ATLAS_INDEX_DIR at a built index directory (containing chunks.parquet + "
                "fingerprint.json + build_manifest.json) -- e.g. the committed indexes/<name>/ tree "
                "at the repo root, or build one with `task rag:ingest`."
            )
        return json.loads(path.read_text())

    def _verify_fingerprint(self, fp: dict) -> None:
        def do_request() -> httpx.Response:
            response = self._embed_client.get("/info")
            response.raise_for_status()
            return response

        response = call_with_resilience(
            do_request,
            policy=self._embed_retry_policy,
            breaker=self._breaker,
            provider_key="tei-embed",
            error_type=EmbeddingServiceError,
        )
        info = response.json()
        server_model_id = info.get("model_id")
        server_revision = info.get("model_sha")
        if server_model_id != fp["model_id"] or server_revision != fp["revision"]:
            raise FingerprintMismatchError(
                f"Index fingerprint ({self._index_dir / 'fingerprint.json'}) names "
                f"{fp['model_id']!r}@{fp['revision']!r}, which does not match the live TEI server's "
                f"/info ({server_model_id!r}@{server_revision!r}). Refusing to construct "
                "PgvectorRetriever (fail closed, D9): rebuild the index against this server, or point "
                "ATLAS_TEI_EMBED_URL at the server the committed index was actually built from."
            )

    # --- lifecycle ------------------------------------------------------------------------------------

    def close(self) -> None:
        """Close the two httpx clients (embed + rerank). SP3 task 7 ride along: the served
        entrypoint (`server.py`) constructs ONE `PgvectorRetriever` for the app's whole lifetime
        (never per request) and calls this once at shutdown; nothing else about the adapter holds a
        resource across calls (Postgres connections are opened and closed per `search_chunks` call,
        see below)."""
        self._embed_client.close()
        self._rerank_client.close()

    # --- TEI client boundaries ----------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        text = f"{self._query_prefix}{query}" if self._query_prefix else query

        def do_request() -> httpx.Response:
            response = self._embed_client.post("/embed", json={"inputs": [text]})
            response.raise_for_status()
            return response

        response = call_with_resilience(
            do_request,
            policy=self._embed_retry_policy,
            breaker=self._breaker,
            provider_key="tei-embed",
            error_type=EmbeddingServiceError,
        )
        [vector] = response.json()
        return l2_normalize(vector) if self._normalize else list(vector)

    def _rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        # TEI enforces `max_client_batch_size` (`self._rerank_batch_size`) per request; a fused width
        # (`config.k_fused`) above that on this 45 chunk corpus 422s a single unbatched call, caught
        # by the live lane (MockTransport in the hermetic tests does not enforce this limit). Batch
        # and remap each batch's request local `index` back to the caller's full `texts` offsets.
        scores = [0.0] * len(texts)
        for start in range(0, len(texts), self._rerank_batch_size):
            batch = list(texts[start : start + self._rerank_batch_size])

            def do_request(batch: list[str] = batch) -> httpx.Response:
                response = self._rerank_client.post("/rerank", json={"query": query, "texts": batch})
                response.raise_for_status()
                return response

            response = call_with_resilience(
                do_request,
                policy=self._rerank_retry_policy,
                breaker=self._breaker,
                provider_key="tei-rerank",
                error_type=RerankServiceError,
            )
            for item in response.json():
                scores[start + item["index"]] = item["score"]
        return scores

    # --- the port method -----------------------------------------------------------------------------

    def search_chunks(self, query: str, k: int, config: RetrievalConfig) -> list[Chunk]:
        retried = False
        query_vector: list[float] | None = None
        if not config.lexical_only:
            query_vector = self._embed_query(query)
            retried = retried or last_call_retried()
        fused_top, rows_by_id = self._run_sql_arms(query, query_vector, config)
        retried = retried or last_call_retried()

        if not fused_top:
            _search_result.set(SearchResult(retried=retried))
            return []

        chunks, rerank_scores = self._finalize(query, fused_top, rows_by_id, config)
        if config.rerank_enabled:
            retried = retried or last_call_retried()
        chunks = chunks[:k]
        _search_result.set(
            SearchResult(
                chunks=tuple(chunks), fused_scores=tuple(fused_top), rerank_scores=rerank_scores, retried=retried
            )
        )
        return chunks

    def _run_sql_arms(
        self, query: str, query_vector: list[float] | None, config: RetrievalConfig
    ) -> tuple[list[tuple[str, float]], dict[str, Chunk]]:
        """Both SQL arms plus hydration, in ONE Postgres connection, wrapped in the retry policy +
        breaker for the `postgres` provider key (SP4 task 3). A fresh connection is opened per
        attempt (`self._connect()` is called again on every retry, never reused across attempts: a
        connection that just failed is not trusted to still be good). `config.lexical_only` (SP4
        task 4, the degradation ladder's embedding down rung) skips the vector arm's `SET LOCAL`
        knob AND its `SELECT` entirely: `query_vector` is None on that path (the caller never even
        called TEI to produce one), and the fused ranking degrades gracefully to the tsv arm alone
        (an empty ranking contributes nothing to `rrf_fuse`)."""

        def attempt() -> tuple[list[tuple[str, float]], dict[str, Chunk]]:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    vector_ids: list[str] = []
                    if not config.lexical_only:
                        if config.exact_scan:
                            cur.execute("SET LOCAL enable_indexscan = off;")
                        else:
                            cur.execute(_ef_search_statement(config.ef_search))

                        cur.execute(
                            VECTOR_ARM_SQL,
                            {"vector": vector_literal(query_vector), "k": config.k_fused, "build_id": self._build_id},
                        )
                        vector_ids = [row[0] for row in cur.fetchall()]

                    cur.execute(TSV_ARM_SQL, {"query": query, "k": config.k_fused, "build_id": self._build_id})
                    tsv_ids = [row[0] for row in cur.fetchall()]

                    fused = rrf_fuse([tuple(vector_ids), tuple(tsv_ids)], k=RRF_K)
                    fused_top = fused[: config.k_fused]

                    if not fused_top:
                        conn.commit()
                        return fused_top, {}

                    cur.execute(HYDRATE_SQL, {"chunk_ids": [chunk_id for chunk_id, _ in fused_top]})
                    rows_by_id = {row[0]: row_to_chunk(row) for row in cur.fetchall()}
                conn.commit()
                return fused_top, rows_by_id
            finally:
                conn.close()

        return call_with_resilience(
            attempt,
            policy=self._pg_retry_policy,
            breaker=self._breaker,
            provider_key="postgres",
            error_type=RetrievalError,
        )

    def _finalize(
        self,
        query: str,
        fused_top: list[tuple[str, float]],
        rows_by_id: dict[str, Chunk],
        config: RetrievalConfig,
    ) -> tuple[list[Chunk], tuple[tuple[str, float], ...] | None]:
        rerank_scores: tuple[tuple[str, float], ...] | None
        if config.rerank_enabled:
            ordered_ids = [chunk_id for chunk_id, _ in fused_top]
            texts = [rows_by_id[chunk_id].text for chunk_id in ordered_ids]
            raw_scores = self._rerank(query, texts)
            scored = sorted(zip(ordered_ids, raw_scores), key=lambda pair: (-pair[1], pair[0]))
            rerank_scores = tuple(scored)
        else:
            scored = fused_top
            rerank_scores = None
        final = scored[: config.k_final]
        chunks = [replace(rows_by_id[chunk_id], score=score) for chunk_id, score in final]
        return chunks, rerank_scores


# --- pure helpers (duplicated in miniature from rag_tools.ingest; see the module docstring) --------


def _ef_search_statement(ef_search: int) -> str:
    # `SET` does not accept bind parameters (Postgres restricts it to literal constant expressions,
    # the same reason `rag_tools.ingest.create_schema` interpolates `dim`/`m`/`ef_construction`
    # directly); `ef_search` comes from this call's own typed, internally sourced `RetrievalConfig`,
    # never from external input, so validate-then-interpolate is safe here.
    if not isinstance(ef_search, int) or ef_search <= 0:
        raise ValueError(f"config.ef_search must be a positive int, got {ef_search!r}")
    return f"SET LOCAL hnsw.ef_search = {ef_search};"


def row_to_chunk(row: Sequence[Any]) -> Chunk:
    (chunk_id, parent_id, doc_id, doc_version, doc_type, heading_path, char_span_start, char_span_end, text, entity_ids) = row
    return Chunk(
        chunk_id=chunk_id,
        parent_id=parent_id,
        doc_id=doc_id,
        doc_version=doc_version,
        doc_type=doc_type,
        heading_path=tuple(heading_path),
        char_span=(char_span_start, char_span_end),
        text=text,
        entity_ids=tuple(entity_ids),
        score=0.0,  # filled in by `_finalize` (fused or rerank score); a placeholder until then
    )
