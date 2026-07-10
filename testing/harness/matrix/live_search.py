"""Harness-side live search adapters the matrix live driver (`matrix.live_driver`) wires the real
embedder/reranker axes through -- SP9 task 4's own deferred "live caller" (`matrix.embedders`'s own
module docstring: "a live caller ... wires the SAME shape to `atlas.adapters.pgvector_retriever.
PgvectorRetriever.search_chunks` bound to a real `EmbeddingClient` per axis instead"). Both classes
here are small and hermetically testable via constructor injection (an `httpx.Client`, a `connect`
callable, an `EmbeddingClient`): neither ever makes a network call unless a caller genuinely omits
the injected dependency.

`TeiReranker` -- the `Reranker` port (`atlas.ports.reranker.Reranker`) over a live TEI `/rerank`
server. `PgvectorRetriever` already makes this exact call internally (its own private `_rerank`/
`_finalize`), but only as part of its own `search_chunks`, never as a standalone `Reranker` a second
caller (the matrix's stage 2 axis, or the variant comparison stage's shared reranker) can reuse.
This class is that standalone wiring, its first real caller. Stable sort, ties keep the caller's own
input order (`Reranker`'s own documented contract, the same rule
`atlas.adapters.cassette_reranker.CassetteReranker` already honors for its REPLAY axis).

`OpenAiEmbeddedRetriever` -- a minimal hybrid (vector + tsvector, RRF fused) search over the SAME
`chunks` Postgres table `PgvectorRetriever` searches, bound to an injectable
`atlas.ports.embedding.EmbeddingClient` instead of a hardcoded TEI call. `PgvectorRetriever` cannot
serve SP9 task 3's second embedder axis (`text-embedding-3-small`) directly: both its fingerprint
verification (`_verify_fingerprint`, a live TEI `/info` call) and its own `_embed_query` are
hardwired to TEI's specific HTTP contract (`POST /embed`, `{"inputs": [...]}`), never a pluggable
`EmbeddingClient` -- confirmed by reading that adapter's source (SP9's own planning digest names the
embedding client port as new territory `PgvectorRetriever` was never updated to consume). A second,
genuinely different embedder therefore needs its own thin search path; duplicating the small
vector-arm/tsv-arm/hydrate SQL here (rather than importing `PgvectorRetriever`'s own private,
underscore-named module constants) is the SAME deliberate-duplication call that adapter's own module
docstring already makes for the identical reason ("the fingerprint read ... duplicate[s] a few
lines ... rather than importing them; the duplication is deliberate, not an oversight").

Neither adapter retries or trips a circuit breaker (`atlas.adapters.resilience`'s job for the served
retrieval path): both back a live/operator BENCHMARK run, never the served runtime graph, so a
transient failure here should fail the benchmark cell loud, not degrade gracefully the way a served
customer turn must. A documented, scoped simplification, not an oversight.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import replace as _dc_replace
from typing import Any, Optional

import httpx

from atlas.domain.retrieval import rrf_fuse
from atlas.ports.embedding import EmbeddingClient
from atlas.ports.knowledge import Chunk

_RERANK_BATCH_SIZE = 32
_RERANK_TIMEOUT_SECONDS = 120.0


class TeiReranker:
    """`Reranker` over a live TEI cross-encoder server's `/rerank` endpoint. `client` is an
    injectable `httpx.Client` (hermetic tests inject an `httpx.MockTransport`-backed one, matching
    `PgvectorRetriever`'s own injectable-client convention); omitted, this instance owns -- and
    later closes via `close()` -- a real one built against `base_url`."""

    def __init__(
        self, *, base_url: str = "", client: Optional[httpx.Client] = None, batch_size: int = _RERANK_BATCH_SIZE,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=base_url, timeout=_RERANK_TIMEOUT_SECONDS)
        self._batch_size = batch_size

    def _scores(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = [0.0] * len(texts)
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = self._client.post("/rerank", json={"query": query, "texts": batch})
            response.raise_for_status()
            for item in response.json():
                scores[start + item["index"]] = item["score"]
        return scores

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []
        scores = self._scores(query, [c.text for c in chunks])
        paired = list(zip(chunks, scores))
        paired.sort(key=lambda pair: -pair[1])  # stable sort: ties keep the caller's own input order
        return [chunk for chunk, _ in paired]

    def close(self) -> None:
        """Close the owned `httpx.Client`. An injected `client` is the caller's own to close, left
        untouched here (matching `PgvectorRetriever.close()`'s own injected-vs-owned convention)."""
        if self._owns_client:
            self._client.close()


# The vector/tsv arms and the hydrate projection are IDENTICAL, by table and column, to
# `atlas.adapters.pgvector_retriever`'s own private `_VECTOR_ARM_SQL`/`_TSV_ARM_SQL`/`_HYDRATE_SQL`
# (same `chunks` table, same `index_build_id` scoping discipline, same `, chunk_id` tie break -- see
# that module's own docstring on the deterministic tie break a live probe against this corpus
# needed). Restated here, not imported, per this module's own docstring.
_VECTOR_ARM_SQL = """
    SELECT chunk_id
    FROM chunks
    WHERE index_build_id = %(build_id)s
    ORDER BY embedding <=> %(vector)s::vector ASC, chunk_id
    LIMIT %(k)s;
"""

_TSV_ARM_SQL = """
    SELECT chunk_id
    FROM chunks
    WHERE index_build_id = %(build_id)s AND tsv @@ websearch_to_tsquery('english', %(query)s)
    ORDER BY ts_rank(tsv, websearch_to_tsquery('english', %(query)s)) DESC, chunk_id
    LIMIT %(k)s;
"""

_HYDRATE_SQL = """
    SELECT chunk_id, parent_id, doc_id, doc_version, doc_type, heading_path,
           char_span_start, char_span_end, text, entity_ids
    FROM chunks
    WHERE chunk_id = ANY(%(chunk_ids)s);
"""


def _l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return list(vector)
    return [component / norm for component in vector]


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def _row_to_chunk(row: Sequence[Any]) -> Chunk:
    (
        chunk_id, parent_id, doc_id, doc_version, doc_type, heading_path,
        char_span_start, char_span_end, text, entity_ids,
    ) = row
    return Chunk(
        chunk_id=chunk_id, parent_id=parent_id, doc_id=doc_id, doc_version=doc_version, doc_type=doc_type,
        heading_path=tuple(heading_path), char_span=(char_span_start, char_span_end), text=text,
        entity_ids=tuple(entity_ids), score=0.0,  # filled in below from the fused RRF score
    )


class OpenAiEmbeddedRetriever:
    """A minimal hybrid (vector + tsvector, RRF fused) search over the `chunks` table, scoped to
    one `index_build_id`, bound to an injectable `EmbeddingClient` (a real `OpenAiEmbeddingClient`
    live, a stub in every hermetic test) instead of `PgvectorRetriever`'s own hardcoded TEI call.
    `connect` mirrors `PgvectorRetriever`'s/`PgKnowledgeGraph`'s own injectable-callable seam: a
    recording fake in tests, `lambda: psycopg.connect(pg_dsn)` live -- a fresh connection is opened
    per call and always closed, never reused across calls. No rerank step (this axis is
    embedder-only, matching `matrix.embedders`'s own contract: reranking is stage 2's job); no
    retry/circuit-breaker wiring (see the module docstring)."""

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        index_build_id: str,
        connect: Callable[[], Any],
        normalize: bool = True,
        query_prefix: str = "",
        rrf_k: int = 60,
    ) -> None:
        self._embedding_client = embedding_client
        self._build_id = index_build_id
        self._connect = connect
        self._normalize = normalize
        self._query_prefix = query_prefix
        self._rrf_k = rrf_k

    def _embed_query(self, query: str) -> list[float]:
        text = f"{self._query_prefix}{query}" if self._query_prefix else query
        [vector] = self._embedding_client.embed_texts([text])
        return _l2_normalize(vector) if self._normalize else list(vector)

    def search_chunks(self, query: str, k: int) -> list[Chunk]:
        vector = self._embed_query(query)
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(_VECTOR_ARM_SQL, {"vector": _vector_literal(vector), "k": k, "build_id": self._build_id})
                vector_ids = [row[0] for row in cur.fetchall()]
                cur.execute(_TSV_ARM_SQL, {"query": query, "k": k, "build_id": self._build_id})
                tsv_ids = [row[0] for row in cur.fetchall()]
                fused = rrf_fuse([tuple(vector_ids), tuple(tsv_ids)], k=self._rrf_k)[:k]
                if not fused:
                    conn.commit()
                    return []
                cur.execute(_HYDRATE_SQL, {"chunk_ids": [chunk_id for chunk_id, _ in fused]})
                rows_by_id = {row[0]: _row_to_chunk(row) for row in cur.fetchall()}
            conn.commit()
            return [
                _dc_replace(rows_by_id[chunk_id], score=score)
                for chunk_id, score in fused
                if chunk_id in rows_by_id
            ]
        finally:
            conn.close()


__all__ = ["OpenAiEmbeddedRetriever", "TeiReranker"]
