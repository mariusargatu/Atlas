"""`TeiEmbeddingClient`: the `EmbeddingClient` port's local TEI adapter (D9's first embedder, BGE-M3).

A thin `httpx` wrapper over TEI's `POST /embed`, batched, in input order, no retries (a batch
failure raises immediately rather than silently skipping or padding a failed batch, D9's fail closed
discipline). This is a PARALLEL implementation of `testing.harness.rag_tools.ingest.embed_texts`'s
same small call, not a shared one: the agent harness may never import the eval harness (`testing.*`,
`test_import_lint.py::test_agent_harness_never_imports_the_eval_harness`), so the two copies stay
independent on purpose, the same one way boundary `pgvector_retriever.py`'s own `_embed_query` and
`rag_tools.ingest.embed_texts` already both independently implement the same TEI call.

NO record/replay mode (see `atlas.ports.embedding`'s module docstring for the decision): this
adapter always calls TEI live when it runs; hermetic tests inject an `httpx.MockTransport`d client
instead of a real network client, never a recorded cassette.
"""
from __future__ import annotations

from collections.abc import Sequence

import httpx

DEFAULT_BATCH_SIZE = 16
_TIMEOUT_SECONDS = 120.0  # a full batch the size of an ingest, not the smaller single query timeout


class TeiEmbeddingClient:
    """`EmbeddingClient` over a live TEI server's `/embed` endpoint. `client` is an injectable
    `httpx.Client` (defaults to one owned by this instance, closed by `close()`); hermetic tests
    inject an `httpx.MockTransport`d client instead, matching `pgvector_retriever.py`'s own
    injectable client convention."""

    def __init__(
        self,
        base_url: str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        client: httpx.Client | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.Client(base_url=base_url, timeout=_TIMEOUT_SECONDS)
        self._batch_size = batch_size

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = self._client.post("/embed", json={"inputs": batch})
            response.raise_for_status()
            vectors.extend(response.json())
        return vectors

    def close(self) -> None:
        """Close the owned `httpx.Client`. An injected `client` is the caller's own to close, left
        untouched here (matching `PgvectorRetriever.close()`'s own injected versus owned convention)."""
        if self._owns_client:
            self._client.close()


__all__ = ["TeiEmbeddingClient"]
