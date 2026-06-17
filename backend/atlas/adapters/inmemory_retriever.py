"""A deterministic keyword retriever over the seed corpus, the hermetic CI adapter behind the
knowledge port. It is the only retriever today. A real vector adapter (dev/prod) is deferred.
Keyword overlap means no embedding hardware, so no cross machine float drift in the cassette chain.
"""
from __future__ import annotations

from atlas.domain.corpus import CORPUS
from atlas.domain.retrieval import RetrievalConfig
from atlas.ports.knowledge import Chunk


class InMemoryRetriever:
    def __init__(self, corpus: list[Chunk] | None = None) -> None:
        self._corpus = corpus if corpus is not None else CORPUS

    @staticmethod
    def _overlap(query_words: set[str], text: str) -> int:
        return len(query_words & set(text.lower().split()))

    def search_chunks(self, query: str, k: int = 3, *, config: RetrievalConfig) -> list[Chunk]:
        """Keyword-overlap ranking, unchanged in behaviour from the pre-D8 `search` method: `config`
        is REQUIRED (SP3 final review: tightened to match the `Retriever` Protocol's own signature,
        `search_chunks(query, k, config)`, which never made `config` optional -- every real adapter
        call site already threads a `RetrievalConfig` through; only this adapter's own signature
        used to let callers skip it, a conformance gap the Protocol never actually allowed). It is
        accepted for port conformance (hybrid fusion / rerank / exact-scan / HNSW ef_search are all
        no-ops on this toy adapter, which has no vector index to tune), not yet wired into the
        ranking itself. `k` still caps how many RELEVANT chunks come back, best first."""
        words = set(query.lower().split())
        ranked = sorted(self._corpus, key=lambda c: self._overlap(words, c.text), reverse=True)
        return [c for c in ranked if self._overlap(words, c.text) > 0][:k]
