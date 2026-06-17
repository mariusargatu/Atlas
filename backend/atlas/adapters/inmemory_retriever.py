"""A deterministic keyword retriever over the seed corpus, the hermetic CI adapter behind the
knowledge port. It is the only retriever today. A real vector adapter (dev/prod) is deferred.
Keyword overlap means no embedding hardware, so no cross machine float drift in the cassette chain.
"""
from __future__ import annotations

from atlas.domain.corpus import CORPUS
from atlas.ports.knowledge import Chunk


class InMemoryRetriever:
    def __init__(self, corpus: list[Chunk] | None = None) -> None:
        self._corpus = corpus if corpus is not None else CORPUS

    @staticmethod
    def _overlap(query_words: set[str], text: str) -> int:
        return len(query_words & set(text.lower().split()))

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        words = set(query.lower().split())
        ranked = sorted(self._corpus, key=lambda c: self._overlap(words, c.text), reverse=True)
        return [c for c in ranked if self._overlap(words, c.text) > 0][:k]
