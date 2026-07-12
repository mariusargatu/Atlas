"""A deterministic reranker over REPLAYED scores, the hermetic CI adapter behind the reranker port.

Scores are a recorded ``{query: {doc_id: score}}`` table (a cross-encoder run captured once), so the
reordering is byte-stable with no model in the lane, the same record/replay discipline the gateway
uses for the LLM. A real cross-encoder adapter (dev/prod) would compute the scores live behind the
same port. Reordering is a stable sort by score descending: equal or unrecorded scores keep the
caller's (already deterministic) input order, so the output never depends on dict iteration order.
"""
from __future__ import annotations

from collections.abc import Mapping

from atlas.ports.knowledge import Chunk

_MISSING = float("-inf")  # a doc with no recorded score sinks to the back, deterministically


class CassetteReranker:
    def __init__(self, scores: Mapping[str, Mapping[str, float]]) -> None:
        self._scores = scores

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        table = self._scores.get(query, {})
        # Stable sort on score descending: equal (or unrecorded) scores keep the input order, so an
        # unseen query leaves the list untouched and unscored docs sink to the back in place.
        return sorted(chunks, key=lambda c: -table.get(c.doc_id, _MISSING))
