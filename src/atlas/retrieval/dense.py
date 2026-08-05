from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from atlas.contracts import Chunk, ChunkId, SearchResult
from atlas.retrieval.ranking import name_order, ranked_hits


@dataclass(frozen=True, slots=True)
class VectorIndex:
    chunks: tuple[ChunkId, ...]
    matrix: np.ndarray

    @classmethod
    def build(cls, chunks: Sequence[Chunk], matrix: np.ndarray) -> VectorIndex:
        order = name_order(chunks)
        return cls(
            chunks=tuple(chunks[i].name for i in order),
            matrix=matrix[list(order)],
        )

    def search(self, query_text: str, query_vector: np.ndarray, candidates: int) -> SearchResult:
        # Zero norm would divide by zero; treat it as one so the row scores zero instead.
        norms = np.linalg.norm(self.matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        denom = norms * (query_norm if query_norm > 0 else 1.0)
        scores = (self.matrix @ query_vector) / np.where(denom == 0, 1.0, denom)
        return SearchResult(
            query=query_text, arm="vector",
            hits=ranked_hits(zip(self.chunks, scores, strict=True), candidates),
        )
