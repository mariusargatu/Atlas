"""Shared ordering used by every retrieval stage: sort by score descending, break ties by
chunk name ascending, number from one. A single rule here keeps index, fusion and
reranker outputs comparable instead of each breaking ties its own way."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from atlas.contracts import Chunk, ChunkId, Hit


def name_order(chunks: Sequence[Chunk]) -> tuple[int, ...]:
    """Permutation that puts `chunks` in name order, so indexes built from the same
    chunks in different input order come out identical."""
    return tuple(sorted(range(len(chunks)), key=lambda i: chunks[i].name))


def ranked_hits(
    scored: Iterable[tuple[ChunkId, float]], limit: int | None = None
) -> tuple[Hit, ...]:
    """Orders scored chunks and numbers them from one. `limit=None` keeps every
    candidate, for fusion's union-then-let-the-reranker-cut usage."""
    ordered = sorted(scored, key=lambda pair: (-pair[1], pair[0]))
    if limit is not None:
        ordered = ordered[:limit]
    return tuple(
        Hit(chunk=name, score=float(score), rank=rank)
        for rank, (name, score) in enumerate(ordered, start=1)
    )
