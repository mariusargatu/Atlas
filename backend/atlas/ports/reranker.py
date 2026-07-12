"""The reranker port. Pure, no client, no framework.

A reranker takes the first pass (the fast approximate set the vector search returned) and reorders
it, pushing the genuinely best chunks above the merely similar ones. In the intended pipeline it sits
between the vector query and the model; it is NOT yet wired into the runtime graph (which today
retrieves and answers with no rerank step), so this port and its cassette adapter are defined and
tested ahead of that wiring, the same staged approach `domain.spotlight` documents. The CI adapter is
a cassette of REPLAYED scores (deterministic); a real cross-encoder (bge/mxbai/ms-marco-MiniLM) sits
behind this same port in dev/prod, deferred so no torch and no cross-machine float drift enter the
hermetic lane.
"""
from __future__ import annotations

from typing import Protocol

from atlas.ports.knowledge import Chunk


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Return the chunks reordered best-first for ``query``. A stable ordering: equal scores keep
        the caller's (deterministic) input order, so the cassette chain stays byte-stable."""
        ...
