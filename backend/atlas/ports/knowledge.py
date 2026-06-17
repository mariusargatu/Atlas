"""The knowledge (retrieval) port. Pure, no client, no framework. The CI adapter is an
in memory keyword retriever (deterministic); a real vector adapter (deferred) would sit behind it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    text: str
    facts: dict = field(default_factory=dict)  # structured facts the doc asserts


class Retriever(Protocol):
    def search(self, query: str, k: int = 3) -> list[Chunk]:
        """Return up to ``k`` RELEVANT chunks, best first, never ``k`` padded with irrelevant
        hits. A query with no good match yields fewer (or zero) results. Both adapters honour this
        (in memory by an overlap floor; a vector adapter by a score threshold) so cassettes stay symmetric."""
        ...
