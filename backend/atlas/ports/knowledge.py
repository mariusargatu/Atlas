"""The knowledge (retrieval) port. Pure, no client, no framework. The CI adapter is an
in memory keyword retriever (deterministic). A real vector adapter (deferred) would sit behind it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from atlas.domain.retrieval import RetrievalConfig


@dataclass(frozen=True)
class Chunk:
    """One retrieved unit, small-to-big chunk metadata (research note 05) plus a retrieval-time
    `score`. `parent_id` is the small-to-big parent section id (on today's committed corpus this
    equals `doc_id`, since every doc degrades to one chunk). Every field defaults so a caller that
    only cares about `doc_id`/`text` (a toy fixture, a unit test) can construct one without
    supplying the rest."""

    chunk_id: str = ""
    parent_id: str = ""
    doc_id: str = ""
    doc_version: str = ""
    doc_type: str = ""
    heading_path: tuple[str, ...] = ()
    char_span: tuple[int, int] = (0, 0)
    text: str = ""
    entity_ids: tuple[str, ...] = ()
    score: float = 0.0


class Retriever(Protocol):
    def search_chunks(self, query: str, k: int, config: RetrievalConfig) -> list[Chunk]:
        """Return up to ``k`` RELEVANT chunks, best first, never ``k`` padded with irrelevant
        hits. A query with no good match yields fewer (or zero) results. Both adapters honour this
        (in memory by an overlap floor and a vector adapter by a score threshold) so cassettes stay
        symmetric. ``config`` is the one typed knob-set (fused width, rerank toggle, exact-scan
        ground truth, HNSW ef_search) every retrieval call carries; an adapter with no use for a
        given knob (the in memory adapter has no HNSW index, so `ef_search` is a no-op for it)
        still accepts it, so every caller goes through the same boundary (D8)."""
        ...
