"""`Chunk` (de)serialization, shared by every stage that persists retrieved chunks through the content
hash cache or a per query result file: one round trip definition, never re derived per stage (the
same "one shared implementation, not re derived per call site" discipline `atlas.domain.retrieval`'s
`rrf_fuse` and `quality.ir_metrics` already hold elsewhere in this repo).
"""
from __future__ import annotations

from atlas.ports.knowledge import Chunk


def serialize_chunk(chunk: Chunk) -> dict:
    """`Chunk` -> a plain JSON dict (tuples become lists, everything else already a JSON scalar)."""
    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "doc_id": chunk.doc_id,
        "doc_version": chunk.doc_version,
        "doc_type": chunk.doc_type,
        "heading_path": list(chunk.heading_path),
        "char_span": list(chunk.char_span),
        "text": chunk.text,
        "entity_ids": list(chunk.entity_ids),
        "score": chunk.score,
    }


def deserialize_chunk(data: dict) -> Chunk:
    """The inverse of `serialize_chunk`: lists become tuples again, matching `Chunk`'s own frozen
    dataclass field types exactly."""
    return Chunk(
        chunk_id=data["chunk_id"],
        parent_id=data["parent_id"],
        doc_id=data["doc_id"],
        doc_version=data["doc_version"],
        doc_type=data["doc_type"],
        heading_path=tuple(data["heading_path"]),
        char_span=tuple(data["char_span"]),
        text=data["text"],
        entity_ids=tuple(data["entity_ids"]),
        score=data["score"],
    )


__all__ = ["deserialize_chunk", "serialize_chunk"]
