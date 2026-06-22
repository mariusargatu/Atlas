"""The registry to corpus join: given a registry fact_ref (`entity_id:field`), which retrieval
unit(s) ground it. Reads only committed, filesystem local artifacts (`manifest.json`, provenance
sidecars, rendered doc text): no network, no model call, no wall clock, fully hermetic.

`chunk_ids_for_fact` mirrors `rag_tools.chunker`'s own span overlap rule exactly (a placement
grounds a chunk when the placement's span and the chunk's `char_span` intersect at all) rather than
inventing a third variant of that join: `ChunkRecord.entity_ids` is already computed by
`chunker._entity_ids_for_span` with that identical inequality, over the very same placements list
this module reads. `doc_version` (the manifest's per doc content hash) is passed straight into
`chunker.chunk_document` and is folded into the returned `chunk_id`, so it never needs a field of
its own here or in the dataset contract.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag_tools import chunker

DEFAULT_CORPUS_DIR = Path("corpus/rendered/corpus-0.1.1")


@dataclass(frozen=True)
class CorpusIndex:
    """Everything the generator needs from one rendered corpus_version, loaded once. Every mapping
    here is built by iterating `sorted(doc_versions)` (file order, not dict/insertion order), so a
    generator built on top of this index inherits the same determinism guarantee."""

    corpus_dir: Path
    corpus_version: str
    doc_versions: dict[str, str]                    # doc_id -> content hash (manifest's docs map)
    doc_types: dict[str, str]                        # doc_id -> doc_type
    placements_by_doc: dict[str, tuple[dict, ...]]    # doc_id -> its provenance sidecar's placements
    docs_by_fact: dict[str, tuple[str, ...]]          # fact_ref -> sorted doc_ids that place it


def load_corpus_index(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> CorpusIndex:
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    doc_versions: dict[str, str] = dict(manifest["docs"])
    doc_types: dict[str, str] = {}
    placements_by_doc: dict[str, tuple[dict, ...]] = {}
    docs_by_fact: dict[str, list[str]] = {}

    for doc_id in sorted(doc_versions):
        sidecar = json.loads((corpus_dir / "provenance" / f"{doc_id}.json").read_text())
        doc_types[doc_id] = sidecar["doc_type"]
        placements = tuple(sidecar["placements"])
        placements_by_doc[doc_id] = placements
        for placement in placements:
            docs_by_fact.setdefault(placement["fact_ref"], []).append(doc_id)

    return CorpusIndex(
        corpus_dir=corpus_dir,
        corpus_version=manifest["corpus_version"],
        doc_versions=doc_versions,
        doc_types=doc_types,
        placements_by_doc=placements_by_doc,
        docs_by_fact={fact_ref: tuple(sorted(docs)) for fact_ref, docs in docs_by_fact.items()},
    )


def docs_for_fact(index: CorpusIndex, fact_ref: str) -> tuple[str, ...]:
    """Sorted doc_ids whose provenance sidecar places this exact fact_ref. Empty for a fact that is
    never rendered anywhere (the bait entities) or for a registry field the renderer never surfaces."""
    return index.docs_by_fact.get(fact_ref, ())


def doc_type_for_fact(index: CorpusIndex, fact_ref: str) -> str | None:
    """The doc_type of the first (sorted) doc that places this fact, or None if unplaced."""
    doc_ids = docs_for_fact(index, fact_ref)
    return index.doc_types[doc_ids[0]] if doc_ids else None


def _span_overlaps(chunk_span: tuple[int, int], placement_span: tuple[int, int]) -> bool:
    """The exact overlap condition `rag_tools.chunker._entity_ids_for_span` applies internally."""
    chunk_start, chunk_end = chunk_span
    placement_start, placement_end = placement_span
    return placement_start < chunk_end and chunk_start < placement_end


def chunk_ids_for_fact(index: CorpusIndex, fact_ref: str) -> tuple[str, ...]:
    """Every `ChunkRecord.chunk_id` (sorted, deduplicated) whose `char_span` grounds this exact
    fact_ref, across every doc that places it. On corpus-0.1.1 (single chunk per doc, per
    `rag_tools.chunker`'s own docstring) this is one chunk id per matching doc; the span overlap
    check still runs for real so a future doc with more than one chunk is handled correctly, not
    merely assumed."""
    chunk_ids: set[str] = set()
    for doc_id in docs_for_fact(index, fact_ref):
        placements = index.placements_by_doc[doc_id]
        target_span = tuple(next(p["span"] for p in placements if p["fact_ref"] == fact_ref))
        text = (index.corpus_dir / "docs" / f"{doc_id}.txt").read_text()
        chunks = chunker.chunk_document(
            doc_id=doc_id,
            doc_type=index.doc_types[doc_id],
            text=text,
            doc_version=index.doc_versions[doc_id],
            corpus_version=index.corpus_version,
            placements=placements,
        )
        for chunk in chunks:
            if _span_overlaps(chunk.char_span, target_span):
                chunk_ids.add(chunk.chunk_id)
    return tuple(sorted(chunk_ids))
