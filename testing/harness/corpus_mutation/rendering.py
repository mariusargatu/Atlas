"""The one real I/O boundary this package's re render step needs: write ONLY the affected documents
(`selection.affected_doc_ids`) of a real `corpus_tools.render.render_corpus` call into an
`scope.EphemeralScope`'s corpus directory, plus the minimal manifest `rag_tools.ingest.
load_corpus_docs` reads (`corpus_version`, the affected doc_id -> content hash map, and the corpus
wide `content_hash`, the same field name `corpus_tools.build.build_corpus`'s own manifest carries).

Deliberately NOT `corpus_tools.build.build_corpus`: that function's integrity/verify gates exist to
protect the ONE committed corpus (every contradiction's both sides rendered somewhere, no leaked
hidden entity, across the WHOLE corpus); this lane writes a deliberately partial, ephemeral probe
render, so those corpus wide gates do not apply here. `render_corpus` itself, the actual rendering
library, is still called unmodified; only the packaging (which docs to write, what manifest to
write) is this lane's own, narrower code.

Live/burst only, like the rest of this lane's I/O: shared by `corpus_mutation/__main__.py` (`task
corpus-mutation`) and `testing/tests/test_corpus_mutation_live.py`, never exercised by a hermetic
test, per this task's own contract (hermetic tests cover only `selection`, `scope`, and `tracking`).
"""
from __future__ import annotations

import hashlib
import json

from corpus_tools.render import render_corpus

from corpus_mutation.scope import EphemeralScope
from corpus_mutation.selection import TEMPLATES, FactMutation, affected_doc_ids

__all__ = ["write_affected_docs"]


def write_affected_docs(mutated_reg, mutation: FactMutation, scope: EphemeralScope) -> dict[str, str]:
    """Render `mutated_reg` for real, write only the documents `mutation.fact_ref` touches into
    `scope`'s corpus directory (docs/, provenance/, manifest.json), and return the doc_id ->
    content hash map that ended up in that manifest."""
    docs = render_corpus(mutated_reg, TEMPLATES, seed=1)
    affected = affected_doc_ids(docs, mutation.fact_ref)

    corpus_dir = scope.corpus_root / scope.corpus_version
    docs_dir = corpus_dir / "docs"
    provenance_dir = corpus_dir / "provenance"
    docs_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    doc_hashes: dict[str, str] = {}
    for doc in docs:
        if doc.doc_id not in affected:
            continue
        (docs_dir / f"{doc.doc_id}.txt").write_text(doc.text)
        doc_hashes[doc.doc_id] = hashlib.sha256(doc.text.encode()).hexdigest()
        sidecar = {
            "doc_id": doc.doc_id,
            "doc_type": doc.doc_type,
            "placements": [
                {"fact_ref": p.fact_ref, "value": p.value, "span": list(p.span)} for p in doc.placements
            ],
        }
        (provenance_dir / f"{doc.doc_id}.json").write_text(json.dumps(sidecar, sort_keys=True, indent=2) + "\n")

    content_hash = hashlib.sha256("\n".join(sorted(doc_hashes.values())).encode()).hexdigest()
    manifest = {"corpus_version": scope.corpus_version, "docs": doc_hashes, "content_hash": content_hash}
    (corpus_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return doc_hashes
