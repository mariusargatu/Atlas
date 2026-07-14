"""SP8 task 7 live lane reproducer: mutate `contract_term-daniel-2025:contract_months` for real
(`corpus_mutation.selection`), re render only the affected document (`corpus_mutation.rendering`)
under a scoped, never committed ephemeral corpus_version (`corpus_mutation.scope`), re index it
against a real TEI embed server and a real Postgres (`rag_tools.ingest`), and prove the ephemeral,
mutated index is genuinely retrievable. Marked `live`, excluded from `task test`; run via `task
test:live` with `docker compose up postgres tei-embed tei-rerank` already healthy (a TEI endpoint is
needed for embedding; the fastlane node was deleted after SP7, so the compose stack, keyless but
slower under Rosetta, is the documented retrieval path here, exactly the precedent
`test_pgvector_adapter_live.py`/`test_metamorphic_live.py` already set). This file is written,
hermetically syntax/import checked, and ready to run; per this task's own contract the live run is
DEFERRED, like Task 6's own live lane and SP7's own live measurements, and has not been executed as
part of this task.

Per this project's own doctrine (mechanics gated, quality measured):

1. **MECHANICS gated:** after the mutate + re render (affected document only) + re index round trip
   completes under an ephemeral, never committed corpus_version, a real `PgvectorRetriever` query
   for the mutation's own `question` retrieves a chunk whose text carries the NEW, mutated value,
   not the stale pre mutation one -- proof the pipeline actually threads the changed fact through to
   a retrievable chunk. Deterministic retrieval mechanics, no model judgment call.
2. **Registry derived answer tracking is NOT exercised here:** it needs a real generation call (a
   provider key), out of scope for this retrieval focused reproducer, the SAME boundary
   `test_metamorphic_live.py` draws for its own registry derived answer equivalence check.
   `corpus_mutation.tracking.answer_tracks_mutated_truth` is already unit tested hermetically,
   against both a truth tracking and a stale-repeating stub answer, in
   `test_corpus_mutation_tracking.py`.
3. **Cleanup is proved too:** the ephemeral corpus_root/index_root directories `scope` allocated
   no longer exist once the `with` block exits, the same guarantee
   `test_corpus_mutation_scope.py` already proves hermetically for the scope in isolation, exercised
   here end to end against a real render and a real index build.
"""
from __future__ import annotations

import json

import psycopg
import pytest
from atlas.adapters.pgvector_retriever import PgvectorRetriever
from atlas.domain.retrieval import RetrievalConfig
from corpus_tools.registry import load_registry
from rag_tools import ingest

from corpus_mutation.rendering import write_affected_docs
from corpus_mutation.scope import EphemeralCorpusVersion
from corpus_mutation.selection import DEFAULT_REGISTRY_PATHS, mutate_registry, select_mutation

POSTGRES_DSN = "postgresql://atlas:atlas-dev-password@localhost:5433/atlas"
TEI_EMBED_URL = "http://localhost:8081"
TEI_RERANK_URL = "http://localhost:8082"

pytestmark = pytest.mark.live


def test_mutated_fact_is_retrievable_from_the_ephemeral_reindexed_corpus() -> None:
    reg = load_registry(list(DEFAULT_REGISTRY_PATHS))
    mutation = select_mutation(reg)
    mutated_reg = mutate_registry(reg, mutation)

    with EphemeralCorpusVersion(mutation) as scope:
        corpus_root, index_root = scope.corpus_root, scope.index_root
        doc_hashes = write_affected_docs(mutated_reg, mutation, scope)
        assert doc_hashes  # at least one document was actually affected by the mutation

        index_dir = ingest.build_index(
            corpus_version=scope.corpus_version,
            tei_embed_url=TEI_EMBED_URL,
            corpus_root=scope.corpus_root,
            index_root=scope.index_root,
        )
        fp = json.loads((index_dir / "fingerprint.json").read_text())
        build_manifest = json.loads((index_dir / "build_manifest.json").read_text())
        with psycopg.connect(POSTGRES_DSN) as conn:
            loaded = ingest.load_parquet(
                conn, index_dir / "chunks.parquet", dim=fp["dim"], build_id=build_manifest["index_build_id"],
            )
        assert loaded == len(doc_hashes)

        retriever = PgvectorRetriever(
            pg_dsn=POSTGRES_DSN, tei_embed_url=TEI_EMBED_URL, tei_rerank_url=TEI_RERANK_URL,
            index_dir=index_dir,
        )
        try:
            fused_config = RetrievalConfig(k_fused=loaded, k_final=loaded, rerank_enabled=False)
            chunks = retriever.search_chunks(mutation.question, loaded, fused_config)
        finally:
            retriever.close()

    print(f"\n{mutation.question!r} retrieved from the ephemeral, mutated index: {[c.doc_id for c in chunks]}")
    new_value_hits = [c for c in chunks if str(mutation.new_value) in c.text]
    stale_value_hits = [c for c in chunks if str(mutation.old_value) in c.text]
    assert new_value_hits, f"no retrieved chunk carried the mutated value {mutation.new_value!r}"
    assert not stale_value_hits, (
        f"a retrieved chunk still carried the pre mutation value {mutation.old_value!r} -- the "
        "re-render/re-index round trip did not fully replace the stale fact"
    )

    # Cleanup, proved end to end: the ephemeral directories are gone once the scope exits, even
    # though a real render and a real index build wrote real files under them above.
    assert not corpus_root.exists()
    assert not index_root.exists()
