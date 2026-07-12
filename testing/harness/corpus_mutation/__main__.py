"""`task corpus-mutation`: SP8 task 7's live/burst operator entrypoint. Needs `docker compose up
postgres tei-embed tei-rerank` (the fastlane node was deleted after SP7, so the compose stack is the
documented retrieval/embedding path here, exactly `testing/tests/test_metamorphic_live.py`'s own
precedent for Task 6) and a provider key in `.env` for the real agent turn at the end. Per this
task's own contract, the live end to end run is DEFERRED, like SP7's own live measurements: this
file is written and ready to run, but has not been executed as part of this task.

The five steps, in order (see the package docstring for which library backs each one):

1. Load the committed registry, `selection.select_mutation` picks the one fact to mutate,
   `selection.mutate_registry` returns a throwaway mutated copy.
2. `corpus_tools.render.render_corpus` re renders the mutated registry for real;
   `selection.affected_doc_ids` narrows to the documents the mutated fact actually touches, and only
   those are written into the ephemeral corpus_version this step allocates.
3. `rag_tools.ingest.build_index` (a real TEI embed call) plus `rag_tools.ingest.load_parquet` (a
   real Postgres load) re index exactly those documents, under `scope.EphemeralCorpusVersion`'s own
   ephemeral, never committed corpus_version.
4. A real `PgvectorRetriever` against that ephemeral index, driving the REAL agent graph
   (`atlas.orchestration.atlas_graph.build_atlas_graph`) for one live turn asking the mutation's own
   `question`.
5. `tracking.answer_tracks_mutated_truth` grades the agent's real answer against the mutated truth,
   and this script prints the verdict.

Every byte this script writes (the ephemeral corpus render, the ephemeral index) is removed when
`scope.EphemeralCorpusVersion`'s `with` block exits, success or failure; nothing here is committed.
"""
from __future__ import annotations

import asyncio
import json

import psycopg
from langchain_core.messages import HumanMessage

from atlas.adapters.pgvector_retriever import PgvectorRetriever
from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph
from corpus_tools.registry import load_registry
from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from rag_tools import ingest
from replay.gateway import GatewayChatModel
from replay.providers import build_chat_model, provider_tag

from corpus_mutation.rendering import write_affected_docs
from corpus_mutation.scope import EphemeralCorpusVersion
from corpus_mutation.selection import DEFAULT_REGISTRY_PATHS, mutate_registry, select_mutation
from corpus_mutation.tracking import answer_tracks_mutated_truth

POSTGRES_DSN = "postgresql://atlas:atlas-dev-password@localhost:5433/atlas"
TEI_EMBED_URL = "http://localhost:8081"
TEI_RERANK_URL = "http://localhost:8082"
# cust_legacy_term is Daniel: the same customer conflict-daniel-contract's winning fact
# (contract_term-daniel-2025) concerns, so the real agent turn below is asked as the one customer
# this mutated fact actually belongs to.
CUSTOMER_ID = "cust_legacy_term"
THREAD_ID = "corpus-mutation-probe"


async def _run_agent_turn(retriever: PgvectorRetriever, question: str) -> str:
    """Step 4: the REAL agent graph (`atlas.orchestration.atlas_graph.build_atlas_graph`), a live
    model through the gateway (mode="live", the same `GatewayMode` `backend/atlas/server.py` builds
    for a non replay `ATLAS_MODE`), and the ephemeral, mutated-corpus-backed retriever injected the
    same way every hermetic test in this repo injects a fake one -- this is the one call in the
    whole lane that is genuinely live end to end (retrieval AND generation)."""
    gateway = GatewayChatModel(
        model_id=provider_tag(), cassette_dir=None, mode="live", inner=build_chat_model(),
    )
    graph = build_atlas_graph(
        gateway, IdFactory("corpus-mutation"), ActionsBackend(IdFactory("corpus-mutation-ref")),
        new_checkpointer(), retriever=retriever,
    )
    out = await graph.ainvoke(
        {"messages": [HumanMessage(question)], "session": {"customer_id": CUSTOMER_ID}},
        {"configurable": {"thread_id": THREAD_ID}},
    )
    return str(out["final_response"])


def main() -> None:
    print(
        "SP8 task 7: corpus mutation lane (live/burst; needs `docker compose up postgres tei-embed "
        "tei-rerank` and a provider key in .env for the real agent turn)"
    )

    reg = load_registry(list(DEFAULT_REGISTRY_PATHS))
    mutation = select_mutation(reg)
    mutated_reg = mutate_registry(reg, mutation)
    print(f"mutating {mutation.fact_ref}: {mutation.old_value} -> {mutation.new_value}")

    with EphemeralCorpusVersion(mutation) as scope:
        print(f"ephemeral corpus_version={scope.corpus_version} (never committed, cleaned up on exit)")
        doc_hashes = write_affected_docs(mutated_reg, mutation, scope)
        print(f"re-rendered {len(doc_hashes)} affected document(s): {sorted(doc_hashes)}")

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
                conn, index_dir / "chunks.parquet", dim=fp["dim"], build_id=build_manifest["index_build_id"]
            )
        print(f"re-indexed {loaded} affected chunk(s) under index_build_id={build_manifest['index_build_id']}")

        retriever = PgvectorRetriever(
            pg_dsn=POSTGRES_DSN, tei_embed_url=TEI_EMBED_URL, tei_rerank_url=TEI_RERANK_URL,
            index_dir=index_dir,
        )
        try:
            answer = asyncio.run(_run_agent_turn(retriever, mutation.question))
        finally:
            retriever.close()

    result = answer_tracks_mutated_truth(mutation, answer)
    print(f"\nquestion: {mutation.question!r}")
    print(f"agent answer: {answer!r}")
    print(
        f"tracks_new_truth={result.tracks_new_truth}  repeats_stale_truth={result.repeats_stale_truth}  "
        f"holds={result.holds}"
    )
    if not result.holds:
        print(
            "FAILED: the agent's answer does not track the mutated registry truth -- it may be "
            "answering from parametric/training knowledge or a stale cache/index instead of the "
            "freshly re-indexed retrieved context."
        )


if __name__ == "__main__":
    main()
