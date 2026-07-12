"""SP8 task 6 live lane: the same three D32 invariants `test_metamorphic.py` checks hermetically
against the stub retriever, reproduced against the real compose stack (real Postgres, real TEI
embed + rerank) over the pinned `corpus-0.1.1-bge-m3-03f983e0` index. Marked `live`, excluded from
`task test`; run via `task test:live` with `docker compose up postgres tei-embed tei-rerank`
already healthy (a TEI endpoint is needed for embedding; the fastlane node was deleted after SP7,
so the compose stack, keyless but slower under Rosetta, is the documented retrieval path here,
exactly the precedent `test_pgvector_adapter_live.py`/`test_sp7_retrieval_metrics_live.py` already
set). This file is written, hermetically syntax/import checked, and ready to run; per this task's
own contract the live run is deferred, like SP7's own live measurements, and has not been executed
as part of this task.

Per this project's own doctrine (mechanics gated, quality measured -- the SAME doctrine
`test_pgvector_adapter_live.py`'s Daniel test already applies to this exact conflict query):

1. **ID based retrieval agreement, MECHANICS gated:** the fused, pre rerank candidate pool (rerank
   disabled, `k_final == k_fused` so the whole ranked pool is visible) contains the winning chunk
   (`DANIEL_CHUNK_ID`, the SAME hash `families.WINNING_DOC_ID`/`dataset_tools/seed_cases.jsonl`'s
   `gen-fact-contract_term-daniel-2025-contract_months` case already commit) for EVERY member of
   `metamorphic.families.PARAPHRASE_FAMILY`, regardless of wording. Deterministic retrieval
   mechanics (embed, two SQL arms, RRF fusion), no model judgment call.
2. **Rank overlap, QUALITY measured only:** at the real DEPLOYED_K width with reranking ON (the
   actual production config), the reranked top-k id sets across the paraphrase family are recorded
   and printed, not asserted on. SP3's own recorded finding for the single query "is my plan
   contract free" already established that a generic cross-encoder demotes this customer-specific
   override below generically worded "No contract" marketing pages (fused rank 5 of 45, reranked
   rank 14); a full paraphrase family may plausibly reproduce or vary that finding, which is
   measurement material for SP8/SP9's quality plane, not a defect in this test or the adapter.
3. **Registry derived answer equivalence** is NOT exercised here: it needs a real generation call
   (a provider key, `rag_tools.smoke`'s own tier 2 doctrine), which is out of scope for this
   retrieval focused live reproducer; `metamorphic.report.registry_answer_equivalence_holds` is
   already unit tested hermetically against both the frozen `WINNING_ANSWER` and the deliberately
   drifted `DRIFTED_ANSWER` fixture in `test_metamorphic.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest
from atlas.adapters.pgvector_retriever import PgvectorRetriever
from atlas.domain.retrieval import RetrievalConfig
from atlas.mcp_servers.knowledge_server import DEPLOYED_K
from rag_tools import ingest

from metamorphic.families import PARAPHRASE_FAMILY
from .fixtures import corpus_expectations

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "indexes" / "corpus-0.1.1-bge-m3-03f983e0"
POSTGRES_DSN = "postgresql://atlas:atlas-dev-password@localhost:5433/atlas"
TEI_EMBED_URL = "http://localhost:8081"
TEI_RERANK_URL = "http://localhost:8082"

# The SAME chunk hash `metamorphic.families.WINNING_DOC_ID` names, reproduced literally here so
# this file's own intent (is the Daniel chunk in the fused pool for every paraphrase) reads
# standalone; compared against `chunk.chunk_id` (the pgvector adapter's content addressed hash),
# never `chunk.doc_id` (that field carries the human readable "doc-..." id on the real corpus,
# a genuinely different string -- see `test_sp7_retrieval_metrics_live.py`'s own
# `DANIEL_CHUNK_ID`/`c.chunk_id` convention, which this file follows rather than inventing a
# second one).
DANIEL_CHUNK_ID = "2514487e4633b47b"

pytestmark = pytest.mark.live


@pytest.fixture(scope="session")
def ensure_chunks_loaded() -> int:
    """Identical discipline to `test_pgvector_adapter_live.py`'s own fixture: idempotent load of
    the committed index build, never a rebuild."""
    fp = json.loads((INDEX_DIR / "fingerprint.json").read_text())
    manifest = json.loads((INDEX_DIR / "build_manifest.json").read_text())
    with psycopg.connect(POSTGRES_DSN) as conn:
        row_count = ingest.load_parquet(
            conn, INDEX_DIR / "chunks.parquet", dim=fp["dim"], build_id=manifest["index_build_id"]
        )
    assert row_count == corpus_expectations.CHUNK_COUNT
    return row_count


@pytest.fixture
def retriever(ensure_chunks_loaded: int) -> PgvectorRetriever:
    return PgvectorRetriever(
        pg_dsn=POSTGRES_DSN,
        tei_embed_url=TEI_EMBED_URL,
        tei_rerank_url=TEI_RERANK_URL,
        index_dir=INDEX_DIR,
    )


def test_id_based_retrieval_agreement_holds_in_the_fused_pool_for_every_paraphrase(
    retriever: PgvectorRetriever,
) -> None:
    """MECHANICS gated: every member of the frozen paraphrase family surfaces the Daniel chunk
    somewhere in the fused, pre rerank candidate pool. `k_fused=50 == k_final=50` on a 45 row
    corpus means the whole ranked pool is visible, so membership here cannot fail unless fusion
    itself is broken (the same corpus size honesty note `test_pgvector_adapter_live.py` records)."""
    fused_config = RetrievalConfig(k_fused=50, k_final=50, rerank_enabled=False)
    misses = []
    for member in PARAPHRASE_FAMILY.members:
        chunks = retriever.search_chunks(member.question, 50, fused_config)
        chunk_ids = [c.chunk_id for c in chunks]
        print(f"\n{member.question!r} fused pool contains Daniel: {DANIEL_CHUNK_ID in chunk_ids}")
        if DANIEL_CHUNK_ID not in chunk_ids:
            misses.append(member.question)
    assert not misses, f"fused pool missed the Daniel chunk for: {misses}"


def test_rank_overlap_at_deployed_k_is_measured_not_gated(retriever: PgvectorRetriever) -> None:
    """QUALITY, measured only: the real production config (`DEPLOYED_K`, `RetrievalConfig()`
    defaults, reranking ON) across the whole paraphrase family. Printed for SP8/SP9's quality
    plane; never asserted on, per this project's own doctrine and SP3's own prior finding on this
    exact conflict query (a generic reranker may demote the customer specific override)."""
    from itertools import combinations

    from quality.ir_metrics import rank_overlap_at_k

    deploy_config = RetrievalConfig()
    retrieved_by_member = []
    for member in PARAPHRASE_FAMILY.members:
        chunks = retriever.search_chunks(member.question, DEPLOYED_K, deploy_config)
        chunk_ids = tuple(c.chunk_id for c in chunks)
        retrieved_by_member.append(chunk_ids)
        rank = chunk_ids.index(DANIEL_CHUNK_ID) + 1 if DANIEL_CHUNK_ID in chunk_ids else None
        print(f"\n{member.question!r} reranked top {DEPLOYED_K}: {chunk_ids} (Daniel rank: {rank})")

    overlaps = [
        rank_overlap_at_k(a, b, DEPLOYED_K) for a, b in combinations(retrieved_by_member, 2)
    ]
    print(f"\nmin rank overlap across the paraphrase family at DEPLOYED_K={DEPLOYED_K}: {min(overlaps):.2f}")
    print("recorded, not gated: this is quality measurement material, per this project's doctrine.")
