"""SP7 Task 7 live lane: the seed set's retrieval half against the real compose stack (real
Postgres, real TEI embed + rerank). Marked `live`, excluded from `task test`; run via `task test:live`
with `docker compose up postgres` already healthy and `ATLAS_TEI_EMBED_URL`/`ATLAS_TEI_RERANK_URL`
sourced from `.env.fastlane` (`set -a; source .env.fastlane; set +a`) so `PgvectorRetriever`'s own
env var read (`backend/atlas/adapters/pgvector_retriever.py`) points at the amd64 fastlane node
instead of the Rosetta emulated compose containers -- `docker-compose.yml`'s own comment documents
this exact override mechanism ("An operator can override either before `docker compose up` ... to
point retrieval [elsewhere]"), the precedent this file follows rather than inventing a new one.

Two things happen here, per this project's own doctrine (mechanics gated, quality measured):

1. **The seed set's retrieval half** (55 cases: `factoid_one_hop` (35) + `factoid_two_hop` (10) +
   `grounded_not_true` (10), every case in `seed_cases.jsonl` carrying a nonempty
   `expected_doc_ids`; `hallucination_bait` is deliberately excluded, nothing to retrieve for an
   unanswerable case by construction) is run through the real `PgvectorRetriever`, at the exact
   config production actually deploys (`knowledge_server.DEPLOYED_K=3`, `RetrievalConfig()`
   defaults), and aggregated by `quality.retrieval_report.evaluate` (Task 7's own hermetic
   aggregation layer, unit tested in isolation by `test_retrieval_report.py`). MECHANICS gated: the
   full 55 case count is used (a silent case drop would fail this), and every case's retrieval
   returns at least one chunk (a completely dead retrieval path would fail this). The recall@3/
   MRR/nDCG@3 NUMBERS themselves are QUALITY, measured and recorded, never gated on a specific bar
   -- see `docs/measurements/sp7-datasets-metrics.md` for the committed figures this run produced
   and the honest CI widths at this dataset's real n=55.

2. **The SP3 flagship baseline reproduction** (`docs/measurements/sp3-rag-spine.md`): the seed
   set's own `seed-flagship-daniel-contract-free` case (`"Is my plan contract free?"`,
   `expected_doc_ids: ["2514487e4633b47b"]`) is run through the same fused/reranked split
   `test_pgvector_adapter_live.py`'s own Daniel test uses. MECHANICS gated: the chunk is present in
   the fused, pre rerank candidate pool. QUALITY, measured only: the reranked position is recorded,
   not asserted on (SP3's own recorded figure: fused rank 5 of 45, reranked rank 14, score 0.00136).

3. **The generation half** (only if a provider key is present in `.env`, mirroring
   `rag_tools.smoke`'s own established D36 tier 2 doctrine exactly): a single tiny, `max_tokens`
   bounded grounded completion over the flagship query's own retrieved passages, proving the
   generation half CAN run against real local retrieval without building a new grading pipeline
   this task's own plan text never asked for.

STATUS AT AUTHORING (SP7 Task 7): the fastlane node (`atlas-fastlane`, `.env.fastlane`) went
unreachable mid run (an unplanned reboot leaving TEI OOM thrashing: the socket accepts but neither
SSH nor HTTP completes a request) and a reboot is pending an operator decision, not run as part of
this task. This file is written, hermetically syntax/import checked, and ready to run; no run of it
has yet completed successfully. `docs/measurements/sp7-datasets-metrics.md` records this honestly
as PENDING LIVE CAPTURE with this file's own invocation as the exact rerun command -- see that
document for the committed hermetic figures (seed set facts, honest CI width sizing) that do not
depend on this file ever having run.
"""
from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest
from atlas.adapters.pgvector_retriever import PgvectorRetriever
from atlas.domain.retrieval import RetrievalConfig
from atlas.mcp_servers.knowledge_server import DEPLOYED_K
from dataset_tools import manifest
from quality.retrieval_report import CaseRetrieval, evaluate
from rag_tools import ingest
from rag_tools.smoke import MAX_TOKENS, QUESTION as _SMOKE_QUESTION, _generation_half, _keyed_provider
from .fixtures import corpus_expectations

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "indexes" / "corpus-0.1.1-bge-m3-03f983e0"
SEED_PATH = REPO_ROOT / "testing" / "harness" / "dataset_tools" / "seed_cases.jsonl"
POSTGRES_DSN = "postgresql://atlas:atlas-dev-password@localhost:5433/atlas"

DANIEL_CHUNK_ID = "2514487e4633b47b"
FLAGSHIP_CASE_ID = "seed-flagship-daniel-contract-free"

# Fixed, committed: a live retrieval quality run is not the seed set's own split assignment seed
# (`dataset_tools.manifest.DEFAULT_SEED`, a different concern), a seed purely for this report's own
# bootstrap resampling, so the recorded CI in the measurements doc reproduces byte for byte on rerun.
METRICS_SEED = 20260720

# The full retrieval relevant slice per the plan's own wording ("the retrieval half of the seed
# set's factoid + adversarial cases"): every case whose `expected_doc_ids` is nonempty, which is
# exactly `factoid_one_hop` + `factoid_two_hop` + `grounded_not_true` on the committed seed set
# (`hallucination_bait` is answerable: false by construction, nothing to retrieve; `other` is the
# action/multi turn/persona overflow bucket, no case level `expected_doc_ids` either). Pinned count
# asserted below so a future edit to the seed set that changes this is a visible, deliberate change.
_RETRIEVAL_SLICES = frozenset({"factoid_one_hop", "factoid_two_hop", "grounded_not_true"})

pytestmark = pytest.mark.live


def _retrieval_relevant_cases() -> tuple[dict, ...]:
    cases = manifest.load_cases_from_jsonl(SEED_PATH)
    return tuple(c for c in cases if manifest.case_slice(c) in _RETRIEVAL_SLICES)


@pytest.fixture(scope="session")
def ensure_chunks_loaded() -> int:
    """Identical discipline to `test_pgvector_adapter_live.py`'s own fixture: idempotent load of the
    committed index build (schema `IF NOT EXISTS`, rows `ON CONFLICT DO NOTHING`), never a rebuild."""
    fp = json.loads((INDEX_DIR / "fingerprint.json").read_text())
    build_manifest = json.loads((INDEX_DIR / "build_manifest.json").read_text())
    with psycopg.connect(POSTGRES_DSN) as conn:
        row_count = ingest.load_parquet(
            conn, INDEX_DIR / "chunks.parquet", dim=fp["dim"], build_id=build_manifest["index_build_id"]
        )
    assert row_count == corpus_expectations.CHUNK_COUNT
    return row_count


@pytest.fixture
def retriever(ensure_chunks_loaded: int) -> PgvectorRetriever:
    # No explicit tei_embed_url/tei_rerank_url: unlike test_pgvector_adapter_live.py (which pins the
    # compose container ports directly), this test lets PgvectorRetriever's own constructor read
    # ATLAS_TEI_EMBED_URL/ATLAS_TEI_RERANK_URL from the environment, so sourcing .env.fastlane before
    # the test run points every call at the fast amd64 node instead of Rosetta emulated compose TEI.
    return PgvectorRetriever(pg_dsn=POSTGRES_DSN, index_dir=INDEX_DIR)


def test_seed_set_retrieval_half_recall_mrr_ndcg(retriever: PgvectorRetriever) -> None:
    cases = _retrieval_relevant_cases()
    assert len(cases) == 55  # MECHANICS: the full retrieval relevant slice, no silent drop

    deploy_config = RetrievalConfig()  # k_fused=50, k_final=5, rerank_enabled=True: the real default
    results = []
    empty_retrievals = []
    for case in cases:
        query = case["turns"][0]["user"]
        chunks = retriever.search_chunks(query, DEPLOYED_K, deploy_config)
        retrieved_ids = tuple(c.chunk_id for c in chunks)
        if not retrieved_ids:
            empty_retrievals.append(case["case_id"])
        results.append(CaseRetrieval(case["case_id"], retrieved_ids, frozenset(case["expected_doc_ids"])))

    # MECHANICS, gated: real retrieval ran for every case and returned something (a dead retrieval
    # path -- a bad TEI call, a broken SQL arm -- would show up as empty results, not as low recall).
    assert not empty_retrievals, f"retrieval returned zero chunks for: {empty_retrievals}"

    report = evaluate(results, k=DEPLOYED_K, seed=METRICS_SEED)
    assert report.n == 55

    recall_point, recall_lo, recall_hi = report.recall_at_k_ci
    mrr_point, mrr_lo, mrr_hi = report.mrr_ci
    ndcg_point, ndcg_lo, ndcg_hi = report.ndcg_at_k_ci

    print(f"\nSP7 seed set retrieval half, n={report.n}, k={report.k} (DEPLOYED_K, RetrievalConfig() defaults):")
    print(f"  hit_rate@{report.k}: {report.hit_rate_at_k:.4f}  wilson 95% CI: ({report.hit_rate_at_k_ci[0]:.4f}, {report.hit_rate_at_k_ci[1]:.4f})")
    print(f"  recall@{report.k}:   {recall_point:.4f}  bootstrap 95% CI: ({recall_lo:.4f}, {recall_hi:.4f})")
    print(f"  MRR:          {mrr_point:.4f}  bootstrap 95% CI: ({mrr_lo:.4f}, {mrr_hi:.4f})")
    print(f"  nDCG@{report.k}:     {ndcg_point:.4f}  bootstrap 95% CI: ({ndcg_lo:.4f}, {ndcg_hi:.4f})")
    print(f"  detectable nDCG effect at n={report.n}: {report.detectable_effect_ndcg}")
    print("  QUALITY, measured only -- see docs/measurements/sp7-datasets-metrics.md for the committed figures.")


def test_flagship_baseline_fused_membership_gated_reranked_position_measured(retriever: PgvectorRetriever) -> None:
    case = next(c for c in _retrieval_relevant_cases() if c["case_id"] == FLAGSHIP_CASE_ID)
    query = case["turns"][0]["user"]
    assert case["expected_doc_ids"] == [DANIEL_CHUNK_ID]

    # (a) MECHANICS, gated: the Daniel chunk is a candidate in the fused, pre rerank pool at all.
    fused_config = RetrievalConfig(k_fused=50, k_final=50, rerank_enabled=False)
    fused_results = retriever.search_chunks(query, 50, fused_config)
    fused_ids = [c.chunk_id for c in fused_results]
    assert DANIEL_CHUNK_ID in fused_ids
    fused_rank = fused_ids.index(DANIEL_CHUNK_ID) + 1

    # (b) QUALITY, measured only: the reranked position, printed, never asserted on (SP3's own
    # recorded baseline: fused rank 5 of 45, reranked rank 14, score 0.00136).
    reranked_config = RetrievalConfig(k_fused=50, k_final=50, rerank_enabled=True)
    reranked_results = retriever.search_chunks(query, 50, reranked_config)
    reranked_ids = [c.chunk_id for c in reranked_results]
    reranked_rank = reranked_ids.index(DANIEL_CHUNK_ID) + 1 if DANIEL_CHUNK_ID in reranked_ids else None
    reranked_score = reranked_results[reranked_rank - 1].score if reranked_rank is not None else None

    print(f"\nSP7 flagship baseline reproduction, query {query!r}:")
    print(f"  fused (pre rerank) rank: {fused_rank} of {len(fused_ids)}")
    print(f"  reranked rank/score:     rank={reranked_rank} score={reranked_score}")
    print("  see docs/measurements/sp3-rag-spine.md (original) and sp7-datasets-metrics.md (reproduction).")


def test_generation_half_grounded_in_real_local_retrieval(retriever: PgvectorRetriever) -> None:
    """Mirrors `rag_tools.smoke`'s own D36 tier 2 generation half doctrine exactly (reused, not
    reimplemented): if a recognized provider key is present in `.env`, real retrieval feeds a
    single tiny (`MAX_TOKENS` bounded) grounded completion; if not, this SKIPS cleanly rather than
    failing, per the plan's own "generation half only if a provider key is present... if absent,
    say so and skip, do not fail" instruction. This is deliberately the SAME narrow check
    `rag_tools.smoke._generation_half` already performs (real local retrieval, one small live
    provider call), not a new generation grading pipeline: Task 7's own plan text scopes the live
    lane to retrieval, and building a full agentic generation benchmark here would read SP9's
    matrix and SP8's judge boundary, not SP7's."""
    keyed = _keyed_provider()
    if keyed is None:
        pytest.skip("no recognized provider key present in .env (ANTHROPIC_API_KEY/OPENAI_API_KEY); "
                    "generation half skipped, per doctrine, not failed")
    provider, key_var = keyed

    chunks = retriever.search_chunks(_SMOKE_QUESTION, DEPLOYED_K, RetrievalConfig())
    passages = [{"doc_id": c.doc_id, "text": c.text} for c in chunks]
    print(f"\nSP7 generation half ({key_var} present, provider={provider!r}, max_tokens={MAX_TOKENS}):")
    _generation_half(passages, provider)
