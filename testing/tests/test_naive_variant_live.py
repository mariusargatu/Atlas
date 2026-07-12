"""The naive variant on real retrieval, live (SP3 task 7, D36 tier 2 compose acceptance): with
`docker compose up` healthy, the chat pipeline's retrieval step answers the Daniel question from
the REAL corpus-0.1.1, not the hermetic toy corpus. Marked `live`, excluded from `task test`; run
via `task test:live` with `docker compose up postgres tei-embed tei-rerank` already healthy.

Seam used: `ChatOut` (`atlas/chat_app.py`) never exposes what was retrieved -- only `final_response`
-- and a full live agent turn through `/chat` cannot be replayed deterministically either (round 1's
cassette key only covers the human question, but round 2's covers the WHOLE history including the
ToolMessage carrying the just-retrieved real text, which cannot be pre-seeded without already
knowing what pgvector will return). So this test hits the identical MCP construction the graph's own
`tools_read` node calls for every knowledge turn -- `atlas.orchestration.atlas_graph._knowledge_call`
builds `build_knowledge_server(retriever)` and calls its `search_knowledge` tool -- with a REAL
`PgvectorRetriever` pointed at the compose stack. This is "the chat pipeline's retrieved context" via
the one seam that is both real and assertable: the exact MCP tool the running graph calls, not a
hand rolled alternative.
"""
from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest
from atlas.adapters.pgvector_retriever import PgvectorRetriever
from atlas.mcp_servers.knowledge_server import DEPLOYED_K, build_knowledge_server
from mcp.shared.memory import create_connected_server_and_client_session
from rag_tools import ingest
from .fixtures import corpus_expectations

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "indexes" / "corpus-0.1.1-bge-m3-03f983e0"
POSTGRES_DSN = "postgresql://atlas:atlas-dev-password@localhost:5433/atlas"
TEI_EMBED_URL = "http://localhost:8081"
TEI_RERANK_URL = "http://localhost:8082"

QUESTION = "is my plan contract free"  # Daniel's question, the corpus's planted grounding conflict

# The hermetic toy corpus's doc ids (atlas.domain.corpus.CORPUS): none carry the "doc-" prefix real
# corpus-0.1.1 doc ids always do (e.g. "doc-contract_terms-contract_term-daniel-2025"), so this is a
# clean, structural "not the toy corpus" check, not a fragile content match.
TOY_CORPUS_DOC_IDS = {"plan-current-page", "troubleshoot-router", "poisoned-doc"}

pytestmark = pytest.mark.live


@pytest.fixture(scope="session")
def ensure_chunks_loaded() -> int:
    """Idempotent (`IF NOT EXISTS` schema, `ON CONFLICT DO NOTHING` inserts): reuses the same
    committed index build the compose `rag-init` service loads, so this test proves the SAME data a
    real `docker compose up` would have loaded, not a test-only fixture. `build_id` (SP3 final
    review, table scoping) comes from the same index dir's `build_manifest.json`, exactly what
    `rag-init`'s `--load-existing` CLI path and `PgvectorRetriever` itself both read."""
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


@pytest.mark.asyncio
async def test_chat_pipelines_knowledge_tool_seam_returns_real_corpus_doc_ids_for_the_daniel_question(
    retriever: PgvectorRetriever,
) -> None:
    server = build_knowledge_server(retriever)
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        result = await client.call_tool("search_knowledge", {"query": QUESTION})
    payload = json.loads(result.content[0].text)

    assert payload, "the chat pipeline's retrieval seam returned nothing for the Daniel question"
    assert len(payload) <= DEPLOYED_K  # the production width the graph actually asks for

    doc_ids = [item["doc_id"] for item in payload]
    assert not (set(doc_ids) & TOY_CORPUS_DOC_IDS)  # never the hermetic toy corpus
    assert all(doc_id.startswith("doc-") for doc_id in doc_ids)  # the real corpus-0.1.1 id shape

    print(f"\nDaniel question {QUESTION!r} via the chat pipeline's own search_knowledge tool call:")
    for item in payload:
        print(f"  doc_id={item['doc_id']!r}")
