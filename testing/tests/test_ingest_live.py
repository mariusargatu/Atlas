"""The ingestion pipeline, live: end to end against the compose retrieval stack (real TEI, real
Postgres). Marked `live`, excluded from `task test` (the hermetic PR lane); run via `task test:live`
with `docker compose up postgres tei-embed tei-rerank` already healthy.

Builds the index into a throwaway `tmp_path` (never the committed `indexes/` tree: the committed
artifact is a deliberate, separate `task rag:ingest` run, not a side effect of a test), loads it
into the real `chunks` table (dropped and recreated at the start of the test for isolation), then
proves the HNSW index actually returns the right document: querying "contract free plan" should
surface `doc-plan_page-plan-fiber-100`, the only no-contract Fiber 100 plan page in the corpus.

Ingest idempotence finding (SP3 final review, recorded in full in
`docs/measurements/sp3-rag-spine.md`): `chunks.parquet` is NOT byte identical across independent
rebuilds of the same corpus_version/model/chunker_hash. Two independent live probes rebuilding
corpus-0.1.1 found 9 of 45 and 25 of 45 committed vectors respectively differing from a fresh
rebuild's corresponding chunk_id (the count is not stable run to run); minimum cosine similarity
was 0.999999999998 and 0.9999999999993494 respectively (both twelve nines) -- real float noise, not
a determinism bug in this codebase: TEI's ONNX Runtime backend sums per-token embeddings across CPU
threads in an order that is not pinned run to run, so the last few mantissa bits of a float32
embedding can differ between two otherwise-identical builds. The committed
`indexes/corpus-0.1.1-bge-m3-03f983e0/chunks.parquet` is therefore treated as the frozen reference
artifact, not something a rebuild is expected to reproduce byte for byte;
`test_rebuilt_index_vectors_agree_with_the_committed_parquet_within_cosine_tolerance` below is the
live gate on that tolerance (the plan's Task 5 fallback, made real).
"""
from __future__ import annotations

import math
from pathlib import Path

import psycopg
import pyarrow.parquet as pq
import pytest
from rag_tools import ingest
from .fixtures import corpus_expectations

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_INDEX_DIR = REPO_ROOT / "indexes" / "corpus-0.1.1-bge-m3-03f983e0"
CORPUS_VERSION = "corpus-0.1.1"
TEI_EMBED_URL = "http://localhost:8081"
POSTGRES_DSN = "postgresql://atlas:atlas-dev-password@localhost:5433/atlas"
COSINE_FLOOR = 0.9999  # generous: the measured worst case is 0.999999999998, twelve nines in

# SP4 final fix wave carryover: also live_slow -- every test here calls ingest.build_index(...), a
# full corpus rebuild against the live TEI embed server, the dominant cost the report's own final
# gate measurement named.
pytestmark = [pytest.mark.live, pytest.mark.live_slow]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@pytest.fixture
def pg_conn():
    # WARNING: `DROP TABLE IF EXISTS chunks` wipes the SHARED live `chunks` table, not a table
    # scoped to this test file. `test_pgvector_adapter_live.py` and `test_naive_variant_live.py`
    # both have session scoped `ensure_chunks_loaded` fixtures that assume the committed index
    # build's rows are already there (or idempotently reload them); running THIS module against the
    # same Postgres instance drops that data out from under a fixture that has already run in the
    # same `task test:live` session (harmless -- the next call to `ensure_chunks_loaded` just
    # reloads it -- but not free, and a source of surprise if you're trying to inspect the table's
    # contents mid session). Isolation here means "start from an empty table," not "this table is
    # private to this test."
    conn = psycopg.connect(POSTGRES_DSN)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS chunks;")
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def test_ingest_end_to_end_loads_45_rows_and_hnsw_finds_fiber_100(tmp_path: Path, pg_conn) -> None:
    out_dir = ingest.build_index(
        corpus_version=CORPUS_VERSION,
        tei_embed_url=TEI_EMBED_URL,
        index_root=tmp_path,
    )

    assert (out_dir / "chunks.parquet").exists()
    assert (out_dir / "fingerprint.json").exists()
    assert (out_dir / "build_manifest.json").exists()

    import json

    fp = json.loads((out_dir / "fingerprint.json").read_text())
    assert fp["server_version"]  # filled in from the live TEI's /info, not None
    manifest = json.loads((out_dir / "build_manifest.json").read_text())

    row_count = ingest.load_parquet(pg_conn, out_dir / "chunks.parquet", dim=fp["dim"], build_id=manifest["index_build_id"])
    assert row_count == corpus_expectations.CHUNK_COUNT

    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks;")
        (count,) = cur.fetchone()
    assert count == corpus_expectations.CHUNK_COUNT

    [query_vector] = ingest.embed_texts(TEI_EMBED_URL, ["contract free plan"])
    if fp["normalize"]:
        query_vector = ingest.l2_normalize(query_vector)
    literal = ingest.vector_literal(query_vector)

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id FROM chunks ORDER BY embedding <=> %s::vector LIMIT 5;",
            (literal,),
        )
        top_doc_ids = [row[0] for row in cur.fetchall()]

    assert "doc-plan_page-plan-fiber-100" in top_doc_ids


def test_load_parquet_is_idempotent_on_rerun(tmp_path: Path, pg_conn) -> None:
    out_dir = ingest.build_index(
        corpus_version=CORPUS_VERSION,
        tei_embed_url=TEI_EMBED_URL,
        index_root=tmp_path,
    )
    import json

    fp = json.loads((out_dir / "fingerprint.json").read_text())
    manifest = json.loads((out_dir / "build_manifest.json").read_text())
    build_id = manifest["index_build_id"]

    first = ingest.load_parquet(pg_conn, out_dir / "chunks.parquet", dim=fp["dim"], build_id=build_id)
    second = ingest.load_parquet(pg_conn, out_dir / "chunks.parquet", dim=fp["dim"], build_id=build_id)
    assert first == corpus_expectations.CHUNK_COUNT
    assert second == corpus_expectations.CHUNK_COUNT  # ON CONFLICT DO NOTHING: rerun reports the same count, no duplicate rows

    with pg_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks;")
        (count,) = cur.fetchone()
    assert count == corpus_expectations.CHUNK_COUNT


def test_rebuilt_index_vectors_agree_with_the_committed_parquet_within_cosine_tolerance(tmp_path: Path) -> None:
    """The plan's Task 5 cosine gate, made real (SP3 final review). See this module's own docstring
    for the full finding: rebuilding corpus-0.1.1 is not byte identical to the committed
    `indexes/corpus-0.1.1-bge-m3-03f983e0/chunks.parquet` (TEI's ONNX Runtime backend sums per-token
    embeddings across CPU threads in a non-pinned order, so float32 rounding noise differs run to
    run even though nothing about the corpus, model, or chunker config changed) -- two independent
    probes observed 9 of 45 and 25 of 45 vectors differing respectively (the count itself is not
    stable run to run), minimum cosine similarity 0.999999999998 and 0.9999999999993494. This test builds
    a FRESH index into `tmp_path` (never touching the committed `indexes/` tree -- no Postgres
    involved either, this is a pure parquet-to-parquet comparison) and asserts every rebuilt vector
    agrees with the committed parquet's same chunk_id above a generous 0.9999 floor: the committed
    artifact is the frozen reference, this is the live check that a fresh rebuild still agrees with
    it, not a byte-identity assertion that would be flaky by construction."""
    out_dir = ingest.build_index(
        corpus_version=CORPUS_VERSION,
        tei_embed_url=TEI_EMBED_URL,
        index_root=tmp_path,
    )
    rebuilt_rows = pq.read_table(out_dir / "chunks.parquet").to_pylist()
    committed_rows = pq.read_table(COMMITTED_INDEX_DIR / "chunks.parquet").to_pylist()

    rebuilt_by_id = {row["chunk_id"]: row["embedding"] for row in rebuilt_rows}
    committed_by_id = {row["chunk_id"]: row["embedding"] for row in committed_rows}
    # Content addressed chunk_ids: a fresh rebuild against the same corpus_version/doc_version/
    # chunker_version must name the exact same 45 chunk_ids as the committed build, or this
    # comparison is meaningless (comparing two different chunks' vectors, not the same chunk
    # rebuilt). This is itself a determinism check, gated, before the cosine measurement below.
    assert set(rebuilt_by_id) == set(committed_by_id)

    cosines = {chunk_id: _cosine(rebuilt_by_id[chunk_id], committed_by_id[chunk_id]) for chunk_id in rebuilt_by_id}
    min_chunk_id = min(cosines, key=lambda cid: cosines[cid])
    min_cosine = cosines[min_chunk_id]
    exact_matches = sum(1 for c in cosines.values() if c == 1.0)

    print(f"\nrebuilt vs committed parquet, corpus-0.1.1 ({len(cosines)} chunks):")
    print(f"  exact float equality: {exact_matches} of {len(cosines)}")
    print(f"  min cosine similarity: {min_cosine!r} (chunk_id={min_chunk_id!r})")

    assert min_cosine > COSINE_FLOOR, (
        f"chunk {min_chunk_id!r} drifted below the cosine floor ({min_cosine!r} <= {COSINE_FLOOR}); "
        "this is no longer float noise, treat it as a real embedding regression"
    )
