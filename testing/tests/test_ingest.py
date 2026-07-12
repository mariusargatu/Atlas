"""The ingestion pipeline, hermetic parts: parquet schema assembly, manifest field construction,
loader SQL text (against a recording fake connection), embedding batching against a mock HTTP
transport. No Docker, no network: `httpx.MockTransport` and a hand rolled fake psycopg connection
stand in for TEI and Postgres. The live end to end path (real TEI, real Postgres, HNSW query) is
`test_ingest_live.py`, marked `live` and excluded from this hermetic lane.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pyarrow.parquet as pq
import pytest
from rag_tools import chunker, fingerprint, ingest
from rag_tools.fingerprint import EmbeddingFingerprint

from .fixtures import corpus_expectations

CORPUS_VERSION = "corpus-0.1.1"
MODELS_LOCK = Path("models.lock")


# --- corpus reading + chunking -----------------------------------------------------------------


def test_load_corpus_docs_reads_every_committed_doc_sorted_by_doc_id() -> None:
    docs = ingest.load_corpus_docs(CORPUS_VERSION)
    assert len(docs) == corpus_expectations.DOC_COUNT
    doc_ids = [d["doc_id"] for d in docs]
    assert doc_ids == sorted(doc_ids)
    for doc in docs:
        assert doc["text"]
        assert doc["doc_version"]
        assert isinstance(doc["placements"], list)


def test_chunk_corpus_degrades_to_one_chunk_per_doc_sorted_by_chunk_id() -> None:
    records = ingest.chunk_corpus(CORPUS_VERSION)
    assert len(records) == corpus_expectations.CHUNK_COUNT
    chunk_ids = [r.chunk_id for r in records]
    assert chunk_ids == sorted(chunk_ids)
    # one chunk per doc on this corpus: every doc is far under the chunker's split threshold
    assert len({r.doc_id for r in records}) == corpus_expectations.DOC_COUNT


# --- L2 normalization -----------------------------------------------------------------------------


def test_l2_normalize_unit_length() -> None:
    import math

    normalized = ingest.l2_normalize([3.0, 4.0])
    assert normalized == pytest.approx([0.6, 0.8])
    assert math.sqrt(sum(c * c for c in normalized)) == pytest.approx(1.0)


def test_l2_normalize_zero_vector_is_returned_unchanged() -> None:
    assert ingest.l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


# --- parquet schema + table assembly ---------------------------------------------------------------


def test_parquet_schema_has_every_chunk_record_field_plus_embedding() -> None:
    schema = ingest.parquet_schema(dim=4)
    names = schema.names
    assert names == [
        "chunk_id",
        "parent_id",
        "doc_id",
        "doc_version",
        "doc_type",
        "heading_path",
        "char_span",
        "token_count",
        "content_hash",
        "entity_ids",
        "chunker_version",
        "corpus_version",
        "doc_title",
        "text",
        "embedding",
    ]
    embedding_field = schema.field("embedding")
    assert embedding_field.type.list_size == 4  # fixed size list column


def _fixture_record(**overrides) -> chunker.ChunkRecord:
    defaults = dict(
        doc_id="doc-fixture",
        doc_type="plan_page",
        text="# Fixture Doc\n\nA short fixture body under the split threshold.",
        doc_version="version-a",
        corpus_version=CORPUS_VERSION,
        placements=[],
    )
    defaults.update(overrides)
    (record,) = chunker.chunk_document(**defaults)
    return record


def test_build_table_pairs_each_record_with_its_vector_in_order() -> None:
    records = [_fixture_record(doc_id="doc-a"), _fixture_record(doc_id="doc-b")]
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    table = ingest.build_table(records, vectors, dim=2)
    assert table.num_rows == 2
    assert table.column("doc_id").to_pylist() == ["doc-a", "doc-b"]
    assert table.column("embedding").to_pylist() == [[pytest.approx(0.1), pytest.approx(0.2)], [pytest.approx(0.3), pytest.approx(0.4)]]
    assert table.column("char_span").to_pylist() == [list(records[0].char_span), list(records[1].char_span)]


def test_build_table_rejects_mismatched_records_and_vectors_lengths() -> None:
    records = [_fixture_record()]
    with pytest.raises(ValueError, match="same length"):
        ingest.build_table(records, [[0.1], [0.2]], dim=1)


def test_write_parquet_round_trips_every_field(tmp_path: Path) -> None:
    records = [_fixture_record(doc_id="doc-a")]
    vectors = [[0.5, 0.25]]
    table = ingest.build_table(records, vectors, dim=2)
    out = tmp_path / "chunks.parquet"
    ingest.write_parquet(out, table)

    read_back = pq.read_table(out)
    assert read_back.num_rows == 1
    row = read_back.to_pylist()[0]
    assert row["chunk_id"] == records[0].chunk_id
    assert row["embedding"] == [pytest.approx(0.5), pytest.approx(0.25)]
    assert row["heading_path"] == list(records[0].heading_path)


# --- manifest field construction ------------------------------------------------------------------


def _fingerprint(**overrides) -> EmbeddingFingerprint:
    defaults = dict(
        model_id="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        dim=1024,
        normalize=True,
        query_prefix="",
        document_prefix="",
        provider="local-tei",
        server_version="1.9.0",
    )
    defaults.update(overrides)
    return EmbeddingFingerprint(**defaults)


def test_build_fingerprint_dict_carries_every_field_including_server_version() -> None:
    fp = _fingerprint()
    data = ingest.build_fingerprint_dict(fp)
    assert data == {
        "model_id": "BAAI/bge-m3",
        "revision": "5617a9f61b028005a4858fdac845db406aefb181",
        "dim": 1024,
        "normalize": True,
        "query_prefix": "",
        "document_prefix": "",
        "provider": "local-tei",
        "server_version": "1.9.0",
    }


def test_build_index_manifest_fields() -> None:
    fp = _fingerprint()
    reranker_entry = {"model_id": "BAAI/bge-reranker-v2-m3", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"}
    manifest = ingest.build_index_manifest(
        corpus_version=CORPUS_VERSION,
        corpus_content_hash="deadbeef" * 8,
        chunker_hash_value=chunker.chunker_hash(),
        fp=fp,
        reranker_lock_entry=reranker_entry,
        index_params={"m": 16, "ef_construction": 128},
        doc_count=45,
        chunk_count=45,
    )
    assert manifest["corpus_version"] == CORPUS_VERSION
    assert manifest["corpus_content_hash"] == "deadbeef" * 8
    assert manifest["chunker_hash"] == chunker.chunker_hash()
    assert manifest["chunker_version"] == chunker.CHUNKER_VERSION
    assert manifest["index_params"] == {"m": 16, "ef_construction": 128}
    assert manifest["doc_count"] == 45
    assert manifest["chunk_count"] == 45
    assert manifest["models_lock"]["embedding"] == {"model_id": fp.model_id, "revision": fp.revision}
    assert manifest["models_lock"]["reranker"] == reranker_entry
    # index_build_id is a real content addressed id, not a placeholder
    assert len(manifest["index_build_id"]) == 16
    assert manifest["index_build_id"] == ingest.index_build_id(
        CORPUS_VERSION, chunker.chunker_hash(), fp, {"m": 16, "ef_construction": 128}
    )


def test_build_index_manifest_flips_index_build_id_when_chunker_hash_changes() -> None:
    fp = _fingerprint()
    reranker_entry = {"model_id": "BAAI/bge-reranker-v2-m3", "revision": "x" * 40}
    baseline = ingest.build_index_manifest(
        corpus_version=CORPUS_VERSION,
        corpus_content_hash="a",
        chunker_hash_value="aaaaaaaaaaaaaaaa",
        fp=fp,
        reranker_lock_entry=reranker_entry,
        index_params={"m": 16, "ef_construction": 128},
        doc_count=1,
        chunk_count=1,
    )
    changed = ingest.build_index_manifest(
        corpus_version=CORPUS_VERSION,
        corpus_content_hash="a",
        chunker_hash_value="bbbbbbbbbbbbbbbb",
        fp=fp,
        reranker_lock_entry=reranker_entry,
        index_params={"m": 16, "ef_construction": 128},
        doc_count=1,
        chunk_count=1,
    )
    assert baseline["index_build_id"] != changed["index_build_id"]


def test_load_reranker_lock_entry_reads_the_real_committed_models_lock() -> None:
    entry = ingest.load_reranker_lock_entry(MODELS_LOCK, "BAAI/bge-reranker-v2-m3")
    assert entry == {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    }


def test_load_reranker_lock_entry_raises_on_missing_model_id(tmp_path: Path) -> None:
    lock_path = tmp_path / "models.lock"
    lock_path.write_text(json.dumps({"embedding": [], "reranker": [], "generator": []}))
    with pytest.raises(ValueError, match="no reranker entry"):
        ingest.load_reranker_lock_entry(lock_path, "BAAI/bge-reranker-v2-m3")


# --- embedding over a mock HTTP transport (no real network) ----------------------------------------


def test_embed_texts_batches_requests_and_preserves_order() -> None:
    seen_batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        batch = payload["inputs"]
        seen_batches.append(batch)
        return httpx.Response(200, json=[[float(len(text)), 0.0] for text in batch])

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei-embed.test")
    texts = [f"text-{i}" for i in range(5)]
    vectors = ingest.embed_texts("http://tei-embed.test", texts, batch_size=2, client=client)

    assert len(vectors) == 5
    assert [v[0] for v in vectors] == [float(len(t)) for t in texts]
    assert seen_batches == [["text-0", "text-1"], ["text-2", "text-3"], ["text-4"]]


def test_embed_texts_fails_loud_with_no_retry_on_http_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": "backend not ready"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei-embed.test")
    with pytest.raises(httpx.HTTPStatusError):
        ingest.embed_texts("http://tei-embed.test", ["a", "b"], batch_size=2, client=client)
    assert calls["n"] == 1  # exactly one attempt: no retry


def test_fetch_server_version_reads_info_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/info"
        return httpx.Response(200, json={"version": "1.9.0", "model_id": "BAAI/bge-m3"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei-embed.test")
    assert ingest.fetch_server_version("http://tei-embed.test", client=client) == "1.9.0"


def test_fetch_server_version_fails_loud_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei-embed.test")
    with pytest.raises(httpx.HTTPStatusError):
        ingest.fetch_server_version("http://tei-embed.test", client=client)


# --- pg loader: SQL text against a recording fake connection ---------------------------------------


class _RecordingCursor:
    def __init__(self, sink: list[tuple], *, count_result: int | None = None) -> None:
        self._sink = sink
        self._count_result = count_result
        self._last_sql = ""

    def __enter__(self) -> "_RecordingCursor":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        self._sink.append(("execute", sql, params))
        self._last_sql = sql

    def executemany(self, sql: str, params_seq: object) -> None:
        params_list = list(params_seq)
        self._sink.append(("executemany", sql, params_list))
        self._last_sql = sql

    def fetchone(self) -> tuple:
        # only the post load row count check (SP4 task 3) fetches anything in this fake's world.
        if "count(*)" not in self._last_sql.lower():
            raise AssertionError(f"fetchone() called after an unexpected statement: {self._last_sql!r}")
        if self._count_result is not None:
            return (self._count_result,)
        # the happy path default: the count check passes because it matches what was just inserted.
        executemany_calls = [c for c in self._sink if c[0] == "executemany"]
        inserted = len(executemany_calls[-1][2]) if executemany_calls else 0
        return (inserted,)


class _RecordingConnection:
    def __init__(self, *, count_result: int | None = None) -> None:
        self.calls: list[tuple] = []
        self._count_result = count_result

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self.calls, count_result=self._count_result)

    def commit(self) -> None:
        self.calls.append(("commit",))


def test_vector_literal_formats_a_pgvector_array_literal() -> None:
    assert ingest.vector_literal([0.1, -0.5, 1.0]) == "[0.1,-0.5,1.0]"


def test_create_schema_emits_expected_ddl_text() -> None:
    conn = _RecordingConnection()
    ingest.create_schema(conn, dim=1024, index_params={"m": 16, "ef_construction": 128})

    executed_sql = [call[1] for call in conn.calls if call[0] == "execute"]
    assert any("CREATE EXTENSION IF NOT EXISTS vector" in sql for sql in executed_sql)

    table_sql = next(sql for sql in executed_sql if "CREATE TABLE" in sql)
    assert "vector(1024)" in table_sql
    assert "tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED" in table_sql
    assert "chunk_id text PRIMARY KEY" in table_sql
    assert "index_build_id text NOT NULL" in table_sql

    gin_sql = next(sql for sql in executed_sql if "USING gin" in sql)
    assert "chunks_tsv_idx" in gin_sql

    build_id_idx_sql = next(sql for sql in executed_sql if "chunks_index_build_id_idx" in sql)
    assert "CREATE INDEX IF NOT EXISTS chunks_index_build_id_idx ON chunks (index_build_id)" in build_id_idx_sql

    hnsw_sql = next(sql for sql in executed_sql if "USING hnsw" in sql)
    assert "m = 16" in hnsw_sql
    assert "ef_construction = 128" in hnsw_sql
    assert "vector_cosine_ops" in hnsw_sql

    assert conn.calls[-1] == ("commit",)


def test_create_schema_rejects_a_non_positive_dim() -> None:
    conn = _RecordingConnection()
    with pytest.raises(ValueError, match="dim must be a positive int"):
        ingest.create_schema(conn, dim=0, index_params={"m": 16, "ef_construction": 128})


def test_load_parquet_creates_schema_then_inserts_every_row(tmp_path: Path) -> None:
    records = [_fixture_record(doc_id="doc-a"), _fixture_record(doc_id="doc-b")]
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    table = ingest.build_table(records, vectors, dim=2)
    parquet_path = tmp_path / "chunks.parquet"
    ingest.write_parquet(parquet_path, table)

    conn = _RecordingConnection()
    row_count = ingest.load_parquet(
        conn, parquet_path, dim=2, build_id="testbuild0000001", index_params={"m": 16, "ef_construction": 128}
    )

    assert row_count == 2
    call_kinds = [call[0] for call in conn.calls]
    # schema creation (execute calls + a commit) happens before the insert (executemany + a commit)
    assert call_kinds.index("executemany") > call_kinds.index("commit")

    executemany_call = next(call for call in conn.calls if call[0] == "executemany")
    _, insert_sql, params_list = executemany_call
    assert "INSERT INTO chunks" in insert_sql
    assert "ON CONFLICT (chunk_id) DO NOTHING" in insert_sql
    assert "%(embedding)s::vector" in insert_sql
    assert "%(index_build_id)s" in insert_sql
    assert len(params_list) == 2
    assert {p["doc_id"] for p in params_list} == {"doc-a", "doc-b"}
    assert {p["index_build_id"] for p in params_list} == {"testbuild0000001"}  # every row stamped
    # Parquet stores embedding as float32 (the fixed size list column), so the round tripped literal
    # carries float32 precision noise vs. the original float64 python list; compare numerically.
    literal = params_list[0]["embedding"]
    parsed = [float(x) for x in literal.strip("[]").split(",")]
    assert parsed == pytest.approx(vectors[0], abs=1e-6)
    assert params_list[0]["char_span_start"] == records[0].char_span[0]
    assert params_list[0]["char_span_end"] == records[0].char_span[1]

    assert conn.calls[-1] == ("commit",)


def test_load_parquet_runs_a_post_load_count_check_against_the_build_id(tmp_path: Path) -> None:
    # SP4 task 3: the count check queries specifically `WHERE index_build_id = %(build_id)s`, the
    # same filter both SQL arms in `atlas.adapters.pgvector_retriever` bind, not a bare `count(*)`.
    records = [_fixture_record(doc_id="doc-a")]
    vectors = [[0.1, 0.2]]
    table = ingest.build_table(records, vectors, dim=2)
    parquet_path = tmp_path / "chunks.parquet"
    ingest.write_parquet(parquet_path, table)

    conn = _RecordingConnection()
    ingest.load_parquet(conn, parquet_path, dim=2, build_id="testbuild0000001", index_params={"m": 16, "ef_construction": 128})

    count_call = next(call for call in conn.calls if call[0] == "execute" and "count(*)" in call[1].lower())
    assert "index_build_id = %(build_id)s" in count_call[1]
    assert count_call[2]["build_id"] == "testbuild0000001"


def test_load_parquet_raises_loud_on_a_post_load_count_mismatch(tmp_path: Path) -> None:
    """The reviewer's fail empty scenario (SP4 task 3): two builds of the SAME corpus_version that
    differ only in something `chunk_id`'s hash does not see (an embedding model change is the real
    example) collide on every chunk_id, so `ON CONFLICT (chunk_id) DO NOTHING` silently keeps the
    FIRST build's rows and the second build's `index_build_id` backs zero of them. This fake pins
    that: `_RecordingConnection(count_result=...)` simulates the post load count coming back lower
    than the parquet's own row count (as if DO NOTHING had swallowed a row under a stale
    index_build_id), and `load_parquet` must raise loud instead of returning a silently wrong count."""
    records = [_fixture_record(doc_id="doc-a"), _fixture_record(doc_id="doc-b")]
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    table = ingest.build_table(records, vectors, dim=2)
    parquet_path = tmp_path / "chunks.parquet"
    ingest.write_parquet(parquet_path, table)

    # 1 row actually carries this build_id even though the parquet has 2: a collision swallowed one.
    conn = _RecordingConnection(count_result=1)
    with pytest.raises(ingest.LoadCountMismatchError) as excinfo:
        ingest.load_parquet(
            conn, parquet_path, dim=2, build_id="testbuild0000002", index_params={"m": 16, "ef_construction": 128}
        )
    message = str(excinfo.value)
    assert "testbuild0000002" in message
    assert "1" in message
    assert "2" in message


# --- CLI: --load-existing (SP3 task 7's compose init service reuses this loader) -----------------


def test_main_load_existing_skips_build_and_loads_the_given_index_dir(tmp_path: Path, monkeypatch, capsys) -> None:
    # The compose init service runs this CLI in --load-existing mode to load the COMMITTED
    # indexes/<name>/chunks.parquet into Postgres without rebuilding it (no TEI call, no
    # corpus/rendered read): "a small compose init service running the existing ingest loader."
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "fingerprint.json").write_text(json.dumps({"dim": 1024}))
    (index_dir / "build_manifest.json").write_text(json.dumps({"index_build_id": "cli-test-buildid"}))
    (index_dir / "chunks.parquet").write_bytes(b"")  # never read: load_parquet is faked below

    def _boom(**kwargs):
        raise AssertionError("build_index must not run in --load-existing mode")

    monkeypatch.setattr(ingest, "build_index", _boom)

    calls: list[tuple] = []

    def _fake_load_parquet(conn, parquet_path, *, dim, build_id, index_params=ingest.INDEX_PARAMS):
        calls.append((parquet_path, dim, build_id))
        return 45

    monkeypatch.setattr(ingest, "load_parquet", _fake_load_parquet)

    class _FakeConn:
        def __enter__(self) -> "_FakeConn":
            return self

        def __exit__(self, *exc_info: object) -> bool:
            return False

    monkeypatch.setattr("psycopg.connect", lambda dsn: _FakeConn())

    ingest.main(["--load-existing", str(index_dir), "--postgres-dsn", "postgresql://unused/unused"])

    assert calls == [(index_dir / "chunks.parquet", 1024, "cli-test-buildid")]
    assert "loaded 45 rows into postgres" in capsys.readouterr().out


def test_main_load_existing_can_skip_the_load_too(tmp_path: Path, monkeypatch, capsys) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "fingerprint.json").write_text(json.dumps({"dim": 1024}))

    monkeypatch.setattr(ingest, "build_index", lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not build")))
    monkeypatch.setattr(ingest, "load_parquet", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not load")))

    ingest.main(["--load-existing", str(index_dir), "--skip-load"])
    assert "wrote index build to" not in capsys.readouterr().out


# --- the committed index's identity, recomputed from its own declared inputs -----------------------


def test_the_committed_index_identity_recomputes_from_its_own_declared_inputs() -> None:
    """`index_build_id` and the index directory NAME are both content addressed over
    (corpus_version, chunker_hash, embedding model, revision, index_params). Nothing checked that
    the committed build actually satisfies its own definition, so the directory name was hand typed
    into 25 files and the build id into two more, with no gate anywhere: mutating a chunker
    threshold changes `chunker_hash()` by design, and left the whole suite green.

    Fully hermetic: reads the committed manifests and recomputes, no TEI, no Postgres, no network.
    """
    manifest = corpus_expectations.index_manifest()
    fp_data = corpus_expectations.index_fingerprint()
    fp = EmbeddingFingerprint(
        provider=fp_data["provider"],
        model_id=fp_data["model_id"],
        revision=fp_data["revision"],
        dim=fp_data["dim"],
        normalize=fp_data["normalize"],
        query_prefix=fp_data["query_prefix"],
        document_prefix=fp_data["document_prefix"],
    )

    # the chunker config the build declares must be the chunker this repo currently ships
    assert manifest["chunker_version"] == chunker.CHUNKER_VERSION
    assert manifest["chunker_hash"] == chunker.chunker_hash()

    # the embedding model the fingerprint names must be the one the build manifest locked
    assert manifest["models_lock"]["embedding"]["model_id"] == fp.model_id
    assert manifest["models_lock"]["embedding"]["revision"] == fp.revision

    # and the two identities must both recompute
    assert manifest["index_build_id"] == fingerprint.index_build_id(
        manifest["corpus_version"], manifest["chunker_hash"], fp, manifest["index_params"]
    )
    assert corpus_expectations.COMMITTED_INDEX_DIR.name == fingerprint.index_name(
        manifest["corpus_version"], fp.model_id, manifest["chunker_hash"]
    )


def test_the_committed_index_agrees_with_the_corpus_it_was_built_from() -> None:
    """The index build manifest carries `corpus_version` and `corpus_content_hash` copied from the
    corpus manifest at build time. If they disagree, the committed index was built from a corpus
    that no longer exists, and every live retrieval test is scoring against stale embeddings."""
    corpus = corpus_expectations.corpus_manifest()
    index = corpus_expectations.index_manifest()
    assert index["corpus_version"] == corpus["corpus_version"]
    assert index["corpus_content_hash"] == corpus["content_hash"]
    assert index["doc_count"] == corpus["doc_count"]


def test_the_committed_index_params_are_the_ones_the_ingest_module_builds_with() -> None:
    """`ingest.INDEX_PARAMS` is interpolated into the HNSW DDL. A committed build whose manifest
    declares different params than the module would use today means a re-ingest would silently
    produce a different index under a different build id."""
    assert corpus_expectations.INDEX_PARAMS == ingest.INDEX_PARAMS
