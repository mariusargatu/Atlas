"""`PgvectorRetriever` (SP3 task 6), hermetic: SQL construction, RRF wiring, and the fail closed
fingerprint check, all against a recording fake psycopg connection and `httpx.MockTransport` TEI
doubles. No Docker, no network. The live end to end path (real Postgres, real TEI, HNSW vs exact
scan agreement, rerank participation) is `test_pgvector_adapter_live.py`, marked `live` and
excluded from this hermetic lane.

SP4 task 3 additions: typed error surfaces (embed/rerank/postgres failures never leak a raw
httpx/psycopg exception), the breaker fail fast short circuit at the adapter level, the rerank
batch size read off the rerank server's own `/info`, and the `SearchResult` contextvar carrier's
isolation across interleaved calls on different threads.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import psycopg
import pytest
import stamina
from atlas.adapters import pgvector_retriever
from atlas.adapters.pgvector_retriever import FingerprintMismatchError, PgvectorRetriever
from atlas.adapters.resilience import EmbeddingServiceError, RerankServiceError, RetrievalError
from atlas.domain.retrieval import RetrievalConfig

MODEL_ID = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


DEFAULT_BUILD_ID = "0123456789abcdef"


@pytest.fixture(autouse=True)
def _no_real_backoff_sleep():
    # SP4 task 3: RetryPolicy wraps every TEI/pg call now, so a test that drives a failure path
    # would otherwise sleep real wall clock time between attempts. `stamina.set_testing` is
    # stamina's own documented pattern for this: `cap=True` keeps RetryPolicy's own `attempts=3`
    # governing attempt counts, only the sleep itself is disabled.
    with stamina.set_testing(True, attempts=50, cap=True):
        yield


def _write_fingerprint(tmp_path: Path, *, build_id: str = DEFAULT_BUILD_ID, **overrides: object) -> Path:
    fields = {
        "dim": 3,
        "document_prefix": "",
        "model_id": MODEL_ID,
        "normalize": True,
        "provider": "local-tei",
        "query_prefix": "",
        "revision": REVISION,
        "server_version": "1.9.3",
    }
    fields.update(overrides)
    index_dir = tmp_path / "index"
    index_dir.mkdir(exist_ok=True)
    (index_dir / "fingerprint.json").write_text(json.dumps(fields))
    (index_dir / "build_manifest.json").write_text(json.dumps({"index_build_id": build_id}))
    return index_dir


def _info_handler(model_id: str = MODEL_ID, model_sha: str = REVISION):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/info"
        return httpx.Response(200, json={"model_id": model_id, "model_sha": model_sha, "version": "1.9.3"})

    return handler


def _embed_handler(vector: list[float]):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        body = json.loads(request.content)
        return httpx.Response(200, json=[vector for _ in body["inputs"]])

    return handler


def _failing_handler(status_code: int, path: str):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == path
        return httpx.Response(status_code, json={"error": f"{path} down"})

    return handler


def _rerank_handler(scores: dict[int, float]):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank"
        return httpx.Response(200, json=[{"index": i, "score": s} for i, s in scores.items()])

    return handler


def _tei_client(info_handler=None, embed_handler=None, rerank_handler=None) -> httpx.Client:
    routes: dict[str, object] = {}
    if info_handler:
        routes["/info"] = info_handler
    if embed_handler:
        routes["/embed"] = embed_handler
    if rerank_handler:
        routes["/rerank"] = rerank_handler

    def dispatch(request: httpx.Request) -> httpx.Response:
        return routes[request.url.path](request)

    return httpx.Client(transport=httpx.MockTransport(dispatch), base_url="http://tei.test")


# --- recording fake psycopg connection --------------------------------------------------------------


class _FakeCursor:
    def __init__(self, sink: list[tuple], vector_rows: list[tuple], tsv_rows: list[tuple], hydrate_rows: list[tuple]) -> None:
        self._sink = sink
        self._vector_rows = vector_rows
        self._tsv_rows = tsv_rows
        self._hydrate_rows = hydrate_rows
        self._last_sql = ""

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        self._sink.append(("execute", sql, params))
        self._last_sql = sql

    def fetchall(self) -> list[tuple]:
        if "embedding <=>" in self._last_sql:
            return self._vector_rows
        if "websearch_to_tsquery" in self._last_sql:
            return self._tsv_rows
        if "ANY(" in self._last_sql:
            return self._hydrate_rows
        raise AssertionError(f"fetchall() called after an unexpected statement: {self._last_sql!r}")


class _FakeConnection:
    def __init__(self, vector_rows: list[tuple], tsv_rows: list[tuple], hydrate_rows: list[tuple]) -> None:
        self.calls: list[tuple] = []
        self.closed = False
        self._vector_rows = vector_rows
        self._tsv_rows = tsv_rows
        self._hydrate_rows = hydrate_rows

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.calls, self._vector_rows, self._tsv_rows, self._hydrate_rows)

    def commit(self) -> None:
        self.calls.append(("commit",))

    def close(self) -> None:
        self.closed = True


def _hydrate_row(chunk_id: str, doc_id: str, text: str) -> tuple:
    return (chunk_id, doc_id, doc_id, "v1", "help", ["Title"], 0, len(text), text, [])


# --- fingerprint fail closed --------------------------------------------------------------------


def test_construct_succeeds_when_fingerprint_matches_tei_info(tmp_path: Path) -> None:
    index_dir = _write_fingerprint(tmp_path)
    client = _tei_client(info_handler=_info_handler())
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=client,
        rerank_client=_tei_client(),
        connect=lambda: _FakeConnection([], [], []),
    )
    assert retriever is not None


def test_breaker_property_exposes_the_one_shared_circuit_breaker(tmp_path: Path) -> None:
    """SP6 task 5: `atlas.metrics`'s `/metrics` route reads `.breaker` off whatever retriever
    `server.py` selected. Inspection only, the same object every call site inside this adapter
    shares (never a fresh one per property access)."""
    from atlas.adapters.resilience import CircuitBreaker

    index_dir = _write_fingerprint(tmp_path)
    client = _tei_client(info_handler=_info_handler())
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=client,
        rerank_client=_tei_client(),
        connect=lambda: _FakeConnection([], [], []),
    )
    assert isinstance(retriever.breaker, CircuitBreaker)
    assert retriever.breaker is retriever.breaker  # same instance every access, not rebuilt
    assert retriever.breaker.state("tei-embed") == "closed"


def test_construct_refuses_on_model_id_mismatch(tmp_path: Path) -> None:
    index_dir = _write_fingerprint(tmp_path)
    client = _tei_client(info_handler=_info_handler(model_id="BAAI/other-model"))
    with pytest.raises(FingerprintMismatchError, match="does not match"):
        PgvectorRetriever(
            pg_dsn="postgresql://unused/unused",
            index_dir=index_dir,
            embed_client=client,
            rerank_client=_tei_client(),
            connect=lambda: _FakeConnection([], [], []),
        )


def test_construct_refuses_on_revision_mismatch(tmp_path: Path) -> None:
    index_dir = _write_fingerprint(tmp_path)
    client = _tei_client(info_handler=_info_handler(model_sha="deadbeef" * 5))
    with pytest.raises(FingerprintMismatchError) as excinfo:
        PgvectorRetriever(
            pg_dsn="postgresql://unused/unused",
            index_dir=index_dir,
            embed_client=client,
            rerank_client=_tei_client(),
            connect=lambda: _FakeConnection([], [], []),
        )
    message = str(excinfo.value)
    # the message must be in English and name both the expected and the actual revision, so an
    # operator reading the raised error (not a debugger) can tell what mismatched.
    assert REVISION in message
    assert "deadbeef" in message


def test_missing_fingerprint_file_raises_a_worded_error_naming_atlas_index_dir(tmp_path: Path) -> None:
    # SP3 task 7 ride along: a bad ATLAS_INDEX_DIR (typo'd, unmounted volume, never built) must fail
    # closed with a message an operator can act on, not a raw FileNotFoundError. This must fire
    # before any network call: embed_client/rerank_client below have no routes wired at all, so a
    # bug that reached _verify_fingerprint first would error with a KeyError on the empty handler
    # dict instead of FingerprintMismatchError, and this test would catch that too.
    missing_dir = tmp_path / "no-such-index"
    with pytest.raises(FingerprintMismatchError) as excinfo:
        PgvectorRetriever(
            pg_dsn="postgresql://unused/unused",
            index_dir=missing_dir,
            embed_client=_tei_client(),
            rerank_client=_tei_client(),
            connect=lambda: _FakeConnection([], [], []),
        )
    message = str(excinfo.value)
    assert "ATLAS_INDEX_DIR" in message
    assert str(missing_dir) in message
    assert "rag:ingest" in message  # a concrete remediation, not just "it's missing"


def test_missing_build_manifest_file_raises_a_worded_error_naming_atlas_index_dir(tmp_path: Path) -> None:
    # SP3 final review, table scoping: build_manifest.json is read alongside fingerprint.json, and
    # missing it must fail closed with the same worded, actionable style -- before any network call
    # (embed_client/rerank_client below have no routes wired), same as the missing fingerprint case
    # above. A build directory with fingerprint.json but no build_manifest.json is exactly the shape
    # an older, pre table scoping index build would have.
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "fingerprint.json").write_text(
        json.dumps(
            {
                "dim": 3,
                "document_prefix": "",
                "model_id": MODEL_ID,
                "normalize": True,
                "provider": "local-tei",
                "query_prefix": "",
                "revision": REVISION,
                "server_version": "1.9.3",
            }
        )
    )
    with pytest.raises(FingerprintMismatchError) as excinfo:
        PgvectorRetriever(
            pg_dsn="postgresql://unused/unused",
            index_dir=index_dir,
            embed_client=_tei_client(),
            rerank_client=_tei_client(),
            connect=lambda: _FakeConnection([], [], []),
        )
    message = str(excinfo.value)
    assert "build_manifest.json" in message
    assert "ATLAS_INDEX_DIR" in message
    assert str(index_dir) in message


def test_close_closes_both_tei_clients(tmp_path: Path) -> None:
    index_dir = _write_fingerprint(tmp_path)
    embed_client = _tei_client(info_handler=_info_handler())
    rerank_client = _tei_client()
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=embed_client,
        rerank_client=rerank_client,
        connect=lambda: _FakeConnection([], [], []),
    )
    assert not embed_client.is_closed and not rerank_client.is_closed
    retriever.close()
    assert embed_client.is_closed
    assert rerank_client.is_closed


# --- SQL construction + RRF wiring ----------------------------------------------------------------


def _make_retriever(
    tmp_path: Path, *, vector_rows, tsv_rows, hydrate_rows, embed_vector=None, rerank_scores=None, build_id=DEFAULT_BUILD_ID
):
    index_dir = _write_fingerprint(tmp_path, build_id=build_id)
    conn = _FakeConnection(vector_rows, tsv_rows, hydrate_rows)
    embed_client = _tei_client(
        info_handler=_info_handler(),
        embed_handler=_embed_handler(embed_vector or [0.1, 0.2, 0.3]),
    )
    rerank_client = _tei_client(rerank_handler=_rerank_handler(rerank_scores or {}))
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=embed_client,
        rerank_client=rerank_client,
        connect=lambda: conn,
    )
    return retriever, conn


def test_search_chunks_runs_both_arms_in_one_connection(tmp_path: Path) -> None:
    vector_rows = [("a",), ("b",), ("c",)]
    tsv_rows = [("b",), ("d",)]
    hydrate_rows = [
        _hydrate_row("a", "doc-a", "text a"),
        _hydrate_row("b", "doc-b", "text b"),
        _hydrate_row("c", "doc-c", "text c"),
        _hydrate_row("d", "doc-d", "text d"),
    ]
    retriever, conn = _make_retriever(tmp_path, vector_rows=vector_rows, tsv_rows=tsv_rows, hydrate_rows=hydrate_rows)

    config = RetrievalConfig(k_fused=10, k_final=4, rerank_enabled=False)
    results = retriever.search_chunks("plan question", 4, config)

    # exactly one connection was opened for the whole call (both rank queries + hydration)
    assert conn.closed is True
    kinds = [call[0] for call in conn.calls]
    assert kinds.count("execute") == 4  # SET LOCAL, vector arm, tsv arm, hydration -- one connection
    assert kinds.count("commit") == 1
    assert kinds.index("commit") == len(kinds) - 1  # commit happens last, after every query

    # RRF by hand: k=60, ranks are 1-based.
    # "a": vector rank 1 -> 1/61
    # "b": vector rank 2 -> 1/62, tsv rank 1 -> 1/61  => 1/62 + 1/61
    # "c": vector rank 3 -> 1/63
    # "d": tsv rank 2 -> 1/62
    expected_order = ["b", "a", "d", "c"]
    assert [c.doc_id for c in results] == [f"doc-{cid}" for cid in expected_order]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_both_sql_arms_order_by_chunk_id_as_a_deterministic_tie_break(tmp_path: Path) -> None:
    """SP3 final review, IMPORTANT: a live probe against this corpus found 13 way ties at identical
    `ts_rank` for one query, and vector distance ties are just as possible; without a stable tie
    break, `LIMIT` can silently return a different row set across otherwise identical runs -- a
    determinism break in fusion, not just cosmetic reordering. SQL text inspection via the
    recording fake, no live rerun needed to pin this."""
    hydrate_rows = [_hydrate_row("a", "doc-a", "text a")]
    retriever, conn = _make_retriever(tmp_path, vector_rows=[("a",)], tsv_rows=[("a",)], hydrate_rows=hydrate_rows)
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False)
    retriever.search_chunks("q", 1, config)

    executed_sql = [call[1] for call in conn.calls if call[0] == "execute"]
    vector_sql = next(sql for sql in executed_sql if "embedding <=>" in sql)
    tsv_sql = next(sql for sql in executed_sql if "ts_rank" in sql)

    assert "ORDER BY embedding <=> %(vector)s::vector ASC, chunk_id" in vector_sql
    assert "ORDER BY ts_rank(tsv, websearch_to_tsquery('english', %(query)s)) DESC, chunk_id" in tsv_sql


def test_both_sql_arms_filter_by_index_build_id_from_build_manifest(tmp_path: Path) -> None:
    """SP3 final review, IMPORTANT: `chunks` is one physical table that can carry more than one
    index build's rows (chunk_id is globally unique across builds, so `ON CONFLICT DO NOTHING`
    never needs a build scoped key -- see `rag_tools.ingest`'s own comment); without this filter, a
    rebuild that leaves an older build's rows in place would silently blend two builds' candidates
    into one ranking. The build id itself comes from `build_manifest.json`, read at construction
    alongside `fingerprint.json`."""
    hydrate_rows = [_hydrate_row("a", "doc-a", "text a")]
    retriever, conn = _make_retriever(
        tmp_path, vector_rows=[("a",)], tsv_rows=[("a",)], hydrate_rows=hydrate_rows, build_id="deadbeefcafef00d"
    )
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False)
    retriever.search_chunks("q", 1, config)

    calls = [call for call in conn.calls if call[0] == "execute"]
    vector_call = next(c for c in calls if "embedding <=>" in c[1])
    tsv_call = next(c for c in calls if "ts_rank" in c[1])

    assert "WHERE index_build_id = %(build_id)s" in vector_call[1]
    assert "WHERE index_build_id = %(build_id)s AND tsv @@" in tsv_call[1]
    assert vector_call[2]["build_id"] == "deadbeefcafef00d"
    assert tsv_call[2]["build_id"] == "deadbeefcafef00d"


def test_ef_search_is_set_local_when_not_exact_scan(tmp_path: Path) -> None:
    hydrate_rows = [_hydrate_row("a", "doc-a", "text a")]
    retriever, conn = _make_retriever(
        tmp_path, vector_rows=[("a",)], tsv_rows=[], hydrate_rows=hydrate_rows
    )
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False, ef_search=77, exact_scan=False)
    retriever.search_chunks("q", 1, config)

    executed_sql = [call[1] for call in conn.calls if call[0] == "execute"]
    set_local_calls = [sql for sql in executed_sql if "SET LOCAL" in sql]
    assert len(set_local_calls) == 1
    assert "hnsw.ef_search = 77" in set_local_calls[0]


def test_exact_scan_disables_indexscan_instead_of_setting_ef_search(tmp_path: Path) -> None:
    hydrate_rows = [_hydrate_row("a", "doc-a", "text a")]
    retriever, conn = _make_retriever(
        tmp_path, vector_rows=[("a",)], tsv_rows=[], hydrate_rows=hydrate_rows
    )
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False, exact_scan=True)
    retriever.search_chunks("q", 1, config)

    executed_sql = [call[1] for call in conn.calls if call[0] == "execute"]
    set_local_calls = [sql for sql in executed_sql if "SET LOCAL" in sql]
    assert len(set_local_calls) == 1
    assert "enable_indexscan = off" in set_local_calls[0]
    assert "hnsw.ef_search" not in set_local_calls[0]


def test_rerank_disabled_returns_fused_scores(tmp_path: Path) -> None:
    hydrate_rows = [_hydrate_row("a", "doc-a", "text a"), _hydrate_row("b", "doc-b", "text b")]
    retriever, _ = _make_retriever(
        tmp_path, vector_rows=[("a",), ("b",)], tsv_rows=[], hydrate_rows=hydrate_rows
    )
    config = RetrievalConfig(k_fused=5, k_final=2, rerank_enabled=False)
    results = retriever.search_chunks("q", 2, config)

    assert [c.doc_id for c in results] == ["doc-a", "doc-b"]
    assert results[0].score == pytest.approx(1 / 61)
    assert results[1].score == pytest.approx(1 / 62)


def test_rerank_enabled_uses_tei_scores_and_can_reorder(tmp_path: Path) -> None:
    hydrate_rows = [_hydrate_row("a", "doc-a", "text a"), _hydrate_row("b", "doc-b", "text b")]
    # fused order is a, b (a ranked first by both arms); rerank flips it: b scores higher.
    retriever, _ = _make_retriever(
        tmp_path,
        vector_rows=[("a",), ("b",)],
        tsv_rows=[("a",), ("b",)],
        hydrate_rows=hydrate_rows,
        rerank_scores={0: 0.2, 1: 0.9},
    )
    config = RetrievalConfig(k_fused=5, k_final=2, rerank_enabled=True)
    results = retriever.search_chunks("q", 2, config)

    assert [c.doc_id for c in results] == ["doc-b", "doc-a"]
    assert results[0].score == pytest.approx(0.9)
    assert results[1].score == pytest.approx(0.2)

    # the pre rerank fused scores stay available for tracing (SP6), keyed by chunk_id, via the
    # contextvar based `SearchResult` carrier (SP4 task 3), not the old `last_scores` global.
    result = pgvector_retriever.last_result()
    assert [c.doc_id for c in result.chunks] == ["doc-b", "doc-a"]

    fused = dict(result.fused_scores)
    assert set(fused) == {"a", "b"}
    assert fused["a"] > fused["b"]  # a was ranked first by both arms pre rerank

    rerank = dict(result.rerank_scores)
    assert rerank == {"a": 0.2, "b": 0.9}


def test_rerank_batches_above_tei_max_client_batch_size(tmp_path: Path) -> None:
    # TEI 422s a single `/rerank` call above its `max_client_batch_size` (observed 32 live -- see
    # the live test report); this reproduces that ceiling hermetically and proves the adapter splits
    # into batches and remaps each batch's request local `index` back to the caller's full offsets.
    n = 40
    chunk_ids = [f"c{i}" for i in range(n)]
    vector_rows = [(cid,) for cid in chunk_ids]
    hydrate_rows = [_hydrate_row(cid, f"doc-{cid}", f"text {cid}") for cid in chunk_ids]

    index_dir = _write_fingerprint(tmp_path)
    conn = _FakeConnection(vector_rows, [], hydrate_rows)
    embed_client = _tei_client(info_handler=_info_handler(), embed_handler=_embed_handler([0.1, 0.2, 0.3]))

    requests_seen: list[list[str]] = []

    def rerank_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        texts = body["texts"]
        if len(texts) > 32:
            return httpx.Response(422, json={"error": f"batch size {len(texts)} > maximum allowed batch size 32"})
        requests_seen.append(texts)
        # score every text in this batch as 1.0 minus its request local index, so batch 2's item 0
        # (global index 32) scores higher than batch 1's item 0 (global index 0) -- proves the
        # remap uses each batch's OWN start offset, not always the first batch's.
        return httpx.Response(200, json=[{"index": i, "score": 1.0 - i / 100} for i in range(len(texts))])

    rerank_client = httpx.Client(transport=httpx.MockTransport(rerank_handler), base_url="http://tei.test")

    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=embed_client,
        rerank_client=rerank_client,
        connect=lambda: conn,
    )
    config = RetrievalConfig(k_fused=n, k_final=n, rerank_enabled=True)
    results = retriever.search_chunks("q", n, config)

    assert len(requests_seen) == 2  # 40 texts, batch size 32 -> two requests (32 + 8), never one 40
    assert len(requests_seen[0]) == 32
    assert len(requests_seen[1]) == 8

    # every text scored `1.0 - local_index/100` within ITS OWN request; a correct remap means both
    # c0 (batch 1's local index 0) and c32 (batch 2's local index 0) land on 1.0, and c1/c33 (each
    # batch's local index 1) both land on 0.99 -- a naive "always add batch-1's offset" bug would
    # instead leave c32..c39 at the placeholder 0.0, which this catches.
    assert len(results) == n
    scores_by_chunk = {c.chunk_id: c.score for c in results}
    assert scores_by_chunk["c0"] == pytest.approx(1.0)
    assert scores_by_chunk["c32"] == pytest.approx(1.0)
    assert scores_by_chunk["c1"] == pytest.approx(0.99)
    assert scores_by_chunk["c33"] == pytest.approx(0.99)
    assert all(c.score > 0.0 for c in results)  # no chunk silently kept the 0.0 placeholder


def test_k_caps_the_returned_list_below_k_final(tmp_path: Path) -> None:
    hydrate_rows = [
        _hydrate_row("a", "doc-a", "text a"),
        _hydrate_row("b", "doc-b", "text b"),
        _hydrate_row("c", "doc-c", "text c"),
    ]
    retriever, _ = _make_retriever(
        tmp_path, vector_rows=[("a",), ("b",), ("c",)], tsv_rows=[], hydrate_rows=hydrate_rows
    )
    config = RetrievalConfig(k_fused=5, k_final=3, rerank_enabled=False)
    results = retriever.search_chunks("q", 1, config)
    assert len(results) == 1
    assert results[0].doc_id == "doc-a"


def test_no_fused_hits_returns_empty_list_without_hydration_or_rerank(tmp_path: Path) -> None:
    retriever, conn = _make_retriever(tmp_path, vector_rows=[], tsv_rows=[], hydrate_rows=[])
    config = RetrievalConfig(k_fused=5, k_final=3, rerank_enabled=True)
    results = retriever.search_chunks("q", 3, config)
    assert results == []
    executed_sql = [call[1] for call in conn.calls if call[0] == "execute"]
    assert not any("ANY(" in sql for sql in executed_sql)  # no hydration query fired


# --- rerank batch size: a constructor kwarg, no construction time probe -------------------------


def test_rerank_batch_size_defaults_to_the_advertised_bge_reranker_ceiling(tmp_path: Path) -> None:
    index_dir = _write_fingerprint(tmp_path)
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=_tei_client(info_handler=_info_handler()),
        rerank_client=_tei_client(),
        connect=lambda: _FakeConnection([], [], []),
    )
    assert retriever._rerank_batch_size == 32


def test_rerank_batch_size_is_injectable_for_a_server_with_a_different_ceiling(tmp_path: Path) -> None:
    """The construction time `/info` probe this replaced fell back to the same 32 literal on any
    failure and could block boot for the rerank client's whole 120s timeout against a reachable but
    hanging server, all to learn a number that only ever costs a few extra requests when wrong. A
    kwarg is the seam `connect`, `clock` and both httpx clients already use, and the one caller that
    ever needed a different value (`matrix.live_search.TeiReranker`) has always taken it this way."""
    index_dir = _write_fingerprint(tmp_path)
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=_tei_client(info_handler=_info_handler()),
        rerank_client=_tei_client(),
        connect=lambda: _FakeConnection([], [], []),
        rerank_batch_size=7,
    )
    assert retriever._rerank_batch_size == 7


# --- SP4 task 3: typed errors, never a raw httpx/psycopg exception past this adapter -----------------


def test_embed_service_failure_raises_embedding_service_error_never_raw_httpx(tmp_path: Path) -> None:
    index_dir = _write_fingerprint(tmp_path)
    embed_client = _tei_client(info_handler=_info_handler(), embed_handler=_failing_handler(500, "/embed"))
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=embed_client,
        rerank_client=_tei_client(),
        connect=lambda: _FakeConnection([], [], []),
    )
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False)
    with pytest.raises(EmbeddingServiceError) as excinfo:
        retriever.search_chunks("q", 1, config)
    assert not isinstance(excinfo.value, httpx.HTTPStatusError)
    assert isinstance(excinfo.value.__cause__, httpx.HTTPStatusError)  # chained, not swallowed


def test_construction_time_tei_failure_also_raises_embedding_service_error(tmp_path: Path) -> None:
    # the fingerprint check's own /info call goes through the same resilience wrapping now.
    index_dir = _write_fingerprint(tmp_path)
    embed_client = _tei_client(info_handler=_failing_handler(503, "/info"))
    with pytest.raises(EmbeddingServiceError):
        PgvectorRetriever(
            pg_dsn="postgresql://unused/unused",
            index_dir=index_dir,
            embed_client=embed_client,
            rerank_client=_tei_client(),
            connect=lambda: _FakeConnection([], [], []),
        )


def test_rerank_service_failure_raises_rerank_service_error_never_raw_httpx(tmp_path: Path) -> None:
    hydrate_rows = [_hydrate_row("a", "doc-a", "text a")]
    index_dir = _write_fingerprint(tmp_path)
    conn = _FakeConnection([("a",)], [], hydrate_rows)
    embed_client = _tei_client(info_handler=_info_handler(), embed_handler=_embed_handler([0.1, 0.2, 0.3]))
    rerank_client = httpx.Client(
        transport=httpx.MockTransport(_failing_handler(503, "/rerank")), base_url="http://tei.test"
    )
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=embed_client,
        rerank_client=rerank_client,
        connect=lambda: conn,
    )
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=True)
    with pytest.raises(RerankServiceError) as excinfo:
        retriever.search_chunks("q", 1, config)
    assert not isinstance(excinfo.value, httpx.HTTPStatusError)


class _FailingConnection:
    """A connect boundary that fails before ever handing back a usable cursor: the closest fake
    shape to a real Postgres connection refused / dropped mid call."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.closed = False

    def cursor(self):
        raise self._exc

    def commit(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def test_postgres_failure_raises_retrieval_error_never_raw_psycopg(tmp_path: Path) -> None:
    index_dir = _write_fingerprint(tmp_path)
    embed_client = _tei_client(info_handler=_info_handler(), embed_handler=_embed_handler([0.1, 0.2, 0.3]))
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=embed_client,
        rerank_client=_tei_client(),
        connect=lambda: _FailingConnection(psycopg.OperationalError("connection refused")),
    )
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False)
    with pytest.raises(RetrievalError) as excinfo:
        retriever.search_chunks("q", 1, config)
    assert not isinstance(excinfo.value, psycopg.Error)
    assert isinstance(excinfo.value.__cause__, psycopg.Error)


def test_postgres_non_operational_error_is_never_retried_single_attempt(tmp_path: Path) -> None:
    index_dir = _write_fingerprint(tmp_path)
    embed_client = _tei_client(info_handler=_info_handler(), embed_handler=_embed_handler([0.1, 0.2, 0.3]))
    attempts = {"n": 0}

    class _CountingFailingConnection(_FailingConnection):
        def cursor(self):
            attempts["n"] += 1
            return super().cursor()

    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=embed_client,
        rerank_client=_tei_client(),
        connect=lambda: _CountingFailingConnection(psycopg.errors.UndefinedTable("no such table")),
    )
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False)
    with pytest.raises(RetrievalError):
        retriever.search_chunks("q", 1, config)
    assert attempts["n"] == 1  # never retried: a bad SQL statement is not a transient failure


# --- SP4 task 3: the breaker fail fast short circuit, exercised through the adapter ------------------


def test_repeated_embed_failures_trip_the_breaker_then_short_circuit_without_a_new_request(
    tmp_path: Path,
) -> None:
    index_dir = _write_fingerprint(tmp_path)
    request_count = {"n": 0}

    def dispatch(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            return httpx.Response(200, json={"model_id": MODEL_ID, "model_sha": REVISION, "version": "1.9.3"})
        request_count["n"] += 1
        return httpx.Response(500, json={"error": "tei embed down"})

    embed_client = httpx.Client(transport=httpx.MockTransport(dispatch), base_url="http://tei.test")
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=embed_client,
        rerank_client=_tei_client(),
        connect=lambda: _FakeConnection([], [], []),
    )
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False)

    # the default breaker failure threshold (`resilience._FAILURE_THRESHOLD`) is 3 call level
    # failures; each of these calls already exhausted its own retry attempts before failing.
    for _ in range(3):
        with pytest.raises(EmbeddingServiceError):
            retriever.search_chunks("q", 1, config)

    requests_before_open = request_count["n"]
    # the fail fast short circuit comes through as EmbeddingServiceError (the call site's own
    # error_type), not a bare ProviderError: an open embedding breaker must be distinguishable from
    # an open rerank breaker so Task 4's ladder routes lexical_only vs drop_rerank correctly.
    with pytest.raises(EmbeddingServiceError, match="circuit breaker open") as excinfo:
        retriever.search_chunks("q", 1, config)
    assert excinfo.value.provider_key == "tei-embed"
    assert request_count["n"] == requests_before_open  # zero new requests: a true fail fast short circuit


# --- SP4 task 4: lexical_only skips the vector arm and the embed call entirely ------------------------


def test_lexical_only_skips_the_vector_arm_and_the_embed_call_entirely(tmp_path: Path) -> None:
    hydrate_rows = [_hydrate_row("a", "doc-a", "text a")]
    index_dir = _write_fingerprint(tmp_path)
    conn = _FakeConnection([("should-never-be-read",)], [("a",)], hydrate_rows)
    embed_calls = {"n": 0}

    def dispatch(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            return _info_handler()(request)
        embed_calls["n"] += 1
        return httpx.Response(200, json=[[0.1, 0.2, 0.3]])

    embed_client = httpx.Client(transport=httpx.MockTransport(dispatch), base_url="http://tei.test")
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=embed_client,
        rerank_client=_tei_client(),
        connect=lambda: conn,
    )
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False, lexical_only=True)
    results = retriever.search_chunks("q", 1, config)

    assert embed_calls["n"] == 0  # the embed call never happened
    executed_sql = [call[1] for call in conn.calls if call[0] == "execute"]
    assert not any("embedding <=>" in sql for sql in executed_sql)  # the vector arm never ran
    assert not any("SET LOCAL" in sql for sql in executed_sql)  # ef_search/exact_scan never touched either
    assert [c.doc_id for c in results] == ["doc-a"]  # the tsv arm alone still answers


def test_lexical_only_still_filters_by_index_build_id_on_the_tsv_arm(tmp_path: Path) -> None:
    hydrate_rows = [_hydrate_row("a", "doc-a", "text a")]
    retriever, conn = _make_retriever(
        tmp_path, vector_rows=[], tsv_rows=[("a",)], hydrate_rows=hydrate_rows, build_id="deadbeefcafef00d"
    )
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False, lexical_only=True)
    retriever.search_chunks("q", 1, config)

    tsv_call = next(c for c in conn.calls if c[0] == "execute" and "ts_rank" in c[1])
    assert tsv_call[2]["build_id"] == "deadbeefcafef00d"


def test_lexical_only_true_and_rerank_enabled_true_is_the_real_production_pairing(tmp_path: Path) -> None:
    """SP4 task 5 ride along: the pairing `knowledge_server.py`'s embedding down fallback ACTUALLY
    constructs. `RetrievalConfig(lexical_only=True)` leaves `rerank_enabled` at its own default
    (`True`), unlike every OTHER test in this file's lexical_only block, which explicitly passes
    `rerank_enabled=False`. Reranking needs only text, never embeddings, so losing the embedder is no
    reason to also lose rerank quality on the tsv only fused candidate set -- this is the ONE test
    proving that end to end: the embed call never fires, the vector arm never runs, and TEI's
    `/rerank` still does, and its own scores (not the tsv fused order) decide the final ranking."""
    hydrate_rows = [_hydrate_row("a", "doc-a", "lexical hit a"), _hydrate_row("b", "doc-b", "lexical hit b")]
    index_dir = _write_fingerprint(tmp_path)
    conn = _FakeConnection([("should-never-be-read",)], [("a",), ("b",)], hydrate_rows)
    embed_calls = {"n": 0}

    def embed_dispatch(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            return _info_handler()(request)
        embed_calls["n"] += 1
        return httpx.Response(200, json=[[0.1, 0.2, 0.3]])

    embed_client = httpx.Client(transport=httpx.MockTransport(embed_dispatch), base_url="http://tei.test")
    rerank_calls: list[list[str]] = []

    def rerank_dispatch(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        rerank_calls.append(body["texts"])
        # the reranker's own scores flip the tsv fused order (a first, b second) so the final
        # order proves rerank actually ran, not just that it was skipped without error.
        return httpx.Response(200, json=[{"index": 0, "score": 0.3}, {"index": 1, "score": 0.9}])

    rerank_client = httpx.Client(transport=httpx.MockTransport(rerank_dispatch), base_url="http://tei.test")
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=embed_client,
        rerank_client=rerank_client,
        connect=lambda: conn,
    )
    # the ladder's own config, not a hand tuned variant: rerank_enabled defaults True
    config = RetrievalConfig(lexical_only=True)
    assert config.rerank_enabled is True
    results = retriever.search_chunks("q", 2, config)

    assert embed_calls["n"] == 0  # no embedder was called
    executed_sql = [call[1] for call in conn.calls if call[0] == "execute"]
    assert not any("embedding <=>" in sql for sql in executed_sql)  # the vector arm never ran
    assert len(rerank_calls) == 1  # /rerank WAS called, over the tsv only fused candidates
    assert set(rerank_calls[0]) == {"lexical hit a", "lexical hit b"}
    assert [c.doc_id for c in results] == ["doc-b", "doc-a"]  # the reranker's own order wins


# --- SP4 task 4: SearchResult.retried, the retry rung's own carrier -----------------------------------


def test_search_result_retried_is_false_when_nothing_needed_a_retry(tmp_path: Path) -> None:
    hydrate_rows = [_hydrate_row("a", "doc-a", "text a")]
    retriever, _ = _make_retriever(tmp_path, vector_rows=[("a",)], tsv_rows=[], hydrate_rows=hydrate_rows)
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False)
    retriever.search_chunks("q", 1, config)
    assert pgvector_retriever.last_result().retried is False


def test_search_result_retried_is_true_when_the_embed_call_needed_a_retry(tmp_path: Path) -> None:
    index_dir = _write_fingerprint(tmp_path)
    attempts = {"n": 0}

    def dispatch(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/info":
            return _info_handler()(request)
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503, json={"error": "tei embed flaky"})
        return httpx.Response(200, json=[[0.1, 0.2, 0.3]])

    embed_client = httpx.Client(transport=httpx.MockTransport(dispatch), base_url="http://tei.test")
    hydrate_rows = [_hydrate_row("a", "doc-a", "text a")]
    conn = _FakeConnection([("a",)], [], hydrate_rows)
    retriever = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=embed_client,
        rerank_client=_tei_client(),
        connect=lambda: conn,
    )
    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False)
    retriever.search_chunks("q", 1, config)
    assert pgvector_retriever.last_result().retried is True


# --- SP4 task 3: SearchResult contextvar isolation across interleaved calls --------------------------


def test_contextvar_isolation_across_interleaved_calls(tmp_path: Path) -> None:
    """Two `PgvectorRetriever` calls interleaved across two threads must never see each other's
    `last_result()`. Thread A pauses mid call (right after hydration, before it finalizes and sets
    its own result) while the main thread runs B's call to completion; only after the main thread
    has read back B's own result does A get released to finish and set its own. Both reads must
    stay correct despite B's call having fully completed while A's was still in flight."""
    index_dir = _write_fingerprint(tmp_path)

    a_paused = threading.Event()
    b_done = threading.Event()

    class _PausingCursor(_FakeCursor):
        def fetchall(self) -> list[tuple]:
            rows = super().fetchall()
            if "ANY(" in self._last_sql:
                a_paused.set()
                assert b_done.wait(timeout=5), "thread B never signalled completion"
            return rows

    class _PausingConnection(_FakeConnection):
        def cursor(self) -> _PausingCursor:
            return _PausingCursor(self.calls, self._vector_rows, self._tsv_rows, self._hydrate_rows)

    conn_a = _PausingConnection([("a",)], [], [_hydrate_row("a", "doc-a", "text a")])
    retriever_a = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=_tei_client(info_handler=_info_handler(), embed_handler=_embed_handler([0.1, 0.2, 0.3])),
        rerank_client=_tei_client(),
        connect=lambda: conn_a,
    )

    conn_b = _FakeConnection([("b",)], [], [_hydrate_row("b", "doc-b", "text b")])
    retriever_b = PgvectorRetriever(
        pg_dsn="postgresql://unused/unused",
        index_dir=index_dir,
        embed_client=_tei_client(info_handler=_info_handler(), embed_handler=_embed_handler([0.4, 0.5, 0.6])),
        rerank_client=_tei_client(),
        connect=lambda: conn_b,
    )

    config = RetrievalConfig(k_fused=5, k_final=1, rerank_enabled=False)
    thread_a_capture: list[tuple[list, object]] = []

    def run_a() -> None:
        chunks = retriever_a.search_chunks("q", 1, config)
        thread_a_capture.append((chunks, pgvector_retriever.last_result()))

    thread_a = threading.Thread(target=run_a)
    thread_a.start()
    assert a_paused.wait(timeout=5), "thread A never reached its pause point"

    # thread B runs to completion entirely on THIS (the main) thread while A is still paused mid call.
    chunks_b = retriever_b.search_chunks("q", 1, config)
    last_b = pgvector_retriever.last_result()
    b_done.set()
    thread_a.join(timeout=5)
    assert not thread_a.is_alive(), "thread A never finished"

    assert [c.doc_id for c in chunks_b] == ["doc-b"]
    assert last_b is not None
    assert [c.doc_id for c in last_b.chunks] == ["doc-b"]

    [(chunks_a, last_a)] = thread_a_capture
    assert [c.doc_id for c in chunks_a] == ["doc-a"]
    assert last_a is not None
    assert [c.doc_id for c in last_a.chunks] == ["doc-a"]  # never B's, despite B finishing first
