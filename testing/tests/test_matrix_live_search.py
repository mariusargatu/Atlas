"""`matrix.live_search`, hermetic: SQL construction, RRF fusion wiring, and the reranker's stable
sort, against a fake TEI transport and a recording fake psycopg connection. No Docker, no network,
no keys -- mirrors `test_pgvector_adapter.py`'s/`test_pg_knowledge_graph.py`'s own hermetic split:
the live end-to-end path (a real TEI server, a real Postgres actually populated) is an operator-lane
concern, covered by the live driver's own operator run, never by this file.
"""
from __future__ import annotations

import httpx
import pytest
from atlas.ports.knowledge import Chunk

from matrix.live_search import OpenAiEmbeddedRetriever, TeiReranker

# ---- TeiReranker ---------------------------------------------------------------------------------

_D1 = Chunk(chunk_id="d1", doc_id="doc-1", text="plan a costs 10")
_D2 = Chunk(chunk_id="d2", doc_id="doc-2", text="plan b costs 20")
_D3 = Chunk(chunk_id="d3", doc_id="doc-3", text="filler")


def _tei_rerank_transport(scores_by_text: dict[str, float]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rerank"
        body = httpx.Request("POST", request.url, content=request.read()).content
        import json as _json

        payload = _json.loads(body)
        texts = payload["texts"]
        results = [{"index": i, "score": scores_by_text.get(text, 0.0)} for i, text in enumerate(texts)]
        return httpx.Response(200, json=results)

    return httpx.MockTransport(handler)


def _reranker(scores_by_text: dict[str, float], *, batch_size: int = 32) -> TeiReranker:
    client = httpx.Client(transport=_tei_rerank_transport(scores_by_text), base_url="http://tei-rerank.test")
    return TeiReranker(client=client, batch_size=batch_size)


def test_rerank_reorders_by_descending_score():
    reranker = _reranker({"plan a costs 10": 0.1, "plan b costs 20": 0.9, "filler": 0.5})
    result = reranker.rerank("q", [_D1, _D2, _D3])
    assert [c.chunk_id for c in result] == ["d2", "d3", "d1"]


def test_rerank_ties_keep_the_callers_own_input_order():
    reranker = _reranker({"plan a costs 10": 0.5, "plan b costs 20": 0.5, "filler": 0.5})
    result = reranker.rerank("q", [_D1, _D2, _D3])
    assert [c.chunk_id for c in result] == ["d1", "d2", "d3"]


def test_rerank_of_an_empty_chunk_list_makes_no_request_and_returns_empty():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei-rerank.test")
    reranker = TeiReranker(client=client)
    assert reranker.rerank("q", []) == []
    assert calls["n"] == 0


def test_rerank_batches_requests_at_the_configured_batch_size():
    chunks = [Chunk(chunk_id=f"d{i}", doc_id=f"doc-{i}", text=f"text {i}") for i in range(5)]
    requests_seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        payload = _json.loads(request.read())
        requests_seen.append(len(payload["texts"]))
        results = [{"index": i, "score": float(i)} for i in range(len(payload["texts"]))]
        return httpx.Response(200, json=results)

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei-rerank.test")
    reranker = TeiReranker(client=client, batch_size=2)
    reranker.rerank("q", chunks)
    assert requests_seen == [2, 2, 1]


def test_rerank_raises_on_a_non_2xx_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei-rerank.test")
    reranker = TeiReranker(client=client)
    with pytest.raises(httpx.HTTPStatusError):
        reranker.rerank("q", [_D1])


def test_close_closes_only_an_owned_client_never_an_injected_one():
    injected = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
    reranker = TeiReranker(client=injected)
    reranker.close()
    assert not injected.is_closed  # an injected client is the caller's own to close


# ---- OpenAiEmbeddedRetriever ----------------------------------------------------------------------


class _StubEmbeddingClient:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.calls: list[list[str]] = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [self._vector for _ in texts]


class _FakeCursor:
    """Records every `execute` call and answers `fetchall()` from a small fixture table, mirroring
    `test_pg_knowledge_graph.py`'s own `_FakeCursor` idiom."""

    def __init__(self, *, vector_ids: list[str], tsv_ids: list[str], rows_by_id: dict[str, tuple]) -> None:
        self._vector_ids = vector_ids
        self._tsv_ids = tsv_ids
        self._rows_by_id = rows_by_id
        self.executed: list[tuple[str, dict]] = []
        self._last_sql = ""

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def execute(self, sql: str, params: dict | None = None) -> None:
        self._last_sql = sql
        self.executed.append((sql, params or {}))

    def fetchall(self):
        sql = self._last_sql
        if "ORDER BY embedding" in sql:
            return [(cid,) for cid in self._vector_ids]
        if "websearch_to_tsquery" in sql:
            return [(cid,) for cid in self._tsv_ids]
        if "chunk_id = ANY" in sql:
            _, params = self.executed[-1]
            return [self._rows_by_id[cid] for cid in params["chunk_ids"] if cid in self._rows_by_id]
        raise AssertionError(f"fetchall() called after an unexpected statement: {sql!r}")


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.committed = 0
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed += 1

    def close(self) -> None:
        self.closed = True


def _row(chunk_id: str, doc_id: str, text: str) -> tuple:
    return (chunk_id, doc_id, doc_id, "v1", "plan_page", [], 0, len(text), text, [])


def test_search_chunks_embeds_the_query_and_fuses_vector_plus_tsv_arms():
    rows = {
        "c1": _row("c1", "doc-1", "vector hit"),
        "c2": _row("c2", "doc-2", "both hit"),
        "c3": _row("c3", "doc-3", "tsv hit"),
    }
    cursor = _FakeCursor(vector_ids=["c1", "c2"], tsv_ids=["c2", "c3"], rows_by_id=rows)
    conn = _FakeConnection(cursor)
    embedding_client = _StubEmbeddingClient([1.0, 0.0])

    retriever = OpenAiEmbeddedRetriever(
        embedding_client=embedding_client, index_build_id="build-openai-1", connect=lambda: conn,
    )
    results = retriever.search_chunks("plan price", k=5)

    # c2 ranks first: it appears in BOTH arms, so RRF sums its reciprocal rank contribution twice.
    assert [c.chunk_id for c in results][0] == "c2"
    assert {c.chunk_id for c in results} == {"c1", "c2", "c3"}
    assert embedding_client.calls == [["plan price"]]
    assert conn.committed == 1
    assert conn.closed is True


def test_search_chunks_scopes_every_arm_to_the_given_index_build_id():
    cursor = _FakeCursor(vector_ids=[], tsv_ids=[], rows_by_id={})
    conn = _FakeConnection(cursor)
    retriever = OpenAiEmbeddedRetriever(
        embedding_client=_StubEmbeddingClient([1.0]), index_build_id="build-xyz", connect=lambda: conn,
    )
    retriever.search_chunks("q", k=3)
    build_ids = {params["build_id"] for _, params in cursor.executed if "build_id" in params}
    assert build_ids == {"build-xyz"}


def test_search_chunks_normalizes_the_query_vector_when_normalize_is_true():
    cursor = _FakeCursor(vector_ids=[], tsv_ids=[], rows_by_id={})
    conn = _FakeConnection(cursor)
    retriever = OpenAiEmbeddedRetriever(
        embedding_client=_StubEmbeddingClient([3.0, 4.0]), index_build_id="b", connect=lambda: conn, normalize=True,
    )
    retriever.search_chunks("q", k=3)
    vector_sql_call = next(params for sql, params in cursor.executed if "vector" in params)
    assert vector_sql_call["vector"] == "[0.6,0.8]"


def test_search_chunks_applies_the_query_prefix_before_embedding():
    cursor = _FakeCursor(vector_ids=[], tsv_ids=[], rows_by_id={})
    conn = _FakeConnection(cursor)
    embedding_client = _StubEmbeddingClient([1.0])
    retriever = OpenAiEmbeddedRetriever(
        embedding_client=embedding_client, index_build_id="b", connect=lambda: conn, query_prefix="query: ",
    )
    retriever.search_chunks("plan price", k=3)
    assert embedding_client.calls == [["query: plan price"]]


def test_search_chunks_returns_empty_when_neither_arm_matches_anything():
    cursor = _FakeCursor(vector_ids=[], tsv_ids=[], rows_by_id={})
    conn = _FakeConnection(cursor)
    retriever = OpenAiEmbeddedRetriever(
        embedding_client=_StubEmbeddingClient([1.0]), index_build_id="b", connect=lambda: conn,
    )
    assert retriever.search_chunks("nothing matches", k=5) == []
    assert conn.closed is True


def test_search_chunks_closes_the_connection_even_when_the_query_step_raises():
    class _BoomCursor(_FakeCursor):
        def execute(self, sql, params=None):
            raise RuntimeError("boom")

    conn = _FakeConnection(_BoomCursor(vector_ids=[], tsv_ids=[], rows_by_id={}))
    retriever = OpenAiEmbeddedRetriever(
        embedding_client=_StubEmbeddingClient([1.0]), index_build_id="b", connect=lambda: conn,
    )
    with pytest.raises(RuntimeError):
        retriever.search_chunks("q", k=1)
    assert conn.closed is True
