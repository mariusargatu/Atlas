"""The `EmbeddingClient` port (SP9 task 3) and its two adapters, both hermetic.

`EmbeddingClient` is a narrow `Protocol` (`embed_texts(texts) -> vectors`, the second embedder axis
finally has somewhere to live behind): `TeiEmbeddingClient` wraps the existing TEI `/embed` call
(against `httpx.MockTransport`, the same hermetic pattern `test_ingest.py::test_embed_texts_*`
already establishes for the harness-side copy of this call) and `OpenAiEmbeddingClient` wraps the
`openai` SDK's `embeddings.create` shape (against a hand rolled stub object, keyless: `openai` is a
`record`-group-only dependency, not installed in the hermetic PR lane, so this adapter must never
import it at module scope -- only lazily, inside the branch that constructs a live client, which
these tests never reach).

NO record/replay mode on this port (the SP9 task 3 decision, see `ports/embedding.py`'s own module
docstring): embedding calls are cheap, and `EmbeddingFingerprint`/`index_build_id` already make an
index build content addressed and cached at the BUILD level, so "paid once" already holds there; the
full D19 embedding record/replay seam (mirroring the generator gateway's REPLAY/RECORD/LIVE modes)
is SP12's, not this task's.
"""
from __future__ import annotations

import json

import httpx
import pytest
from atlas.adapters.openai_embedding import OpenAiEmbeddingClient
from atlas.adapters.tei_embedding import TeiEmbeddingClient

# --- the port itself: structural, not runtime_checkable (matches Reranker/KnowledgeGraph's own -----
# --- convention: no isinstance check anywhere in this repo for a sibling Protocol) ------------------


def test_embedding_client_protocol_is_a_structural_shape_any_matching_object_satisfies() -> None:
    class _Fake:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(t))] for t in texts]

    from atlas.ports.embedding import EmbeddingClient

    client: EmbeddingClient = _Fake()  # a type-checker-only assertion; the real proof is the call below
    assert client.embed_texts(["a", "bb"]) == [[1.0], [2.0]]


# --- TeiEmbeddingClient: httpx.MockTransport, no real network --------------------------------------


def test_tei_embedding_client_batches_requests_and_preserves_order() -> None:
    seen_batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        batch = payload["inputs"]
        seen_batches.append(batch)
        return httpx.Response(200, json=[[float(len(text)), 0.0] for text in batch])

    http_client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei-embed.test")
    client = TeiEmbeddingClient("http://tei-embed.test", batch_size=2, client=http_client)
    texts = [f"text-{i}" for i in range(5)]

    vectors = client.embed_texts(texts)

    assert len(vectors) == 5
    assert [v[0] for v in vectors] == [float(len(t)) for t in texts]
    assert seen_batches == [["text-0", "text-1"], ["text-2", "text-3"], ["text-4"]]


def test_tei_embedding_client_fails_loud_with_no_retry_on_http_error() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": "backend not ready"})

    http_client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei-embed.test")
    client = TeiEmbeddingClient("http://tei-embed.test", batch_size=2, client=http_client)

    with pytest.raises(httpx.HTTPStatusError):
        client.embed_texts(["a", "b"])
    assert calls["n"] == 1  # exactly one attempt: no retry, matching rag_tools.ingest.embed_texts


def test_tei_embedding_client_handles_an_empty_batch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent for an empty text list")

    http_client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://tei-embed.test")
    client = TeiEmbeddingClient("http://tei-embed.test", client=http_client)

    assert client.embed_texts([]) == []


def test_tei_embedding_client_close_closes_an_owned_httpx_client() -> None:
    client = TeiEmbeddingClient("http://tei-embed.test")
    assert client._owns_client is True
    client.close()
    assert client._client.is_closed


def test_tei_embedding_client_close_leaves_an_injected_httpx_client_open() -> None:
    http_client = httpx.Client(base_url="http://tei-embed.test")
    client = TeiEmbeddingClient("http://tei-embed.test", client=http_client)
    client.close()
    assert http_client.is_closed is False
    http_client.close()


# --- OpenAiEmbeddingClient: a hand rolled stub, keyless (openai SDK not installed in the PR lane) ---


class _FakeEmbedding:
    def __init__(self, vector: list[float]) -> None:
        self.embedding = vector


class _FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_FakeEmbedding(v) for v in vectors]


class _FakeEmbeddingsResource:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, *, model: str, input: list[str]) -> _FakeEmbeddingResponse:
        self.calls.append({"model": model, "input": list(input)})
        return _FakeEmbeddingResponse([[float(len(text)), 0.0] for text in input])


class _FakeOpenAiClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsResource()


def test_openai_embedding_client_calls_embeddings_create_with_the_pinned_model_id() -> None:
    fake = _FakeOpenAiClient()
    client = OpenAiEmbeddingClient("text-embedding-3-small", client=fake)

    vectors = client.embed_texts(["hello", "goodbye"])

    assert vectors == [[5.0, 0.0], [7.0, 0.0]]
    assert fake.embeddings.calls == [{"model": "text-embedding-3-small", "input": ["hello", "goodbye"]}]


def test_openai_embedding_client_preserves_input_order() -> None:
    fake = _FakeOpenAiClient()
    client = OpenAiEmbeddingClient("text-embedding-3-small", client=fake)

    vectors = client.embed_texts(["a", "abc", "ab"])

    assert [v[0] for v in vectors] == [1.0, 3.0, 2.0]


def test_openai_embedding_client_handles_an_empty_batch() -> None:
    fake = _FakeOpenAiClient()
    client = OpenAiEmbeddingClient("text-embedding-3-small", client=fake)

    assert client.embed_texts([]) == []
    assert fake.embeddings.calls == [{"model": "text-embedding-3-small", "input": []}]


def test_openai_embedding_client_never_imports_openai_at_module_scope() -> None:
    # The `record` dependency group (which carries the real `openai` package) is not installed in
    # the hermetic PR lane; a module-scope `import openai` would make collecting this test file
    # itself fail closed everywhere except a `record`-group environment. Guarded directly: the
    # module's own compiled globals must name no top-level `openai` import.
    import ast
    import inspect

    import atlas.adapters.openai_embedding as mod

    tree = ast.parse(inspect.getsource(mod))
    top_level_imports = [n for n in ast.iter_child_nodes(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {alias.name for node in top_level_imports for alias in node.names}
    assert "openai" not in names
