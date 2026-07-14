"""SP6 task 2: `atlas_graph.py`'s read loop opens the embed/retrieve/rerank/assemble stage spans
around every knowledge tool call and closes exactly the ones that ran, end to end through the REAL
graph. Driven with a real `OtelTracer` (in memory exporter, fake clock, no network): `InMemoryTracer`
(the CI adapter used everywhere else) records nothing on `close()` by design, so it cannot observe
WHICH stages closed -- only the OTel adapter turns that into an assertable export, per
`otel_tracer.py`'s own module docstring ("a never closed stage's absence from the export IS the
signal"). No Docker, no network: `OtelTracer` here is constructed exactly like `test_otel_tracer.py`
does, with an injected `InMemorySpanExporter`.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from atlas.adapters.otel_tracer import OtelTracer
from atlas.adapters.resilience import EmbeddingServiceError, RerankServiceError, RetrievalError
from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph
from atlas.ports.knowledge import Chunk
from determinism.canonical import serialize_tool_result
from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from replay.gateway import GatewayChatModel

_SESSION = {"customer_id": "cust_current"}


class _FakeClock:
    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        self._value += 0.1
        return self._value


def _tracer():
    exporter = InMemorySpanExporter()
    tracer = OtelTracer(endpoint="http://example.invalid:4318", config_hash="h", exporter=exporter, clock=_FakeClock())
    return tracer, exporter


def _graph(tmp_path, tracer, *, retriever=None):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
    backend = ActionsBackend(IdFactory("ref"))
    return build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer(), retriever=retriever, tracer=tracer)


def _stage_span_names(exporter) -> set[str]:
    return {s.name for s in exporter.get_finished_spans() if s.attributes.get("openinference.span.kind") in
            {"RETRIEVER", "RERANKER"} or s.name in {"embed", "retrieve", "rerank", "assemble"}}


class _RerankDownRetriever:
    def search_chunks(self, query, k, config):
        if config.rerank_enabled:
            raise RerankServiceError("tei-rerank down", provider_key="tei-rerank")
        return [Chunk(doc_id="doc-1", chunk_id="chunk-1", score=0.5, text="rerank-off answer")][:k]


class _EmbeddingDownRetriever:
    def search_chunks(self, query, k, config):
        if not config.lexical_only:
            raise EmbeddingServiceError("tei-embed down", provider_key="tei-embed")
        return [Chunk(doc_id="doc-1", chunk_id="chunk-1", score=0.5, text="lexical-only answer")][:k]


class _AlwaysDownRetriever:
    def search_chunks(self, query, k, config):
        raise RetrievalError("postgres down", provider_key="postgres")


@pytest.mark.asyncio
async def test_happy_path_closes_all_four_stage_spans(tmp_path, seed_cassette):
    query = "what is a data cap"
    user = HumanMessage("What is a data cap?")
    toolcall = [{"name": "search_knowledge", "args": {"query": query}, "id": "k1"}]
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": toolcall})
    from atlas.adapters.inmemory_retriever import InMemoryRetriever
    from atlas.domain.retrieval import RetrievalConfig

    chunks = InMemoryRetriever().search_chunks(query, config=RetrievalConfig())
    passages = serialize_tool_result(
        [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": c.score, "text": c.text} for c in chunks]
    )
    ai = AIMessage(content="", tool_calls=toolcall)
    tool_msg = ToolMessage(content=passages, tool_call_id="k1", name="search_knowledge")
    seed_cassette(tmp_path, [user, ai, tool_msg], {"content": "A data cap is a monthly limit.", "tool_calls": []})

    tracer, exporter = _tracer()
    graph = _graph(tmp_path, tracer)
    await graph.ainvoke({"messages": [user], "session": _SESSION}, {"configurable": {"thread_id": "stage-happy"}})

    assert _stage_span_names(exporter) == {"embed", "retrieve", "rerank", "assemble"}
    embed_span = next(s for s in exporter.get_finished_spans() if s.name == "embed")
    assert embed_span.attributes["atlas.stage.embed_ms"] > 0


@pytest.mark.asyncio
async def test_drop_rerank_closes_everything_but_rerank(tmp_path, seed_cassette):
    query = "is my plan contract free"
    user = HumanMessage("Is my plan contract-free?")
    toolcall = [{"name": "search_knowledge", "args": {"query": query}, "id": "k1"}]
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": toolcall})
    degraded = serialize_tool_result([{"doc_id": "doc-1", "chunk_id": "chunk-1", "score": 0.5, "text": "rerank-off answer"}])
    ai = AIMessage(content="", tool_calls=toolcall)
    tool_msg = ToolMessage(content=degraded, tool_call_id="k1", name="search_knowledge")
    seed_cassette(tmp_path, [user, ai, tool_msg], {"content": "Here is what I found.", "tool_calls": []})

    tracer, exporter = _tracer()
    graph = _graph(tmp_path, tracer, retriever=_RerankDownRetriever())
    await graph.ainvoke({"messages": [user], "session": _SESSION}, {"configurable": {"thread_id": "stage-droprerank"}})

    names = _stage_span_names(exporter)
    assert names == {"embed", "retrieve", "assemble"}
    assert "rerank" not in names  # matches degraded_turn.json: no atlas.stage.rerank_ms this turn


@pytest.mark.asyncio
async def test_lexical_only_closes_everything_but_embed(tmp_path, seed_cassette):
    query = "is my plan contract free"
    user = HumanMessage("Is my plan contract-free?")
    toolcall = [{"name": "search_knowledge", "args": {"query": query}, "id": "k1"}]
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": toolcall})
    degraded = serialize_tool_result([{"doc_id": "doc-1", "chunk_id": "chunk-1", "score": 0.5, "text": "lexical-only answer"}])
    ai = AIMessage(content="", tool_calls=toolcall)
    tool_msg = ToolMessage(content=degraded, tool_call_id="k1", name="search_knowledge")
    seed_cassette(tmp_path, [user, ai, tool_msg], {"content": "Here is what I found.", "tool_calls": []})

    tracer, exporter = _tracer()
    graph = _graph(tmp_path, tracer, retriever=_EmbeddingDownRetriever())
    await graph.ainvoke({"messages": [user], "session": _SESSION}, {"configurable": {"thread_id": "stage-lexical"}})

    names = _stage_span_names(exporter)
    assert names == {"retrieve", "rerank", "assemble"}
    assert "embed" not in names


@pytest.mark.asyncio
async def test_refusal_closes_embed_retrieve_rerank_but_never_opens_assemble(tmp_path, seed_cassette):
    query = "is my plan contract free"
    user = HumanMessage("Is my plan contract-free?")
    toolcall = [{"name": "search_knowledge", "args": {"query": query}, "id": "k1"}]
    seed_cassette(tmp_path, [user], {"content": "", "tool_calls": toolcall})

    tracer, exporter = _tracer()
    graph = _graph(tmp_path, tracer, retriever=_AlwaysDownRetriever())
    await graph.ainvoke({"messages": [user], "session": _SESSION}, {"configurable": {"thread_id": "stage-refusal"}})

    names = _stage_span_names(exporter)
    assert names == {"embed", "retrieve", "rerank"}
    assert "assemble" not in names  # assemble only opens once a real result is known to exist
