"""The graph RAG variant (SP9 task 2), hermetic: a narrow subgraph beside `atlas_graph.py` and
`agentic_rag.py`, never a branch inside either (D6). Exercises the graph end to end against a fixture
`InMemoryGraph` (the same registry-shaped "plan -> region -> fee" 2-hop chain
`test_pg_knowledge_graph.py` uses, over the CI knowledge-graph adapter this time) + a retriever double
+ `GatewayChatModel` in replay mode (keyless, no network): identical signature to naive (query in,
chunks plus an answer out), entity linking pulling in a chunk the query's own words never mention
(the graph's whole reason to exist), the never-silent fallback when nothing resolves, rerank
reordering, and the no-graph-supplied default degrading gracefully rather than crashing.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from replay.gateway import GatewayChatModel

from atlas.adapters.cassette_reranker import CassetteReranker
from atlas.adapters.inmemory_graph import InMemoryGraph
from atlas.orchestration.agentic_rag import build_generate_prompt
from atlas.orchestration.graph_rag import build_graph_rag_graph
from atlas.ports.knowledge import Chunk
from atlas.ports.knowledge_graph import Edge, Node

# The same registry-shaped chain named in this task's own spec: plan-fiber-100 -available_in->
# region-north -overrides_fee-> fee-equipment-rental. Real registry ids, a fixture graph shape.
_GRAPH_NODES = [
    Node("plan-fiber-100", "plan", "Fiber 100"),
    Node("region-north", "region", "North Region"),
    Node("fee-equipment-rental", "fee", "Equipment Rental Fee"),
]
_GRAPH_EDGES = [
    Edge("plan-fiber-100", "available_in", "region-north"),
    Edge("region-north", "overrides_fee", "fee-equipment-rental"),
]
_GRAPH = InMemoryGraph(_GRAPH_NODES, _GRAPH_EDGES)

_PLAN_CHUNK = Chunk(
    chunk_id="plan-1", parent_id="plan-1", doc_id="plan-1",
    text="The Fiber 100 plan is our mid tier fiber offering.", entity_ids=("plan-fiber-100",),
)
# Deliberately shares NO vocabulary with the query below: the only reason this chunk should ever be
# kept is that graph traversal reaches fee-equipment-rental two hops out from the resolved entity,
# never keyword overlap -- the exact "why graph RAG helps" case this variant exists to cover.
_FEE_CHUNK = Chunk(
    chunk_id="fee-1", parent_id="fee-1", doc_id="fee-1",
    text="A regional surcharge applies to hardware you keep on premises.", entity_ids=("fee-equipment-rental",),
)
_UNLINKED_CHUNK = Chunk(
    chunk_id="hours-1", parent_id="hours-1", doc_id="hours-1",
    text="Our support line is open from nine to five on weekdays.", entity_ids=(),
)


class _AllChunksRetriever:
    """A canned wide candidate pool, ignoring the query entirely: decouples this test from
    `InMemoryRetriever`'s own keyword-overlap heuristic, the same reason `test_agentic_rag.py` uses
    its own query-keyed fake retriever for its rewrite tests."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def search_chunks(self, query: str, k: int, config) -> list[Chunk]:
        return list(self._chunks)[:k]


def _graph_app(cassette_dir, *, retriever=None, reranker=None, graph=None):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=cassette_dir, mode="replay")
    return build_graph_rag_graph(gw, retriever=retriever, reranker=reranker, graph=graph)


# ---------------------------------------------------------------------------------------------
# identical signature to naive: query in, chunks + answer out
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_linking_pulls_in_a_chunk_the_querys_own_words_never_mention(tmp_path, seed_cassette):
    query = "does my Fiber 100 plan have any extra charges"
    retriever = _AllChunksRetriever([_PLAN_CHUNK, _FEE_CHUNK, _UNLINKED_CHUNK])
    kept = [_PLAN_CHUNK, _FEE_CHUNK]  # the unlinked chunk is dropped, the fee chunk is kept
    prompt = build_generate_prompt(query, kept, corrective=False)
    answer = "Yes, an equipment rental surcharge applies depending on your region."
    seed_cassette(tmp_path, [HumanMessage(prompt)], {"content": answer, "tool_calls": []})

    app = _graph_app(tmp_path, retriever=retriever, graph=_GRAPH)
    out = await app.ainvoke({"query": query})

    assert out["resolved_entity_ids"] == ("plan-fiber-100",)
    assert out["entity_closure"] == ("fee-equipment-rental", "plan-fiber-100", "region-north")
    assert [c.chunk_id for c in out["chunks"]] == ["plan-1", "fee-1"]
    assert out["answer"] == answer
    assert out["tools_called"] == (
        "resolve_entities", "traverse", "retrieve", "collect_chunks", "rerank", "generate",
    )


@pytest.mark.asyncio
async def test_no_entity_resolves_falls_back_to_the_full_candidate_pool_never_silent(tmp_path, seed_cassette):
    """A query naming no known entity at all resolves to an empty set; `collect_chunks` must fall
    back to the un-joined pool rather than answering from nothing (the same doctrine the retrieval
    degradation ladder and the agentic variant's own ungraded pass-through both already apply)."""
    query = "what are your support hours"
    retriever = _AllChunksRetriever([_UNLINKED_CHUNK])
    prompt = build_generate_prompt(query, [_UNLINKED_CHUNK], corrective=False)
    answer = "We're open nine to five on weekdays."
    seed_cassette(tmp_path, [HumanMessage(prompt)], {"content": answer, "tool_calls": []})

    app = _graph_app(tmp_path, retriever=retriever, graph=_GRAPH)
    out = await app.ainvoke({"query": query})

    assert out["resolved_entity_ids"] == ()
    assert out["entity_closure"] == ()
    assert [c.chunk_id for c in out["chunks"]] == ["hours-1"]  # kept, never dropped to nothing
    assert out["answer"] == answer


@pytest.mark.asyncio
async def test_a_graph_with_no_matching_entities_at_all_also_falls_back(tmp_path, seed_cassette):
    """The entity resolves against the graph, but every neighbour lookup dead-ends before reaching
    any chunk's own entity_ids: the join still empties out, and the fallback still engages."""
    lone_node_graph = InMemoryGraph([Node("plan-starter-50", "plan", "Starter 50")], [])
    query = "tell me about the Starter 50 plan"
    retriever = _AllChunksRetriever([_UNLINKED_CHUNK])
    prompt = build_generate_prompt(query, [_UNLINKED_CHUNK], corrective=False)
    answer = "We're open nine to five on weekdays."
    seed_cassette(tmp_path, [HumanMessage(prompt)], {"content": answer, "tool_calls": []})

    app = _graph_app(tmp_path, retriever=retriever, graph=lone_node_graph)
    out = await app.ainvoke({"query": query})

    assert out["resolved_entity_ids"] == ("plan-starter-50",)  # entity linking DID succeed
    assert [c.chunk_id for c in out["chunks"]] == ["hours-1"]  # but nothing in the pool attaches to it


@pytest.mark.asyncio
async def test_no_graph_supplied_defaults_to_an_empty_graph_and_still_answers(tmp_path, seed_cassette):
    """`graph=None` (the builder's own default) must never crash the turn: an empty `InMemoryGraph`
    resolves nothing, so `collect_chunks` falls back to the full pool, same as the no-match case."""
    query = "does my Fiber 100 plan have any extra charges"
    retriever = _AllChunksRetriever([_PLAN_CHUNK])
    prompt = build_generate_prompt(query, [_PLAN_CHUNK], corrective=False)
    answer = "Let me check that for you."
    seed_cassette(tmp_path, [HumanMessage(prompt)], {"content": answer, "tool_calls": []})

    app = _graph_app(tmp_path, retriever=retriever)  # graph omitted entirely
    out = await app.ainvoke({"query": query})

    assert out["resolved_entity_ids"] == ()
    assert [c.chunk_id for c in out["chunks"]] == ["plan-1"]
    assert out["answer"] == answer


# ---------------------------------------------------------------------------------------------
# traverse: bounded to 1-2 hops, never open ended
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_traverse_stops_at_two_hops_and_excludes_a_third_hop_node(tmp_path, seed_cassette):
    # a fixture chain one hop longer than `_GRAPH`'s own, so a bug that traversed 3 hops instead of
    # 2 would be caught: fee-equipment-rental has a third-hop neighbour the closure must NOT include.
    deep_graph = InMemoryGraph(
        _GRAPH_NODES + [Node("policy-fair-use", "policy", "Fair Use Policy")],
        _GRAPH_EDGES + [Edge("fee-equipment-rental", "governed_by", "policy-fair-use")],
    )
    query = "does my Fiber 100 plan have any extra charges"
    retriever = _AllChunksRetriever([_PLAN_CHUNK])
    prompt = build_generate_prompt(query, [_PLAN_CHUNK], corrective=False)
    answer = "Let me check that for you."
    seed_cassette(tmp_path, [HumanMessage(prompt)], {"content": answer, "tool_calls": []})

    app = _graph_app(tmp_path, retriever=retriever, graph=deep_graph)
    out = await app.ainvoke({"query": query})
    assert "policy-fair-use" not in out["entity_closure"]  # 3 hops out, past the bound
    assert out["entity_closure"] == ("fee-equipment-rental", "plan-fiber-100", "region-north")


# ---------------------------------------------------------------------------------------------
# rerank: the SAME reranker port the naive and agentic variants both call
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_node_uses_the_reranker_port_reordering_takes_effect(tmp_path, seed_cassette):
    reranker = CassetteReranker({"does my Fiber 100 plan have any extra charges": {"plan-1": 1.0, "fee-1": 5.0}})
    retriever = _AllChunksRetriever([_PLAN_CHUNK, _FEE_CHUNK, _UNLINKED_CHUNK])
    query = "does my Fiber 100 plan have any extra charges"
    reordered = [_FEE_CHUNK, _PLAN_CHUNK]  # fee-1 outscores plan-1 per the cassette table above
    prompt = build_generate_prompt(query, reordered, corrective=False)
    answer = "An equipment rental surcharge may apply."
    seed_cassette(tmp_path, [HumanMessage(prompt)], {"content": answer, "tool_calls": []})

    app = _graph_app(tmp_path, retriever=retriever, reranker=reranker, graph=_GRAPH)
    out = await app.ainvoke({"query": query})
    assert [c.chunk_id for c in out["chunks"]] == ["fee-1", "plan-1"]
    assert out["answer"] == answer
