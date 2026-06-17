"""The graph RAG variant (SP9 task 2, D6): a NEW, narrow LangGraph subgraph beside `atlas_graph.py`
and `agentic_rag.py`, never a branch inside either. Query in, chunks plus an answer out (D6's
"identical signatures, selected by config"), same as its sibling variants.

Where the agentic variant widens the candidate pool by RETRYING vector retrieval, this variant widens
it by TRAVERSING: entity link the query against a `KnowledgeGraph` (`atlas.ports.knowledge_graph`),
walk 1 to 2 hops out from whatever resolved, then keep only the retrieved candidates that actually
attach to an entity the walk reached, before handing off to the SAME reranker and generator shape the
other variants use. Nodes, each a named step D14's deterministic trajectory checks can grade off
`tools_called` (this module's own analogue of `AgenticRagState.tools_called`):

  resolve_entities  -- `domain.graph_retrieval.extract_candidate_mentions` (pure tokenisation) tried,
                       one call each, against `KnowledgeGraph.resolve`. A candidate that resolves to
                       nothing is simply not an entity, never an error; a query that names no known
                       entity at all resolves to an empty set, which `collect_chunks` below treats as
                       "nothing to join against", not "abstain".
  traverse          -- breadth first over `KnowledgeGraph.neighbors`, bounded to `_MAX_HOPS` (1 to 2,
                       D1/D6), from every resolved seed. The bound is a fixed loop count, not a
                       `Budget` (unlike the agentic variant's rewrite retry): there is no possibility
                       of this step recursing past its own hop ceiling, so there is nothing for a
                       budget to police here.
  retrieve          -- `Retriever.search_chunks` (the exact port the naive path's `_knowledge_call`
                       already calls), widened (`rerank_enabled=False`) exactly like the agentic
                       variant's own `retrieve` node: a wide, unreranked candidate pool for
                       `collect_chunks` to filter, not yet the final answer set.
  collect_chunks    -- `domain.graph_retrieval.collect_chunks_by_entities`: keep only the retrieved
                       candidates whose `entity_ids` overlap the traversed entity closure. An empty
                       join (no entity resolved, or the entities resolved sit off every retrieved
                       chunk) falls back to the full, unjoined candidate pool rather than answering
                       from nothing -- the same "never silent, still answer" doctrine the retrieval
                       degradation ladder and the agentic variant's own pass through for an ungraded
                       query both already apply.
  rerank            -- `Reranker.rerank`, identical to the agentic variant's own `rerank` node (the
                       SAME port, the SAME truncation to `_K_FINAL`): the naive path's two stage
                       shape (fuse, then optionally rerank) reused verbatim a second time.
  generate          -- `agentic_rag.build_generate_prompt` (imported directly rather than derived
                       again: the SAME prompt construction shape every RAG variant in this repo uses),
                       `corrective=False` always -- this variant has no faithfulness loop (D6 does not
                       ask for one here; CRAG style regeneration is the agentic variant's own
                       contribution), so there is exactly one `generate` call per turn.

Touches NO `backend/atlas/domain` except reusing `atlas.domain.graph_retrieval` (this task's own new
pure module) and `atlas.domain.retrieval.RetrievalConfig` (the existing retrieval knob shape,
unmodified). `atlas.orchestration.agentic_rag.build_generate_prompt` is an orchestration to sibling
orchestration import (no hexagonal boundary crossed: both files sit in the same outer ring), reused
rather than derived a third time.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from atlas.adapters.cassette_reranker import CassetteReranker
from atlas.adapters.inmemory_graph import InMemoryGraph
from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.domain.graph_retrieval import collect_chunks_by_entities, extract_candidate_mentions
from atlas.domain.retrieval import K_FINAL, K_FUSED, RetrievalConfig
from atlas.orchestration.agentic_rag import build_generate_prompt
from atlas.ports.knowledge import Chunk, Retriever
from atlas.ports.knowledge_graph import KnowledgeGraph
from atlas.ports.reranker import Reranker

# Retrieval widths come from `domain.retrieval` (K_FUSED/K_FINAL), the one declaration every
# variant and `knowledge_server.DEPLOYED_K` read.

# 1 to 2 hop traversal (D1/D6): a fixed loop count, never open ended, so a graph with a cycle can
# never make this step run long -- `frontier`/`reached` below are sets, so a node reached twice (two
# different paths converging) is only ever expanded from once.
_MAX_HOPS = 2


class GraphRagState(TypedDict):
    query: str
    resolved_entity_ids: Optional[tuple[str, ...]]
    entity_closure: Optional[tuple[str, ...]]
    chunks: Optional[list[Chunk]]
    answer: Optional[str]
    tools_called: Optional[tuple[str, ...]]  # every node visited, in order: the D14 trajectory tally


def build_graph_rag_graph(
    model: BaseChatModel,
    retriever: Retriever | None = None,
    reranker: Reranker | None = None,
    graph: KnowledgeGraph | None = None,
    checkpointer=None,
):
    """`retriever` defaults to `InMemoryRetriever()`, `reranker` to `CassetteReranker({})` (a stable
    reorder with no effect), mirroring `build_agentic_rag_graph`'s own defaults. `graph` defaults to an EMPTY
    `InMemoryGraph((), ())`: with no nodes and no edges, `resolve_entities`/`traverse` reach nothing,
    and `collect_chunks` degrades gracefully to the unjoined candidate pool -- a caller that omits
    `graph` entirely still gets a working (if graph free) RAG pipeline, never a crash, the same
    "never silent, still answer" doctrine `collect_chunks` itself documents. `checkpointer` is
    optional: one query in, one answer out, no cross turn memory, so a caller that omits it gets an
    uncheckpointed compiled graph, fine for a single `ainvoke`."""
    retriever = retriever or InMemoryRetriever()
    reranker = reranker or CassetteReranker({})
    graph = graph or InMemoryGraph((), ())

    def resolve_entities(state: GraphRagState) -> dict:
        mentions = extract_candidate_mentions(state["query"])
        resolved = sorted({rid for m in mentions if (rid := graph.resolve(m)) is not None})
        return {
            "resolved_entity_ids": tuple(resolved),
            "tools_called": (state.get("tools_called") or ()) + ("resolve_entities",),
        }

    def traverse(state: GraphRagState) -> dict:
        seeds = frozenset(state.get("resolved_entity_ids") or ())
        reached: set[str] = set(seeds)
        frontier: set[str] = set(seeds)
        for _ in range(_MAX_HOPS):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for node_id in frontier:
                for neighbor in graph.neighbors(node_id):
                    if neighbor not in reached:
                        next_frontier.add(neighbor)
            reached |= next_frontier
            frontier = next_frontier
        return {
            "entity_closure": tuple(sorted(reached)),
            "tools_called": (state.get("tools_called") or ()) + ("traverse",),
        }

    def retrieve(state: GraphRagState) -> dict:
        chunks = retriever.search_chunks(
            state["query"], k=K_FUSED, config=RetrievalConfig(rerank_enabled=False, k_fused=K_FUSED, k_final=K_FUSED)
        )
        return {"chunks": chunks, "tools_called": (state.get("tools_called") or ()) + ("retrieve",)}

    def collect_chunks(state: GraphRagState) -> dict:
        entity_ids = frozenset(state.get("entity_closure") or ())
        joined = collect_chunks_by_entities(state["chunks"], entity_ids)
        # an empty join (no entity resolved, or every resolved entity sits off every retrieved
        # chunk) falls back to the full candidate pool rather than answering from nothing.
        chunks = joined if joined else state["chunks"]
        return {"chunks": chunks, "tools_called": (state.get("tools_called") or ()) + ("collect_chunks",)}

    def rerank(state: GraphRagState) -> dict:
        reranked = reranker.rerank(state["query"], state["chunks"])[:K_FINAL]
        return {"chunks": reranked, "tools_called": (state.get("tools_called") or ()) + ("rerank",)}

    async def generate(state: GraphRagState) -> dict:
        prompt = build_generate_prompt(state["query"], state["chunks"], corrective=False)
        result = await model._agenerate([HumanMessage(prompt)])
        message = result.generations[0].message
        text = message.content if isinstance(message.content, str) else str(message.content)
        return {"answer": text, "tools_called": (state.get("tools_called") or ()) + ("generate",)}

    g = StateGraph(GraphRagState)
    g.add_node("resolve_entities", resolve_entities)
    g.add_node("traverse", traverse)
    g.add_node("retrieve", retrieve)
    g.add_node("collect_chunks", collect_chunks)
    g.add_node("rerank", rerank)
    g.add_node("generate", generate)
    g.add_edge(START, "resolve_entities")
    g.add_edge("resolve_entities", "traverse")
    g.add_edge("traverse", "retrieve")
    g.add_edge("retrieve", "collect_chunks")
    g.add_edge("collect_chunks", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)
    return g.compile(checkpointer=checkpointer)
