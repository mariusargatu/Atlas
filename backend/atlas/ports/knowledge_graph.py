"""The knowledge-graph (traversal) port. Pure, no client, no framework.

Alongside the vector ``Retriever`` sits a graph of entities and the relationships between them:
plans, terms, add-ons, regional exceptions, and the edges that connect a customer's plan to the
clause that governs it. When retrieval runs over this graph it becomes traversal, and traversal
fails in ways a flat document set cannot, which is why it gets its own port and its own metrics.

The CI adapter is an in-memory graph over a hand-authored gold KG (deterministic, no DB). Two real
adapters sit behind this same port, each a distinct arm of SP9 task 2's two-graph disposition, named
explicitly rather than left to guesswork: ``PgKnowledgeGraph`` (a recursive-CTE traversal over
Postgres tables materialized straight from the registry's own typed edges) is the GOLD graph -- what
the graph-RAG variant (``orchestration.graph_rag``) actually traverses -- and ``Neo4jKnowledgeGraph``
(the ``neo4j-graphrag`` ``SimpleKGPipeline``) is the LLM-EXTRACTED comparison arm, scored against the
gold graph via ``quality.graph_metrics``. Neither is dead weight; there is no pure-Python embedded
Postgres or Neo4j, so the in-memory adapter is not a shortcut either, it is the only way to keep
graph RAG in the hermetic gate. Everything a metric grades is a stable id.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class Node:
    id: str            # the canonical entity id, the ground truth every metric compares against
    type: str          # the entity type, e.g. "Plan", "Term", "Customer"
    name: str          # the surface form
    aliases: tuple[str, ...] = ()  # other surface forms that must resolve to this same node


@dataclass(frozen=True)
class Edge:
    src: str           # Node.id
    rel: str           # the relation / predicate, e.g. "HAS_TERM"
    dst: str           # Node.id


class KnowledgeGraph(Protocol):
    def resolve(self, mention: str) -> Optional[str]:
        """The canonical node id for a surface form (name or alias), or ``None`` if unknown. This is
        entity resolution: the query's mention landing on the right node, not a similarly named one."""
        ...

    def neighbors(self, node_id: str, rel: Optional[str] = None) -> tuple[str, ...]:
        """The out-neighbours of a node (optionally filtered to one relation), returned SORTED so no
        dict/set iteration order leaks into a traversal."""
        ...

    def paths(self, start: str, goal: str, max_hops: int) -> tuple[tuple[str, ...], ...]:
        """All simple paths from ``start`` to ``goal`` within ``max_hops`` edges, each a tuple of node
        ids, the whole result SORTED. Empty when no path exists, the abstention signal a null query
        turns on."""
        ...

    def triples(self) -> frozenset[tuple[str, str, str]]:
        """The graph's ``(src, rel, dst)`` edge set, for relationship/triple-level scoring."""
        ...
