"""A deterministic in-memory knowledge graph, the hermetic CI adapter behind the graph port.

Frozen ``Node``/``Edge`` primitives, a resolution index (surface form -> canonical id), and a
breadth-first simple-path traversal. Determinism is a contract, not an accident: neighbours are
sorted, path expansion is sorted, and the returned path set is sorted, so a run reproduces byte for
byte with no wall clock and no ``random`` (ADR-007). Because every method returns ids, the graph-RAG
metrics reduce to set arithmetic over its output. A real Neo4j adapter (deferred) implements the same
port; there is no in-process Cypher from Python, so this is the graph the gate actually traverses.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from atlas.ports.knowledge_graph import Edge, Node


class InMemoryGraph:
    def __init__(self, nodes: Iterable[Node], edges: Iterable[Edge]) -> None:
        self._nodes = {n.id: n for n in nodes}
        self._edges = tuple(edges)
        # surface form (lowercased) -> canonical node id, name and every alias. On an ambiguous
        # surface (two nodes share a name/alias) keep the LOWEST id, matching Neo4jKnowledgeGraph's
        # `ORDER BY n.id LIMIT 1` so the two adapters of this one port never disagree on which node a
        # mention resolves to (a last-writer-wins index would depend on node insertion order instead).
        self._index: dict[str, str] = {}
        for node in self._nodes.values():
            for surface in (node.name, *node.aliases):
                key = surface.lower()
                current = self._index.get(key)
                self._index[key] = node.id if current is None else min(current, node.id)

    def resolve(self, mention: str) -> Optional[str]:
        return self._index.get(mention.lower())

    def neighbors(self, node_id: str, rel: Optional[str] = None) -> tuple[str, ...]:
        # Distinct destination ids: a multigraph can hold parallel edges src->dst under different
        # relations, and a neighbour is a node, not an edge. De-duplicated (and sorted for
        # deterministic expansion, ADR-007) so paths() cannot emit the same path twice — the parity
        # Neo4jKnowledgeGraph.neighbors holds via RETURN DISTINCT.
        return tuple(sorted({
            e.dst for e in self._edges if e.src == node_id and (rel is None or e.rel == rel)
        }))

    def paths(self, start: str, goal: str, max_hops: int) -> tuple[tuple[str, ...], ...]:
        """Breadth-first over simple paths (no repeated node), expansion sorted for determinism.
        A path of n nodes uses n-1 edges; ``max_hops`` bounds edges, so a path is kept only when it
        reaches ``goal`` within that budget."""
        if max_hops < 1:
            # parity with Neo4jKnowledgeGraph.paths: a degenerate budget is a caller error, not a
            # silent empty result an eval would read as a legitimate "no path / abstain".
            raise ValueError(f"max_hops must be >= 1, got {max_hops}")
        found: list[tuple[str, ...]] = []
        frontier: list[tuple[str, ...]] = [(start,)]
        while frontier:
            path = frontier.pop(0)
            if len(path) - 1 >= max_hops:
                continue  # no edge budget left to extend this path
            for nxt in self.neighbors(path[-1]):
                if nxt in path:
                    continue  # simple path: never revisit a node
                extended = path + (nxt,)
                if nxt == goal:
                    found.append(extended)
                else:
                    frontier.append(extended)
        return tuple(sorted(found))

    def triples(self) -> frozenset[tuple[str, str, str]]:
        return frozenset((e.src, e.rel, e.dst) for e in self._edges)
