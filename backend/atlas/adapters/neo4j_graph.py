"""``Neo4jKnowledgeGraph``: the graph port over a LIVE Neo4j, the operator/dev-prod adapter (deferred).

Implements the same ``KnowledgeGraph`` protocol as ``InMemoryGraph`` via Cypher, so the graph-RAG
metrics run unchanged against a real database; the shared contract suite (``evals.retrieval
.graph_contract``) exercises both. There is no in-process Cypher from Python, so this needs a running
server, ``docker compose --profile operator up neo4j`` or a ``testcontainers`` Neo4j, which is exactly
why it is NOT in the hermetic gate and is coverage-omitted infra. The ``neo4j`` driver (and
``neo4j-graphrag`` for vector/hybrid retrieval, separate from this traversal port) are imported lazily
so the pure lane never needs them.
"""
from __future__ import annotations

from typing import Optional


class Neo4jKnowledgeGraph:
    def __init__(self, uri: str, auth: tuple[str, str], database: str = "neo4j") -> None:
        from neo4j import GraphDatabase  # lazy: graph group only

        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._database = database

    def close(self) -> None:
        self._driver.close()

    def resolve(self, mention: str) -> Optional[str]:
        query = (
            "MATCH (n) WHERE toLower(n.name) = toLower($m) "
            "OR any(a IN coalesce(n.aliases, []) WHERE toLower(a) = toLower($m)) "
            "RETURN n.id AS id ORDER BY n.id LIMIT 1"  # stable tie-break on an ambiguous mention
        )
        with self._driver.session(database=self._database) as session:
            record = session.run(query, m=mention).single()
            return record["id"] if record else None

    def neighbors(self, node_id: str, rel: Optional[str] = None) -> tuple[str, ...]:
        # `rel is not None` (not truthiness) so an explicit rel="" filters like InMemoryGraph's
        # `e.rel == rel` (match nothing) rather than being read as "no filter" — None is the only
        # no-filter sentinel, keeping the two adapters substitutable on the relation filter too.
        clause = "WHERE type(r) = $rel " if rel is not None else ""
        # DISTINCT so parallel edges (a->b under >1 relation) yield b once, matching InMemoryGraph.
        query = f"MATCH (a {{id: $id}})-[r]->(b) {clause}RETURN DISTINCT b.id AS id ORDER BY b.id"
        params = {"id": node_id, **({"rel": rel} if rel is not None else {})}
        with self._driver.session(database=self._database) as session:
            return tuple(record["id"] for record in session.run(query, **params))

    def paths(self, start: str, goal: str, max_hops: int) -> tuple[tuple[str, ...], ...]:
        if max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {max_hops}")
        # apoc-free simple paths; the hop bound is an int we validated, so the f-string is safe.
        query = (
            f"MATCH p = (a {{id: $start}})-[*1..{int(max_hops)}]->(b {{id: $goal}}) "
            "WHERE all(n IN nodes(p) WHERE single(m IN nodes(p) WHERE m = n)) "  # simple paths only
            "RETURN [n IN nodes(p) | n.id] AS ids"
        )
        with self._driver.session(database=self._database) as session:
            paths = {tuple(record["ids"]) for record in session.run(query, start=start, goal=goal)}
        return tuple(sorted(paths))

    def triples(self) -> frozenset[tuple[str, str, str]]:
        query = "MATCH (a)-[r]->(b) RETURN a.id AS s, type(r) AS rel, b.id AS o"
        with self._driver.session(database=self._database) as session:
            return frozenset((r["s"], r["rel"], r["o"]) for r in session.run(query))
