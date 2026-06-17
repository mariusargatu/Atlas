"""`PgKnowledgeGraph` (SP9 task 2): the `KnowledgeGraph` port over Postgres graph tables materialized
from the registry, the GOLD graph. Implements the identical Protocol `InMemoryGraph` (the hermetic CI
adapter) and `Neo4jKnowledgeGraph` already do -- `resolve`/`neighbors`/`paths`/`triples`, all
deterministic, all sorted -- so graph-RAG's metrics and the graph-RAG variant subgraph
(`orchestration/graph_rag.py`) run unchanged against whichever adapter is wired in. `paths` is a
genuine `WITH RECURSIVE` traversal over `graph_edges`, hop bounded, matching D1/D6's own words:
"registry materialized edge tables, 1 to 2 hop recursive CTE traversal."

Disposition (the two graph decision this repo makes explicit, not a silent swap): THIS adapter, over
`graph_nodes`/`graph_edges` populated from `corpus/registry/core.yaml` (`rag_tools.graph_ingest`,
harness side, mirroring `rag_tools.ingest`'s own DDL + load split -- schema and ETL live in the
harness, this adapter only queries, exactly like `pgvector_retriever.py` never creates the `chunks`
table it searches), is the GOLD graph: deterministically correct by construction, since it is derived
from the registry's own typed edges, never guessed or extracted. `Neo4jKnowledgeGraph`
(`neo4j_graph.py`) is REPOSITIONED, not deprecated: it is the LLM EXTRACTED comparison arm
(`neo4j-graphrag`'s `SimpleKGPipeline`, reading the rendered corpus text the way a real graph-RAG
deployment WITHOUT a typed registry would have to build its graph), scored against this gold graph
via `quality.graph_metrics` (`triple_prf`/`pairwise_prf`/`bcubed_prf`/`path_recall`,
`testing/harness/evals/graphrag/__main__.py`'s `task graph` study). Neither is dead weight: one is
the graph the graph-RAG variant actually answers from, the other is the measured "what if we had to
extract it instead of materializing it" baseline.

Hermetic/live split, mirroring `pgvector_retriever.py` exactly: `connect` is an injectable callable (a
recording fake in the hermetic tests, `lambda: psycopg.connect(dsn)` live), so this module's SQL
construction and row parsing are provable with no Docker; the live end to end path (a real Postgres,
the schema actually populated by `rag_tools.graph_ingest.load_graph`) is an operator lane concern,
matching this task's own hermetic test list (SQL construction and traversal correctness here, a real
database there, deferred).
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Optional

import psycopg

# The local development DSN default is named once in `atlas.config`; this module used to carry its
# own byte identical copy of the same literal, as did `pgvector_retriever.py`.
from atlas.config import DEFAULT_PG_DSN


# `unnest(aliases)` (rather than `ANY(aliases)` with `ILIKE`) keeps this an exact, case folded
# equality match, never a pattern match: a registry entity name containing `%` or `_` (neither
# appears on this corpus, but nothing guarantees it never will) would otherwise be silently
# interpreted as an ILIKE wildcard instead of a literal character. `ORDER BY id LIMIT 1` mirrors
# `Neo4jKnowledgeGraph.resolve`'s own tie break on an ambiguous mention (lowest id wins), so the two
# real adapters behind this one port never disagree on which node a mention resolves to.
_RESOLVE_SQL = """
    SELECT id FROM graph_nodes
    WHERE lower(name) = lower(%(mention)s)
       OR EXISTS (SELECT 1 FROM unnest(aliases) AS alias WHERE lower(alias) = lower(%(mention)s))
    ORDER BY id LIMIT 1;
"""

# DISTINCT so a multigraph (src->dst under more than one relation) yields dst once, matching
# `InMemoryGraph.neighbors`'/`Neo4jKnowledgeGraph.neighbors`'s own deduplication.
_NEIGHBORS_SQL = "SELECT DISTINCT dst FROM graph_edges WHERE src = %(id)s ORDER BY dst;"
_NEIGHBORS_REL_SQL = (
    "SELECT DISTINCT dst FROM graph_edges WHERE src = %(id)s AND relation = %(rel)s ORDER BY dst;"
)

_TRIPLES_SQL = "SELECT src, relation, dst FROM graph_edges;"

# The recursive CTE (D1/D6): `path` accumulates the node ids visited so far (`ARRAY[src, dst]` seeds
# it at hop 1), `NOT (e.dst = ANY(t.path))` is the simple path guard (never revisit a node, the same
# rule `InMemoryGraph.paths`'s `if nxt in path: continue` and `Neo4jKnowledgeGraph.paths`'s
# `single(m IN nodes(p) WHERE m = n)` both enforce), and `t.hops < %(max_hops)s` is the hop budget --
# a path of n nodes uses n-1 edges, so this bounds edges, matching both sibling adapters' own
# `max_hops` semantics exactly.
_PATHS_SQL = """
    WITH RECURSIVE traversal(dst, path, hops) AS (
        SELECT dst, ARRAY[src, dst], 1
        FROM graph_edges
        WHERE src = %(start)s
        UNION ALL
        SELECT e.dst, t.path || e.dst, t.hops + 1
        FROM traversal t
        JOIN graph_edges e ON e.src = t.dst
        WHERE t.hops < %(max_hops)s AND NOT (e.dst = ANY(t.path))
    )
    SELECT DISTINCT path FROM traversal WHERE dst = %(goal)s;
"""


class PgKnowledgeGraph:
    """`pg_dsn` defaults to `ATLAS_PG_DSN` (the same env var `PgvectorRetriever` reads: one Postgres,
    D1), falling back to the local compose DSN. `connect` is the same injectable callable seam
    `PgvectorRetriever` uses: a fresh connection is opened per call and always closed, never reused
    across calls (a connection that just failed is not trusted to still be good)."""

    def __init__(self, *, pg_dsn: str | None = None, connect: Callable[[], Any] | None = None) -> None:
        self._pg_dsn = pg_dsn or os.environ.get("ATLAS_PG_DSN", DEFAULT_PG_DSN)
        self._connect = connect or (lambda: psycopg.connect(self._pg_dsn))

    def resolve(self, mention: str) -> Optional[str]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(_RESOLVE_SQL, {"mention": mention})
                row = cur.fetchone()
            conn.commit()
            return row[0] if row else None
        finally:
            conn.close()

    def neighbors(self, node_id: str, rel: Optional[str] = None) -> tuple[str, ...]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                # `rel is not None` (not truthiness), matching `Neo4jKnowledgeGraph.neighbors`'s own
                # comment: an explicit `rel=""` filters like `InMemoryGraph`'s `e.rel == rel` (match
                # nothing), `None` is the only no filter sentinel.
                if rel is not None:
                    cur.execute(_NEIGHBORS_REL_SQL, {"id": node_id, "rel": rel})
                else:
                    cur.execute(_NEIGHBORS_SQL, {"id": node_id})
                rows = cur.fetchall()
            conn.commit()
            return tuple(row[0] for row in rows)
        finally:
            conn.close()

    def paths(self, start: str, goal: str, max_hops: int) -> tuple[tuple[str, ...], ...]:
        if max_hops < 1:
            # parity with both sibling adapters: a degenerate budget is a caller error, not a silent
            # empty result an eval would read as a legitimate "no path / abstain".
            raise ValueError(f"max_hops must be >= 1, got {max_hops}")
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(_PATHS_SQL, {"start": start, "goal": goal, "max_hops": max_hops})
                rows = cur.fetchall()
            conn.commit()
            paths = {tuple(row[0]) for row in rows}
            return tuple(sorted(paths))
        finally:
            conn.close()

    def triples(self) -> frozenset[tuple[str, str, str]]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(_TRIPLES_SQL)
                rows = cur.fetchall()
            conn.commit()
            return frozenset((row[0], row[1], row[2]) for row in rows)
        finally:
            conn.close()
