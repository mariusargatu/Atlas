"""Materialize the registry's real edges into Postgres graph tables (SP9 task 2): the GOLD graph
`atlas.adapters.pg_knowledge_graph.PgKnowledgeGraph` queries via recursive CTE. Mirrors `rag_tools.
ingest`'s own DDL + load split (schema and ETL live here, in the harness; the backend adapter only
queries, never creates or populates its own tables, exactly as `pgvector_retriever.py` never creates
the `chunks` table it searches).

`registry_to_graph` is the pure half (a `corpus_tools.registry.Registry` -> `Node`/`Edge` pairs, the
same frozen dataclasses `InMemoryGraph`/`Neo4jKnowledgeGraph` already take); `create_schema`/
`load_graph` are the two Postgres I/O calls, idempotent (`IF NOT EXISTS` / `ON CONFLICT DO NOTHING`)
so a rerun over an already-loaded database never duplicates rows or errors.
"""
from __future__ import annotations

from collections.abc import Iterable

from atlas.ports.knowledge_graph import Edge, Node
from corpus_tools.registry import Registry

_CREATE_NODES_SQL = """
    CREATE TABLE IF NOT EXISTS graph_nodes (
        id text PRIMARY KEY,
        type text NOT NULL,
        name text NOT NULL,
        aliases text[] NOT NULL DEFAULT '{}'
    );
"""

_CREATE_EDGES_SQL = """
    CREATE TABLE IF NOT EXISTS graph_edges (
        src text NOT NULL REFERENCES graph_nodes(id),
        relation text NOT NULL,
        dst text NOT NULL REFERENCES graph_nodes(id),
        PRIMARY KEY (src, relation, dst)
    );
"""

_CREATE_EDGES_SRC_INDEX_SQL = "CREATE INDEX IF NOT EXISTS graph_edges_src_idx ON graph_edges (src);"

_INSERT_NODE_SQL = (
    "INSERT INTO graph_nodes (id, type, name, aliases) VALUES (%s, %s, %s, %s) "
    "ON CONFLICT (id) DO NOTHING;"
)
_INSERT_EDGE_SQL = (
    "INSERT INTO graph_edges (src, relation, dst) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;"
)


def registry_to_graph(registry: Registry) -> tuple[list[Node], list[Edge]]:
    """The registry's own entities and edges (`corpus/registry/core.yaml`, D4's root artifact) as
    `KnowledgeGraph`-port primitives: every entity becomes a `Node` (`id`, `kind` as `type`, the
    entity's own declared `name` field, falling back to its id when a fields mapping omits `name`
    -- none of today's committed entities do, but nothing enforces it registry wide), every edge
    becomes an `Edge` verbatim (`relation`/`src`/`dst`; the registry's own optional per-edge
    `fields`, e.g. `overrides_fee`'s `override_amount`, carry no analogue on the graph port's `Edge`
    and are intentionally dropped here -- the graph variant traverses relationships, it does not
    re-derive fee amounts, which stay the render/verify pipeline's job). Aliases are always empty:
    the registry declares no alias field today, so entity linking (`extract_candidate_mentions` in
    `domain/graph_retrieval.py`) resolves only by the entity's own `name`, the same limitation
    `Neo4jKnowledgeGraph`'s own LLM-extracted comparison arm is scored against, not silently padded
    out here to look better than it is."""
    nodes = [Node(id=e.id, type=e.kind, name=str(e.fields.get("name", e.id))) for e in registry.entities]
    edges = [Edge(src=e.src, rel=e.relation, dst=e.dst) for e in registry.edges]
    return nodes, edges


def create_schema(conn) -> None:
    """Idempotent (`IF NOT EXISTS` throughout): `graph_nodes` (id, kind-as-type, name, aliases) and
    `graph_edges` (src, relation, dst, a composite primary key so a rerun's re-insert of an unchanged
    edge is a no-op, never a duplicate row), plus a btree index over `src` (every `neighbors`/`paths`
    query in `pg_knowledge_graph.py` filters on it)."""
    with conn.cursor() as cur:
        cur.execute(_CREATE_NODES_SQL)
        cur.execute(_CREATE_EDGES_SQL)
        cur.execute(_CREATE_EDGES_SRC_INDEX_SQL)
    conn.commit()


def load_graph(conn, nodes: Iterable[Node], edges: Iterable[Edge]) -> tuple[int, int]:
    """Create the schema (idempotent), then load every node and edge. `ON CONFLICT DO NOTHING` on
    both tables means a rerun over an already-loaded registry reports the same rows without
    duplicating or erroring; returns the (nodes, edges) count this call attempted to insert (not
    necessarily the count actually new, mirroring `rag_tools.ingest.load_parquet`'s own
    attempted-vs-actually-loaded distinction being the caller's concern, not this function's)."""
    create_schema(conn)
    node_rows = [(n.id, n.type, n.name, list(n.aliases)) for n in nodes]
    edge_rows = [(e.src, e.rel, e.dst) for e in edges]
    with conn.cursor() as cur:
        cur.executemany(_INSERT_NODE_SQL, node_rows)
        cur.executemany(_INSERT_EDGE_SQL, edge_rows)
    conn.commit()
    return len(node_rows), len(edge_rows)
