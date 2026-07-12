"""`rag_tools.graph_ingest` (SP9 task 2), hermetic: `registry_to_graph`'s pure conversion, and the
DDL/insert SQL construction against a recording fake psycopg connection (the same convention
`test_ingest.py` already uses for `rag_tools.ingest.create_schema`/`load_parquet`). No Docker, no
network: the live end-to-end path (a real Postgres, actually populated) is an operator-lane concern.
"""
from __future__ import annotations

from pathlib import Path

from atlas.ports.knowledge_graph import Edge, Node
from corpus_tools.registry import Entity, Registry, load_registry
from rag_tools.graph_ingest import create_schema, load_graph, registry_to_graph

CORE_REGISTRY = Path("corpus/registry/core.yaml")


# --- registry_to_graph: pure conversion, no I/O -----------------------------------------------------


def _tiny_registry() -> Registry:
    entities = (
        Entity(id="plan-fiber-100", kind="plan", render=True, fields={"name": "Fiber 100"}),
        Entity(id="region-north", kind="region", render=True, fields={"name": "North Region"}),
        Entity(id="fee-no-name", kind="fee", render=True, fields={"amount": "10.00"}),  # no "name" field
    )
    from corpus_tools.registry import Edge as RegistryEdge

    edges = (RegistryEdge(relation="available_in", src="plan-fiber-100", dst="region-north"),)
    return Registry(entities=entities, edges=edges, contradictions=())


def test_registry_to_graph_converts_entities_to_nodes_and_edges_verbatim() -> None:
    nodes, edges = registry_to_graph(_tiny_registry())
    assert nodes == [
        Node(id="plan-fiber-100", type="plan", name="Fiber 100"),
        Node(id="region-north", type="region", name="North Region"),
        Node(id="fee-no-name", type="fee", name="fee-no-name"),  # falls back to its own id
    ]
    assert edges == [Edge(src="plan-fiber-100", rel="available_in", dst="region-north")]


def test_registry_to_graph_over_the_real_committed_registry_yields_19_edges_21_nodes() -> None:
    nodes, edges = registry_to_graph(load_registry([CORE_REGISTRY]))
    assert len(nodes) == 21
    assert len(edges) == 19
    assert Edge(src="region-north", rel="overrides_fee", dst="fee-equipment-rental") in edges


# --- create_schema / load_graph: DDL + insert SQL, against a recording fake connection --------------


class _RecordingCursor:
    def __init__(self, sink: list[tuple]) -> None:
        self._sink = sink

    def __enter__(self) -> "_RecordingCursor":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        self._sink.append(("execute", sql, params))

    def executemany(self, sql: str, params_seq: object) -> None:
        self._sink.append(("executemany", sql, list(params_seq)))


class _RecordingConnection:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self.calls)

    def commit(self) -> None:
        self.calls.append(("commit",))


def test_create_schema_emits_both_tables_and_the_src_index() -> None:
    conn = _RecordingConnection()
    create_schema(conn)
    executed_sql = [call[1] for call in conn.calls if call[0] == "execute"]
    assert any("CREATE TABLE IF NOT EXISTS graph_nodes" in sql for sql in executed_sql)
    edges_sql = next(sql for sql in executed_sql if "CREATE TABLE IF NOT EXISTS graph_edges" in sql)
    assert "REFERENCES graph_nodes(id)" in edges_sql
    assert "PRIMARY KEY (src, relation, dst)" in edges_sql
    assert any("graph_edges_src_idx" in sql for sql in executed_sql)
    assert conn.calls[-1] == ("commit",)


def test_load_graph_creates_schema_then_inserts_nodes_and_edges() -> None:
    conn = _RecordingConnection()
    nodes = [Node(id="plan-fiber-100", type="plan", name="Fiber 100")]
    edges = [Edge(src="plan-fiber-100", rel="available_in", dst="region-north")]

    node_count, edge_count = load_graph(conn, nodes, edges)

    assert (node_count, edge_count) == (1, 1)
    call_kinds = [call[0] for call in conn.calls]
    assert call_kinds.index("executemany") > call_kinds.index("commit")  # schema first, then load

    node_insert = next(call for call in conn.calls if call[0] == "executemany" and "graph_nodes" in call[1])
    assert "ON CONFLICT (id) DO NOTHING" in node_insert[1]
    assert node_insert[2] == [("plan-fiber-100", "plan", "Fiber 100", [])]

    edge_insert = next(call for call in conn.calls if call[0] == "executemany" and "graph_edges" in call[1])
    assert "ON CONFLICT DO NOTHING" in edge_insert[1]
    assert edge_insert[2] == [("plan-fiber-100", "available_in", "region-north")]

    assert conn.calls[-1] == ("commit",)
