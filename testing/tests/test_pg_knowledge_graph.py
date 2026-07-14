"""`PgKnowledgeGraph` (SP9 task 2), hermetic: SQL construction and the recursive-CTE traversal's row
parsing, against a recording fake psycopg connection. No Docker, no network -- mirrors
`test_pgvector_adapter.py`'s own hermetic/live split exactly (the live end-to-end path, a real
Postgres actually populated, is an operator-lane concern, not part of this task's hermetic list).

The fake connection's `paths()` response is computed by an independent breadth-first search over the
SAME fixture edge set (`_bfs_paths`, written fresh here, never imported from `InMemoryGraph` or the
adapter under test), so a correct round trip proves two things at once: the adapter sends the right
SQL and params, and it parses whatever rows come back into the right sorted `tuple[tuple[str, ...],
...]` shape. The fixture edges are a VERBATIM excerpt of `corpus/registry/core.yaml` (checked against
the loaded registry below), not fabricated data: plan-fiber-100 is `available_in` region-north and
region-central, and region-north alone `overrides_fee` fee-equipment-rental -- the exact "plan ->
region -> fee" 2-hop chain this task's own spec names, with region-central as the genuine dead end
that proves the traversal does not hallucinate a path where none exists.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from atlas.adapters.pg_knowledge_graph import PgKnowledgeGraph
from corpus_tools.registry import load_registry
from evals.retrieval.graph_contract import check_knowledge_graph_contract

CORE_REGISTRY = Path("corpus/registry/core.yaml")

# A verbatim excerpt of corpus/registry/core.yaml's own entities/edges (see the module docstring and
# `test_fixture_edges_are_a_real_excerpt_of_the_committed_registry` below).
NODES: list[tuple[str, str, str, list[str]]] = [
    ("plan-fiber-100", "plan", "Fiber 100", []),
    ("plan-fiber-500", "plan", "Fiber 500", []),
    ("region-north", "region", "North Region", []),
    ("region-central", "region", "Central Region", []),
    ("fee-equipment-rental", "fee", "Equipment Rental Fee", []),
    ("fee-installation", "fee", "Installation Fee", []),
]
EDGES: list[tuple[str, str, str]] = [
    ("plan-fiber-100", "available_in", "region-north"),
    ("plan-fiber-100", "available_in", "region-central"),
    ("plan-fiber-500", "available_in", "region-north"),
    ("region-north", "overrides_fee", "fee-equipment-rental"),
]


def test_fixture_edges_are_a_real_excerpt_of_the_committed_registry() -> None:
    registry = load_registry([CORE_REGISTRY])
    real_edges = {(e.src, e.relation, e.dst) for e in registry.edges}
    assert set(EDGES) <= real_edges  # every fixture edge genuinely exists in the committed registry
    # and the dead end this module's docstring claims is real: region-central never overrides a fee.
    assert not any(src == "region-central" and rel == "overrides_fee" for src, rel, _ in real_edges)


def _bfs_paths(edges: list[tuple[str, str, str]], start: str, goal: str, max_hops: int) -> list[tuple[str, ...]]:
    """An independent breadth-first simple-path search, written fresh for this test file (never
    imported from `InMemoryGraph` or the adapter under test), standing in for what a real recursive
    CTE computes against the same edge set."""
    found: list[tuple[str, ...]] = []
    frontier: list[tuple[str, ...]] = [(start,)]
    while frontier:
        path = frontier.pop(0)
        if len(path) - 1 >= max_hops:
            continue
        for src, _rel, dst in edges:
            if src != path[-1] or dst in path:
                continue
            extended = path + (dst,)
            (found if dst == goal else frontier).append(extended)
    return found


class _FakeCursor:
    def __init__(self, nodes: list[tuple], edges: list[tuple]) -> None:
        self._nodes = nodes
        self._edges = edges
        self._last_sql = ""
        self._last_params: dict = {}

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def execute(self, sql: str, params: dict | None = None) -> None:
        self._last_sql = sql
        self._last_params = params or {}

    def fetchone(self):
        sql = self._last_sql
        if "unnest(aliases)" not in sql:
            raise AssertionError(f"fetchone() called after an unexpected statement: {sql!r}")
        mention = self._last_params["mention"].lower()
        matches = sorted(
            node_id
            for node_id, _type, name, aliases in self._nodes
            if name.lower() == mention or mention in [a.lower() for a in aliases]
        )
        return (matches[0],) if matches else None

    def fetchall(self):
        sql, params = self._last_sql, self._last_params
        if "WITH RECURSIVE" in sql:
            found = _bfs_paths(self._edges, params["start"], params["goal"], params["max_hops"])
            return [(list(path),) for path in found]
        if "DISTINCT dst" in sql:
            rel = params.get("rel")
            dsts = sorted({d for s, r, d in self._edges if s == params["id"] and (rel is None or r == rel)})
            return [(d,) for d in dsts]
        if sql.strip().startswith("SELECT src, relation, dst"):
            return list(self._edges)
        raise AssertionError(f"fetchall() called after an unexpected statement: {sql!r}")


class _FakeConnection:
    def __init__(self, nodes: list[tuple] = NODES, edges: list[tuple] = EDGES) -> None:
        self.closed = False
        self.commits = 0
        self._nodes = nodes
        self._edges = edges

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._nodes, self._edges)

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed = True


def _graph(**kwargs) -> PgKnowledgeGraph:
    conn = _FakeConnection(**kwargs)
    return PgKnowledgeGraph(pg_dsn="postgresql://unused/unused", connect=lambda: conn)


# --- resolve: case-insensitive name/alias match, lowest-id tie break, unknown -> None --------------


def test_resolve_matches_by_name_case_insensitively() -> None:
    graph = _graph()
    assert graph.resolve("Fiber 100") == "plan-fiber-100"
    assert graph.resolve("fiber 100") == "plan-fiber-100"
    assert graph.resolve("FIBER 100") == "plan-fiber-100"


def test_resolve_returns_none_for_an_unknown_mention() -> None:
    graph = _graph()
    assert graph.resolve("a mention that is not in the graph at all") is None


def test_resolve_matches_by_alias() -> None:
    nodes = [("plan-fiber-100", "plan", "Fiber 100", ["the fiber plan"])]
    graph = _graph(nodes=nodes, edges=[])
    assert graph.resolve("the fiber plan") == "plan-fiber-100"


def test_resolve_ties_break_on_the_lowest_id_like_the_sibling_adapters() -> None:
    # Two nodes share the exact same name: InMemoryGraph/Neo4jKnowledgeGraph both resolve an
    # ambiguous mention to the LOWEST id (ORDER BY id LIMIT 1), never insertion order.
    nodes = [
        ("plan-z-newer", "plan", "Ambiguous Plan", []),
        ("plan-a-older", "plan", "Ambiguous Plan", []),
    ]
    graph = _graph(nodes=nodes, edges=[])
    assert graph.resolve("Ambiguous Plan") == "plan-a-older"


def test_resolve_opens_and_closes_exactly_one_connection() -> None:
    conn = _FakeConnection()
    graph = PgKnowledgeGraph(pg_dsn="postgresql://unused/unused", connect=lambda: conn)
    graph.resolve("Fiber 100")
    assert conn.closed is True
    assert conn.commits == 1


# --- neighbors: sorted, distinct, optional relation filter -----------------------------------------


def test_neighbors_returns_sorted_distinct_destinations() -> None:
    graph = _graph()
    assert graph.neighbors("plan-fiber-100") == ("region-central", "region-north")


def test_neighbors_filters_by_relation() -> None:
    graph = _graph()
    assert graph.neighbors("region-north", rel="overrides_fee") == ("fee-equipment-rental",)
    assert graph.neighbors("region-north", rel="available_in") == ()  # no such outgoing edge


def test_neighbors_of_a_leaf_node_is_empty() -> None:
    graph = _graph()
    assert graph.neighbors("fee-equipment-rental") == ()


# --- paths: the recursive CTE, 1 and 2 hop, the named plan -> region -> fee chain -------------------


def test_one_hop_path_from_plan_to_region() -> None:
    graph = _graph()
    assert graph.paths("plan-fiber-100", "region-north", 1) == (("plan-fiber-100", "region-north"),)


def test_two_hop_path_plan_to_region_to_fee_is_the_only_path_found() -> None:
    """The task's own named chain: plan-fiber-100 -available_in-> region-north -overrides_fee->
    fee-equipment-rental. plan-fiber-100 is ALSO available_in region-central, a genuine dead end
    (region-central has no overrides_fee edge at all), so a correct 2-hop traversal must return
    EXACTLY the one real path, never a phantom second one through the dead end."""
    graph = _graph()
    paths = graph.paths("plan-fiber-100", "fee-equipment-rental", 2)
    assert paths == (("plan-fiber-100", "region-north", "fee-equipment-rental"),)


def test_a_one_hop_budget_cannot_reach_the_two_hop_fee() -> None:
    graph = _graph()
    assert graph.paths("plan-fiber-100", "fee-equipment-rental", 1) == ()


def test_no_path_exists_between_unconnected_nodes() -> None:
    graph = _graph()
    assert graph.paths("fee-installation", "region-north", 2) == ()


def test_paths_rejects_a_non_positive_max_hops() -> None:
    graph = _graph()
    with pytest.raises(ValueError, match="max_hops must be >= 1"):
        graph.paths("plan-fiber-100", "fee-equipment-rental", 0)


class _RecordingConnection(_FakeConnection):
    """Same fake, plus a `last_cursor` handle so a test can inspect exactly what SQL/params the one
    cursor `paths()` opens actually received, without needing a second live query round trip."""

    def __init__(self, nodes: list[tuple] = NODES, edges: list[tuple] = EDGES) -> None:
        super().__init__(nodes, edges)
        self.last_cursor: _FakeCursor | None = None

    def cursor(self) -> _FakeCursor:
        self.last_cursor = super().cursor()
        return self.last_cursor


def test_paths_sql_is_a_recursive_cte_bounded_by_hops_and_the_simple_path_guard() -> None:
    conn = _RecordingConnection()
    graph = PgKnowledgeGraph(pg_dsn="postgresql://unused/unused", connect=lambda: conn)
    graph.paths("plan-fiber-100", "fee-equipment-rental", 2)
    assert "WITH RECURSIVE" in conn.last_cursor._last_sql
    assert "NOT (e.dst = ANY(t.path))" in conn.last_cursor._last_sql  # the simple-path guard
    assert conn.last_cursor._last_params == {
        "start": "plan-fiber-100", "goal": "fee-equipment-rental", "max_hops": 2,
    }


# --- triples: the whole edge set, as a frozenset of 3-tuples ----------------------------------------


def test_triples_returns_the_whole_edge_set_as_a_frozenset() -> None:
    graph = _graph()
    assert graph.triples() == frozenset(EDGES)
    assert isinstance(graph.triples(), frozenset)


# --- the shared behavioural contract (M4, SP9 final review): PgKnowledgeGraph over the fake -------


def test_knowledge_graph_contract_holds_for_pg_knowledge_graph_over_the_fake() -> None:
    """`evals.retrieval.graph_contract.check_knowledge_graph_contract` was, until this fix, only
    ever run against `InMemoryGraph` (`test_graph_rag.py`); `PgKnowledgeGraph` was unit tested here
    but never contract-checked against the SAME shared suite the other two adapters behind this one
    port are held to. This runs it over the fake connection above (no Docker, no network), the exact
    "plan -> region -> fee" 2-hop chain this module's own fixture already proves the adapter's SQL
    and row parsing get right."""
    graph = _graph()
    check_knowledge_graph_contract(
        graph,
        start="plan-fiber-100",
        goal="fee-equipment-rental",
        max_hops=2,
        expected_paths=frozenset({("plan-fiber-100", "region-north", "fee-equipment-rental")}),
        a_mention="Fiber 100",
        its_id="plan-fiber-100",
    )
