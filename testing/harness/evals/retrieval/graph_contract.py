"""One contract, three adapters (SP9 task 2 repositioned this from two: `PgKnowledgeGraph`, the
registry-materialized GOLD graph, joined `InMemoryGraph` and `Neo4jKnowledgeGraph` behind this same
port). The behaviours a ``KnowledgeGraph`` must honour, asserted against whichever implementation is
passed in: `InMemoryGraph` and `PgKnowledgeGraph` (over a recording fake connection,
`test_pg_knowledge_graph.py`) both run through this in the hermetic gate; the real Neo4j adapter runs
it in the operator lane only (a live server, never hermetic). The shared check is what keeps the
fakes from drifting from the real adapters on the behaviours all three can express (the live lane
additionally tolerates recall < 1.0 and reports distributions; the gate asserts exact ids).
"""
from __future__ import annotations


def check_knowledge_graph_contract(
    graph,
    *,
    start: str,
    goal: str,
    max_hops: int,
    expected_paths: frozenset,
    a_mention: str,
    its_id: str,
) -> None:
    # resolution: a known surface form lands on its canonical node, an unknown one is None
    assert graph.resolve(a_mention) == its_id
    assert graph.resolve("a mention that is not in the graph at all") is None

    # neighbours are sorted (no iteration-order leak) and every edge target is a real node id
    triples = graph.triples()
    node_ids = {t[0] for t in triples} | {t[2] for t in triples}
    neigh = graph.neighbors(start)
    assert list(neigh) == sorted(neigh)
    assert all(n in node_ids for n in neigh)

    # traversal is deterministic, sorted, and returns exactly the expected reasoning paths
    paths = graph.paths(start, goal, max_hops)
    assert list(paths) == sorted(paths)
    assert frozenset(paths) == expected_paths

    # triples() is a set of 3-tuples
    assert isinstance(triples, frozenset)
    assert all(isinstance(t, tuple) and len(t) == 3 for t in triples)
