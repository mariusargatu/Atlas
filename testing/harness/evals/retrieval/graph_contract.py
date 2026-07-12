"""One contract, two adapters. The behaviours a ``KnowledgeGraph`` must honour, asserted against
whichever implementation is passed in: the in-memory graph in the hermetic gate, and the real Neo4j
adapter in the operator lane. The shared check is what keeps the fake from drifting from the real one
on the behaviours both can express (the real lane additionally tolerates recall < 1.0 and reports
distributions; the gate asserts exact ids).
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
