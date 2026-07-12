"""Graph RAG, hermetic: when retrieval runs over a graph it stops being similarity search and
becomes traversal, which fails in ways a flat document set cannot. Three deterministic
checks over a hand-authored gold KG and an in-memory adapter, no Neo4j, no LLM: entity resolution
(did the query land on the right node), relationship/triple correctness (did the graph carry the
right edges), and multi-hop path recall (did traversal collect every hop). All reduce to set
arithmetic on stable ids, so a wrong extraction or a hop-short traversal fails an `assert`, not a
judge. Neo4j and judged answer quality live in the operator lane (Option A).
"""
from __future__ import annotations

import pytest

from atlas.adapters.inmemory_graph import InMemoryGraph
from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.domain.retrieval import RetrievalConfig
from atlas.ports.knowledge_graph import Edge, Node
from evals.datasets.graph_golden import (
    COLD_OPEN_CASE,
    GRAPH_EDGES,
    GRAPH_NODES,
    GOLD_ENTITY_CLUSTERS,
    GOLD_TRIPLES,
    POISON_EDGE,
    POISON_NODE,
    RELATIONAL_CORPUS,
    RELATIONAL_QUERY,
    RELATIONAL_RELEVANT,
)
from evals.retrieval.graph_contract import check_knowledge_graph_contract
from quality.graph_metrics import bcubed_prf, pairwise_prf, path_recall, triple_prf
from quality.ir_metrics import recall_at_k


@pytest.fixture
def graph() -> InMemoryGraph:
    return InMemoryGraph(GRAPH_NODES, GRAPH_EDGES)


def _predicted_clusters(graph: InMemoryGraph, mentions: list[str]) -> list[set[str]]:
    """Group the mentions by the canonical node each resolves to: the graph's entity resolution."""
    by_id: dict[str, set[str]] = {}
    for m in mentions:
        node_id = graph.resolve(m) or f"__unresolved__:{m}"
        by_id.setdefault(node_id, set()).add(m)
    return list(by_id.values())


# --- graph-metric math, with teeth (a wrong prediction must not score 1.0) ---

def test_triple_f1_is_perfect_on_the_gold_graph_and_drops_on_a_bad_extraction():
    assert triple_prf(GOLD_TRIPLES, GOLD_TRIPLES) == (1.0, 1.0, 1.0)
    # drop one gold edge, invent one bogus edge: 4 correct of 5 predicted, 4 of 5 gold -> F1 0.8
    predicted = (frozenset(GOLD_TRIPLES) - {next(iter(GOLD_TRIPLES))}) | {("plan:saver", "HAS_TERM", "term:bogus")}
    p, r, f = triple_prf(predicted, GOLD_TRIPLES)
    assert (p, r, f) == pytest.approx((0.8, 0.8, 0.8))


def test_graph_metric_degenerate_inputs():
    assert triple_prf(frozenset(), frozenset()) == (0.0, 0.0, 0.0)
    assert bcubed_prf([{"x"}], [{"y"}])[2] == 0.0        # disjoint mention sets -> no shared cluster
    assert path_recall({("a",)}, frozenset()) == 0.0     # no gold paths to recall


def test_entity_resolution_f1_perfect_then_penalises_a_miscluster():
    perfect = [{"Ada", "the customer", "account holder"}, {"Saver"}]
    assert bcubed_prf(perfect, GOLD_ENTITY_CLUSTERS)[2] == pytest.approx(1.0)
    assert pairwise_prf(perfect, GOLD_ENTITY_CLUSTERS)[2] == pytest.approx(1.0)
    # "account holder" torn from the customer and glued to the plan: B-Cubed F1 drops below 1
    miscluster = [{"Ada", "the customer"}, {"account holder", "Saver"}]
    assert bcubed_prf(miscluster, GOLD_ENTITY_CLUSTERS)[2] == pytest.approx(0.70588, abs=1e-4)


# --- the in-memory adapter: resolution, traversal, contract ---

def test_the_adapter_resolves_surface_forms_to_canonical_nodes(graph):
    assert graph.resolve("the customer") == "cust:legacy"
    assert graph.resolve("ADA") == "cust:legacy"        # case-insensitive
    assert graph.resolve("Saver") == "plan:saver"
    assert graph.resolve("nobody") is None
    clusters = _predicted_clusters(graph, ["Ada", "the customer", "account holder", "Saver"])
    assert bcubed_prf(clusters, GOLD_ENTITY_CLUSTERS)[2] == pytest.approx(1.0)


def test_multi_hop_traversal_collects_every_gold_path(graph):
    paths = graph.paths(COLD_OPEN_CASE.start, COLD_OPEN_CASE.goal, COLD_OPEN_CASE.max_hops)
    assert path_recall(paths, COLD_OPEN_CASE.gold_paths) == 1.0
    assert list(paths) == sorted(paths)                 # deterministic, sorted output


def test_a_hop_short_traversal_collapses_recall_and_one_of_two_paths_scores_half(graph):
    # stopping one hop short (max_hops=2) reaches neither 3-edge path: recall collapses
    short = graph.paths(COLD_OPEN_CASE.start, COLD_OPEN_CASE.goal, 2)
    assert path_recall(short, COLD_OPEN_CASE.gold_paths) == 0.0
    # and a retrieval that gets one of the two reasoning paths scores exactly one half
    one_path = frozenset({next(iter(COLD_OPEN_CASE.gold_paths))})
    assert path_recall(one_path, COLD_OPEN_CASE.gold_paths) == 0.5


def test_null_query_abstains_when_no_path_exists(graph):
    # the customer node has no path to the router-reset command node: traversal returns nothing
    assert graph.paths("cust:legacy", "cmd:reset", 5) == ()


def test_a_poisoned_node_is_data_and_does_not_corrupt_the_multi_hop_answer():
    poisoned = InMemoryGraph(GRAPH_NODES + [POISON_NODE], GRAPH_EDGES + [POISON_EDGE])
    paths = poisoned.paths(COLD_OPEN_CASE.start, COLD_OPEN_CASE.goal, COLD_OPEN_CASE.max_hops)
    # the injected node is retrievable data, but it sits off every legitimate reasoning path
    assert path_recall(paths, COLD_OPEN_CASE.gold_paths) == 1.0
    assert all(POISON_NODE.id not in path for path in paths)


def test_knowledge_graph_contract_holds_for_the_in_memory_adapter(graph):
    check_knowledge_graph_contract(
        graph,
        start=COLD_OPEN_CASE.start,
        goal=COLD_OPEN_CASE.goal,
        max_hops=COLD_OPEN_CASE.max_hops,
        expected_paths=COLD_OPEN_CASE.gold_paths,
        a_mention="the customer",
        its_id="cust:legacy",
    )


# --- the two-adapter contract, at the seams the shared suite doesn't reach (one port, two backends) ---

def test_an_ambiguous_mention_resolves_to_the_lowest_id_like_the_neo4j_adapter():
    # Two nodes share the alias 'Ada'. The port's tie-break must be stable AND identical across
    # adapters: Neo4jKnowledgeGraph.resolve is `ORDER BY n.id LIMIT 1` (lowest id), so InMemoryGraph
    # must return the lowest id too — not the last-inserted node its index would otherwise keep.
    ambiguous = InMemoryGraph(
        [Node("cust:z-newer", "Customer", "Ada"), Node("cust:a-older", "Customer", "Ada")],
        [],
    )
    assert ambiguous.resolve("ada") == "cust:a-older"      # lowest id, not insertion-order last


def test_parallel_edges_dedupe_so_neighbors_and_paths_match_the_neo4j_adapter():
    # A multigraph: two parallel edges a->b under different relations. neighbours is a set of nodes,
    # and paths must not emit the same path twice (Neo4j de-dupes via DISTINCT / a set of paths); the
    # in-memory adapter must agree or the same traversal scores differently across the two backends.
    multi = InMemoryGraph(
        [Node("a", "N", "A"), Node("b", "N", "B")],
        [Edge("a", "REL_X", "b"), Edge("a", "REL_Y", "b")],
    )
    assert multi.neighbors("a") == ("b",)                  # distinct, not ('b', 'b')
    assert multi.paths("a", "b", 1) == (("a", "b"),)       # one path, not two identical ones


def test_paths_rejects_a_non_positive_max_hops_like_the_neo4j_adapter(graph):
    # A degenerate budget is abstain-vs-crash across backends unless both validate: Neo4jKnowledgeGraph
    # raises on max_hops < 1, so InMemoryGraph must too (not silently return () an eval reads as
    # "no path / abstain").
    with pytest.raises(ValueError):
        graph.paths(COLD_OPEN_CASE.start, COLD_OPEN_CASE.goal, 0)


def test_bcubed_recall_charges_for_a_prediction_that_omits_a_gold_mention():
    # Entity-resolution honesty: a system that DROPS a mention must be charged a recall miss, not have
    # the mention silently excluded from the average (which inflates the score). B-Cubed averages
    # recall over the GOLD mentions, so the omitted 'c' contributes 0.
    p, r, _f = bcubed_prf([{"a", "b"}], [{"a", "b", "c"}])
    assert p == pytest.approx(1.0)                          # the two kept mentions are perfectly precise
    assert r == pytest.approx(4 / 9)                        # (2/3 + 2/3 + 0)/3 — NOT 2/3 (old intersection bug)


# --- the graph-vs-vector study earns its verdict at the deployed budget (task graph) ---

def test_the_relational_query_needs_a_graph_edge_that_a_keyword_vector_misses_even_at_deployed_k():
    # The honest graph-RAG case that makes 'adopt the graph on the relational slice' TRUE: the
    # customer->plan link is an account EDGE, not co-occurring text, so a keyword vector reaches the
    # throttling chunk but never the plan chunk from a customer-worded query — recall tops out at 0.5
    # even at the DEPLOYED k=3, while graph traversal collects every hop (1.0). A pure retrieval-budget
    # gap would close as k rises; this one does not, which is the structural tell.
    retriever = InMemoryRetriever(RELATIONAL_CORPUS)
    got1 = [c.doc_id for c in retriever.search_chunks(RELATIONAL_QUERY, k=1, config=RetrievalConfig())]
    got3 = [c.doc_id for c in retriever.search_chunks(RELATIONAL_QUERY, k=3, config=RetrievalConfig())]
    assert recall_at_k(got1, RELATIONAL_RELEVANT, 1) == 0.5
    assert recall_at_k(got3, RELATIONAL_RELEVANT, 3) == 0.5      # deployed budget still misses the plan hop
    graph = InMemoryGraph(GRAPH_NODES, GRAPH_EDGES)
    assert graph.resolve("account holder") == "cust:legacy"     # the bridge the vector has no text for
    graph_recall = path_recall(
        graph.paths(COLD_OPEN_CASE.start, COLD_OPEN_CASE.goal, COLD_OPEN_CASE.max_hops),
        COLD_OPEN_CASE.gold_paths,
    )
    assert graph_recall == 1.0                                   # traversal collects every hop
