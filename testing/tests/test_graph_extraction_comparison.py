"""The gold (registry Postgres CTE) vs LLM-extracted (Neo4j `SimpleKGPipeline`) graph comparison
(SP9 task 2), hermetic: `quality.graph_metrics` scoring `evals.graphrag.registry_graph`'s fixture
gold graph against its fixture extracted graph. The numbers below are hand-derived from the real,
committed `corpus/registry/core.yaml` (19 edges, verified against the loaded registry) plus the three
named, documented extraction failures (see `registry_graph.py`'s own module docstring), not guessed:
this test has teeth precisely because the fixture is real data, not an invented toy.
"""
from __future__ import annotations

import pytest
from atlas.adapters.inmemory_graph import InMemoryGraph
from evals.graphrag.registry_graph import (
    CHAIN_GOAL,
    CHAIN_MAX_HOPS,
    CHAIN_START,
    EXTRACTED_EDGES,
    EXTRACTED_ENTITY_CLUSTERS,
    EXTRACTED_TRIPLES,
    GOLD_EDGES,
    GOLD_ENTITY_CLUSTERS,
    GOLD_NODES,
    GOLD_TRIPLES,
    MISSED_RELATION,
    SPURIOUS_RELATION,
)
from quality.graph_metrics import bcubed_prf, pairwise_prf, path_recall, triple_prf


def test_gold_has_the_real_registrys_19_edges() -> None:
    assert len(GOLD_TRIPLES) == 19
    assert len(GOLD_EDGES) == 19


def test_triple_prf_scores_the_documented_miss_and_spurious_edge() -> None:
    """16 extracted triples (19 gold - 3 collapsed duplicates from the legacy-plan fold - 1 missed
    + 1 spurious), 15 of them correct: precision 15/16, recall 15/19, F1 exactly 6/7."""
    assert MISSED_RELATION not in EXTRACTED_TRIPLES
    assert SPURIOUS_RELATION in EXTRACTED_TRIPLES
    assert len(EXTRACTED_TRIPLES) == 16
    p, r, f1 = triple_prf(EXTRACTED_TRIPLES, GOLD_TRIPLES)
    assert (p, r, f1) == pytest.approx((15 / 16, 15 / 19, 6 / 7))


def test_bcubed_charges_precision_not_recall_for_the_false_merge() -> None:
    """The mirror image of `test_graph_rag.py`'s own `test_bcubed_recall_charges_for_a_prediction_
    that_omits_a_gold_mention`: THIS fixture's extraction error is an invented merge (plan-fiber-100
    and plan-fiber-100-legacy folded into one), which B-Cubed charges to PRECISION (mean 0.75, the
    two merged mentions each score 0.5), never to recall (every gold mention is still reachable
    through the merged cluster, so recall stays a perfect 1.0)."""
    p, r, f1 = bcubed_prf(EXTRACTED_ENTITY_CLUSTERS, GOLD_ENTITY_CLUSTERS)
    assert (p, r, f1) == pytest.approx((0.75, 1.0, 6 / 7))


def test_pairwise_f1_is_zero_when_every_gold_cluster_is_a_singleton() -> None:
    """Every GOLD cluster here is a singleton (no true co-clustering to recall at all), so pairwise
    P/R/F1 is degenerate by construction: the one predicted (false) pair scores a straight precision
    loss, and recall is guarded to 0.0 rather than raising on an empty gold-pairs denominator."""
    assert pairwise_prf(EXTRACTED_ENTITY_CLUSTERS, GOLD_ENTITY_CLUSTERS) == (0.0, 0.0, 0.0)


def test_path_recall_collapses_to_zero_on_the_named_plan_region_fee_chain() -> None:
    """The task's own named chain, plan -> region -> fee: GOLD reaches fee-equipment-rental from
    plan-fiber-100 in exactly 2 hops via region-north's overrides_fee edge; EXTRACTED cannot, since
    that is exactly the relation MISSED_RELATION drops -- multi-hop completeness fails on one
    missing edge, not a hop-short traversal budget."""
    gold_graph = InMemoryGraph(GOLD_NODES, GOLD_EDGES)
    extracted_graph = InMemoryGraph(GOLD_NODES, EXTRACTED_EDGES)

    gold_paths = gold_graph.paths(CHAIN_START, CHAIN_GOAL, CHAIN_MAX_HOPS)
    assert gold_paths == (("plan-fiber-100", "region-north", "fee-equipment-rental"),)

    extracted_paths = extracted_graph.paths(CHAIN_START, CHAIN_GOAL, CHAIN_MAX_HOPS)
    assert extracted_paths == ()

    assert path_recall(extracted_paths, frozenset(gold_paths)) == 0.0
