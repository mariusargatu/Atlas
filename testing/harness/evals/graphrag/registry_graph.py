"""The registry-derived GOLD graph vs a fixture LLM-EXTRACTED comparison graph (SP9 task 2): the data
`task graph` scores via `quality.graph_metrics`, and the same fixture `testing/tests/
test_graph_extraction_comparison.py`'s hermetic comparison test checks.

GOLD is not hand-authored: `GOLD_NODES`/`GOLD_EDGES` come straight from `rag_tools.graph_ingest.
registry_to_graph` over the real, committed `corpus/registry/core.yaml` (the same conversion
`PgKnowledgeGraph`'s own Postgres tables would be loaded from), so a registry edit changes this
fixture's answer too, rather than silently drifting from it.

EXTRACTED is hand-authored, standing in for a real `neo4j-graphrag` `SimpleKGPipeline` run over the
rendered corpus text (deferred, live/operator only: no Neo4j, no LLM call belongs in the hermetic
gate) -- three plausible, DOCUMENTED extraction failures, not a strawman:

  1. One relation an LLM reading prose is plausible to miss entirely: `region-north overrides_fee
     fee-equipment-rental` is stated in one regional fee-schedule document as a flat override
     amount, easy to read as a fee mention rather than the override RELATION the registry encodes.
  2. One spurious relation: `plan-fiber-100 applies_to fee-installation`, inventing a link the
     registry never asserts (the installation fee is universal, not plan specific; a generic
     extractor conflating "this plan's own page also happens to mention the installation fee" with
     "applies to" is a realistic false positive).
  3. One entity-resolution failure: `plan-fiber-100` and `plan-fiber-100-legacy` (whose rendered
     pages both describe "the Fiber 100 plan", differing only in a superseded/current status the
     registry tracks structurally, not always in prose an NER pass reliably keys off) are folded
     into a single extracted node id -- the fractured-cluster mistake B-Cubed is built to catch
     (Amigo et al. 2009). Every gold triple naming the legacy plan's OWN id is therefore either
     dropped (a true duplicate of the current plan's own triple survives) or, where the registry
     never separately declared the equivalent current-plan triple, silently lost.
"""
from __future__ import annotations

from pathlib import Path

from atlas.ports.knowledge_graph import Edge, Node
from corpus_tools.registry import load_registry
from rag_tools.graph_ingest import registry_to_graph

CORE_REGISTRY = Path("corpus/registry/core.yaml")


def _load_gold() -> tuple[list[Node], list[Edge]]:
    registry = load_registry([CORE_REGISTRY])
    return registry_to_graph(registry)


GOLD_NODES, GOLD_EDGES = _load_gold()
GOLD_TRIPLES: frozenset[tuple[str, str, str]] = frozenset((e.src, e.rel, e.dst) for e in GOLD_EDGES)

# ---- the fixture LLM-extracted arm: three named, documented failures (see module docstring) ------

MISSED_RELATION: tuple[str, str, str] = ("region-north", "overrides_fee", "fee-equipment-rental")
SPURIOUS_RELATION: tuple[str, str, str] = ("plan-fiber-100", "applies_to", "fee-installation")
_MISRESOLVED_SRC = {"plan-fiber-100-legacy": "plan-fiber-100"}  # the fractured-cluster fold


def _extracted_triples() -> frozenset[tuple[str, str, str]]:
    remapped = {
        (_MISRESOLVED_SRC.get(src, src), rel, _MISRESOLVED_SRC.get(dst, dst)) for src, rel, dst in GOLD_TRIPLES
    }
    return (remapped - {MISSED_RELATION}) | {SPURIOUS_RELATION}


EXTRACTED_TRIPLES: frozenset[tuple[str, str, str]] = _extracted_triples()
EXTRACTED_EDGES: list[Edge] = [Edge(src=s, rel=r, dst=d) for s, r, d in EXTRACTED_TRIPLES]

# Entity-resolution clusters (pairwise/B-Cubed). The registry declares no alias field, so there is
# no registry-derived mention-clustering ground truth to draw on; this is a small, clearly-labelled
# illustrative fixture instead. GOLD keeps every entity its own singleton cluster except the one pair
# EXTRACTED folds together (the same fractured-cluster mistake as above), so precision/recall on this
# pair is exactly what the merge costs, nothing else.
GOLD_ENTITY_CLUSTERS: list[set[str]] = [
    {"plan-fiber-100"},
    {"plan-fiber-100-legacy"},
    {"region-north"},
    {"fee-equipment-rental"},
]
EXTRACTED_ENTITY_CLUSTERS: list[set[str]] = [
    {"plan-fiber-100", "plan-fiber-100-legacy"},
    {"region-north"},
    {"fee-equipment-rental"},
]

# The 2-hop chain this task's own spec names verbatim: plan -> region -> fee. GOLD reaches it
# through region-north's overrides_fee edge; EXTRACTED cannot, since that is exactly the relation
# MISSED_RELATION drops -- multi-hop completeness collapsing from one missing edge, not a hop-short
# traversal budget.
CHAIN_START = "plan-fiber-100"
CHAIN_GOAL = "fee-equipment-rental"
CHAIN_MAX_HOPS = 2
