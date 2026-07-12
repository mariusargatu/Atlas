"""`task graph`: the graph-extraction study (SP9 task 2), reframed from its pre-rewrite shape (the
old "graph beats vector on the relational cold-open" comparison, which ran over a hand-authored toy
graph predating the registry -- superseded as data, kept only as a shape: `main()` prints a verdict
from deterministic adapters and metrics, no judge, no live model call).

The comparison this study now runs is SP9's own two-graph disposition, named explicitly rather than
left implicit: `registry_graph.GOLD_*` (`corpus/registry/core.yaml`'s own typed edges, the same
conversion the real `PgKnowledgeGraph` Postgres tables would be loaded from) against
`registry_graph.EXTRACTED_*` (a fixture standing in for `neo4j-graphrag`'s `SimpleKGPipeline` reading
the rendered corpus text -- a real Neo4j extraction run is live/operator only, `task graph-up`,
deferred). Neither graph is dead weight: GOLD is what the graph-RAG variant
(`atlas.orchestration.graph_rag`) actually traverses; EXTRACTED is the measured "what if we had to
extract it instead of materializing it from the registry" baseline, scored via `quality.graph_metrics`
(`triple_prf`/`pairwise_prf`/`bcubed_prf`/`path_recall`), the exact numbers
`testing/tests/test_graph_extraction_comparison.py` pins in the hermetic gate.
"""
from __future__ import annotations

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
)
from quality.graph_metrics import bcubed_prf, pairwise_prf, path_recall, triple_prf


def main() -> None:
    gold_graph = InMemoryGraph(GOLD_NODES, GOLD_EDGES)
    extracted_graph = InMemoryGraph(GOLD_NODES, EXTRACTED_EDGES)

    t_p, t_r, t_f = triple_prf(EXTRACTED_TRIPLES, GOLD_TRIPLES)
    pw_p, pw_r, pw_f = pairwise_prf(EXTRACTED_ENTITY_CLUSTERS, GOLD_ENTITY_CLUSTERS)
    bc_p, bc_r, bc_f = bcubed_prf(EXTRACTED_ENTITY_CLUSTERS, GOLD_ENTITY_CLUSTERS)

    gold_paths = gold_graph.paths(CHAIN_START, CHAIN_GOAL, CHAIN_MAX_HOPS)
    extracted_paths = extracted_graph.paths(CHAIN_START, CHAIN_GOAL, CHAIN_MAX_HOPS)
    recall = path_recall(extracted_paths, frozenset(gold_paths))

    print("gold (registry Postgres CTE) vs LLM-extracted (Neo4j SimpleKGPipeline) graph comparison")
    print(f"  triple P/R/F1   (relationship resolution)  : {t_p:.3f} / {t_r:.3f} / {t_f:.3f}")
    print(f"  pairwise P/R/F1 (entity resolution)         : {pw_p:.3f} / {pw_r:.3f} / {pw_f:.3f}")
    print(f"  B-Cubed  P/R/F1 (entity resolution)         : {bc_p:.3f} / {bc_r:.3f} / {bc_f:.3f}")
    print(f"  path-recall on {CHAIN_START} -> {CHAIN_GOAL} ({CHAIN_MAX_HOPS} hop): {recall:.3f}")
    print(
        f"\nthe extracted graph never reads region-north's overrides_fee relation from the fee-schedule "
        f"prose (a plausible NER miss, not a strawman), so path-recall on the named 2-hop chain "
        f"({CHAIN_START} -> region-north -> {CHAIN_GOAL}) collapses to {recall:.1f}: multi-hop "
        "completeness fails on one missing edge, exactly the failure mode this study exists to catch "
        "before it reaches a live deployment. The gold graph, materialized straight from the registry's "
        "own typed edges (never guessed), has no such gap; a real Neo4j extraction run "
        "(`task graph-up`) would report the same three metrics against a genuine SimpleKGPipeline "
        "output instead of this fixture."
    )


if __name__ == "__main__":
    main()
