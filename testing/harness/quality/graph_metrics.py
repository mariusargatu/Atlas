"""Graph-RAG metrics: the three traversal failure modes as set arithmetic over ids.

Entity resolution (did each real entity map to exactly one node), relationship resolution (did the
graph carry the right edges), and multi-hop completeness (did traversal collect every hop). All are
deterministic given a fixed graph and gold labels, which is what lets them gate the hermetic lane
while judged answer quality stays in the operator lane.

Entity resolution reports BOTH pairwise and B-Cubed F1: pairwise alone hides a fractured cluster
(A-B and B-C linked but A-C missed), and B-Cubed catches it (Amigo et al. 2009). Pure stdlib, the
same discipline as ``quality.ir_metrics`` and ``quality.stats``.

Relocated here from ``evals/retrieval/graph_metrics.py`` (SP9 task 2, the same relocation pattern
SP7 task 2 already applied to ``stats``/``ir_metrics``: content and public API unchanged, only the
import path). SP9's own two-graph disposition is what this module now scores: the Postgres registry
CTE graph (``atlas.adapters.pg_knowledge_graph.PgKnowledgeGraph``, the GOLD graph) against the Neo4j
LLM-extracted comparison arm (``atlas.adapters.neo4j_graph.Neo4jKnowledgeGraph``), via
``testing/harness/evals/graphrag/__main__.py``'s ``task graph`` study.
"""
from __future__ import annotations

from collections.abc import Sequence

__all__ = ["bcubed_prf", "pairwise_prf", "path_recall", "triple_prf"]

_PRF = tuple[float, float, float]


def _prf(true_positive: int, predicted: int, gold: int) -> _PRF:
    p = true_positive / predicted if predicted else 0.0
    r = true_positive / gold if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return (p, r, f)


def triple_prf(predicted: frozenset | set, gold: frozenset | set) -> _PRF:
    """Precision/recall/F1 over ``(src, rel, dst)`` triples: fraction of predicted edges that are
    correct, and fraction of gold edges recovered. The relationship-resolution score."""
    pred, gold = set(predicted), set(gold)
    return _prf(len(pred & gold), len(pred), len(gold))


def _pairs(clusters: Sequence[set[str]]) -> set[frozenset[str]]:
    """The set of unordered co-clustered mention pairs across all clusters."""
    out: set[frozenset[str]] = set()
    for cluster in clusters:
        members = sorted(cluster)
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                out.add(frozenset({a, b}))
    return out


def pairwise_prf(predicted: Sequence[set[str]], gold: Sequence[set[str]]) -> _PRF:
    """Pairwise P/R/F1: of the mention pairs the prediction puts together, how many the gold does too,
    and vice versa. Cheap, but blind to fractured clusters, hence B-Cubed alongside it."""
    pred_pairs, gold_pairs = _pairs(predicted), _pairs(gold)
    return _prf(len(pred_pairs & gold_pairs), len(pred_pairs), len(gold_pairs))


def _membership(clusters: Sequence[set[str]]) -> dict[str, frozenset[str]]:
    """Map each mention to the (frozen) cluster it belongs to."""
    return {m: frozenset(cluster) for cluster in clusters for m in cluster}


def bcubed_prf(predicted: Sequence[set[str]], gold: Sequence[set[str]]) -> _PRF:
    """B-Cubed P/R/F1, averaged per mention (Amigo et al. 2009): precision is averaged over the
    PREDICTED mentions (share of each mention's predicted cluster that is in its gold cluster) and
    recall over the GOLD mentions (share of each mention's gold cluster the prediction kept together).
    Averaging over each side's own mention set, not their intersection, is what charges for a mention
    the prediction OMITS (a recall miss, contributing 0) or INVENTS (a precision miss) rather than
    silently dropping it and inflating the score. Iteration is sorted so the float sums are
    order-independent regardless of set hashing (ADR-007). Catches the fractured-cluster error
    pairwise F1 can miss."""
    pred_of, gold_of = _membership(predicted), _membership(gold)
    empty: frozenset[str] = frozenset()
    precisions = [len(pred_of[m] & gold_of.get(m, empty)) / len(pred_of[m]) for m in sorted(pred_of)]
    recalls = [len(gold_of[m] & pred_of.get(m, empty)) / len(gold_of[m]) for m in sorted(gold_of)]
    p = sum(precisions) / len(precisions) if precisions else 0.0
    r = sum(recalls) / len(recalls) if recalls else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return (p, r, f)


def path_recall(retrieved: frozenset | set | Sequence, gold: frozenset | set | Sequence) -> float:
    """Fraction of gold reasoning paths contained in the retrieved subgraph: 'did traversal collect
    every hop'. Equivalent to supporting-fact recall (HotpotQA) / evidence recall (2Wiki). 1.0 only
    when every gold path is present; a hop-short traversal reads below the ceiling."""
    retrieved, gold = set(retrieved), set(gold)
    if not gold:
        return 0.0
    return len(retrieved & gold) / len(gold)
