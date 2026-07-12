"""A frozen registry of REALISTIC IR-metric mutants (plausible human bugs), each paired with a witness
input that a Phase-1 test asserts on. ``kills(m)`` is true when the mutant disagrees with the correct
metric on that witness — i.e. the real Phase-1 assertion would fail on the mutant. Deterministic and
pure: no LLM here, this is the gate-safe proof that the metric suite has teeth against real bugs.

Three of these mutants originally SURVIVED (short-list precision, last-vs-first reciprocal rank, and
÷hits vs ÷relevant MAP): the Phase-1 suite had no assertion that distinguished them. Those gaps were
closed by adding the exact witness assertions below to ``test_ir_metrics.py`` — mutation testing
driving real test coverage, which is the whole point.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from evals.retrieval import ir_metrics


@dataclass(frozen=True)
class Mutant:
    name: str
    realistic_bug: str        # the human mistake this stands in for
    metric: str               # the ir_metrics function it corrupts
    witness: tuple            # args where it must disagree with the correct function
    fn: Callable              # the buggy reimplementation


# --- the buggy reimplementations (each a mistake you have seen in a real review) ---

def _recall_divides_by_k(retrieved, relevant, k):
    top = set(retrieved[:k])
    return sum(1 for d in relevant if d in top) / k                 # BUG: ÷k, not ÷|relevant|


def _precision_shrinks_denominator(retrieved, relevant, k):
    top = retrieved[:k]
    return sum(1 for d in top if d in relevant) / max(1, len(top))  # BUG: ÷len(top), flatters short lists


def _reciprocal_rank_takes_last(retrieved, relevant):
    last = 0
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            last = i                                                # BUG: keeps the LAST relevant rank
    return 1.0 / last if last else 0.0


def _ndcg_exponential_gain(retrieved, relevance, k):
    gains = relevance if isinstance(relevance, dict) else {d: 1.0 for d in relevance}
    dcg = sum((2 ** gains.get(d, 0.0) - 1) / math.log2(i + 1)       # BUG: exponential gain, not linear
              for i, d in enumerate(retrieved[:k], start=1))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return 0.0 if idcg == 0 else dcg / idcg


def _map_divides_by_hits(retrieved, relevant, k):
    hits, summed = 0, 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        if d in relevant:
            hits += 1
            summed += hits / i
    return summed / hits if hits else 0.0                          # BUG: ÷hits, never charges for a miss


IR_METRIC_MUTANTS: list[Mutant] = [
    Mutant("recall_divides_by_k", "confuses recall with precision-style ÷k",
           "recall_at_k", (["a", "b"], frozenset({"a", "b", "c"}), 2), _recall_divides_by_k),
    Mutant("precision_shrinks_denominator", "divides by len(retrieved) so a short list looks perfect",
           "precision_at_k", (["y"], frozenset({"y"}), 3), _precision_shrinks_denominator),
    Mutant("reciprocal_rank_takes_last", "keeps the last relevant rank instead of the first",
           "reciprocal_rank", (["y", "x"], frozenset({"x", "y"})), _reciprocal_rank_takes_last),
    Mutant("ndcg_exponential_gain", "uses 2^rel-1 gain, wrong for graded relevance",
           "ndcg_at_k", (["a", "b", "c", "d"], {"a": 3.0, "b": 2.0, "d": 1.0}, 4), _ndcg_exponential_gain),
    Mutant("map_divides_by_hits", "averages by hits found, so misses are never charged",
           "average_precision_at_k", (["y"], frozenset({"x", "y"}), 3), _map_divides_by_hits),
]


def kills(mutant: Mutant) -> bool:
    """True iff the correct metric and the mutant disagree on the witness — i.e. the Phase-1 assertion
    that pins the correct value would fail on the mutant. Deterministic, no LLM."""
    correct = getattr(ir_metrics, mutant.metric)
    return correct(*mutant.witness) != mutant.fn(*mutant.witness)
