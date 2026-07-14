"""Retrieval IR metrics, hermetic: the classical, judge-free retrieval metrics computed over
a ranked list of chunk ids and a set of relevant ids. Deterministic closed forms (BEIR/trec_eval
convention: linear gain, log2(rank+1) discount), unit-checked against hand-worked values, then run
over the real ``InMemoryRetriever`` on the labelled golden slice.
"""
from __future__ import annotations

import math
from statistics import fmean

import pytest

from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.domain.retrieval import RetrievalConfig
from evals.datasets.retrieval_golden import RETRIEVAL_CORPUS, RETRIEVAL_GOLDEN
from quality.ir_metrics import (
    average_precision_at_k,
    dcg_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    rank_overlap_at_k,
    recall_at_k,
    reciprocal_rank,
)

# A single relevant chunk at rank 2 of three: the running worked example for the binary metrics.
_RETRIEVED = ["x", "y", "z"]
_RELEVANT = frozenset({"y"})


def test_precision_at_k_divides_by_k_trec_style():
    assert precision_at_k(_RETRIEVED, _RELEVANT, 1) == 0.0  # x is not relevant
    assert precision_at_k(_RETRIEVED, _RELEVANT, 2) == pytest.approx(1 / 2)
    assert precision_at_k(_RETRIEVED, _RELEVANT, 3) == pytest.approx(1 / 3)
    # short list: the denominator is still k, not len(retrieved) (closes a gap a mutant found)
    assert precision_at_k(["y"], frozenset({"y"}), 3) == pytest.approx(1 / 3)


def test_recall_at_k_is_fraction_of_relevant_found():
    assert recall_at_k(_RETRIEVED, _RELEVANT, 1) == 0.0
    assert recall_at_k(_RETRIEVED, _RELEVANT, 2) == 1.0
    assert recall_at_k(["a", "b"], frozenset({"a", "b", "c"}), 2) == pytest.approx(2 / 3)


def test_hit_rate_is_binary_did_anything_relevant_appear():
    assert hit_rate_at_k(_RETRIEVED, _RELEVANT, 1) == 0.0
    assert hit_rate_at_k(_RETRIEVED, _RELEVANT, 3) == 1.0


def test_reciprocal_rank_is_one_over_first_relevant_rank():
    assert reciprocal_rank(_RETRIEVED, _RELEVANT) == pytest.approx(1 / 2)
    assert reciprocal_rank(["y", "x"], _RELEVANT) == 1.0
    assert reciprocal_rank(["x", "z"], _RELEVANT) == 0.0  # never retrieved
    # the FIRST relevant rank decides, not the last (closes a gap a mutant found)
    assert reciprocal_rank(["y", "x"], frozenset({"x", "y"})) == 1.0


def test_average_precision_rewards_relevant_ranked_early():
    # one relevant at rank 2 -> AP = precision@2 / |relevant| = 0.5
    assert average_precision_at_k(_RETRIEVED, _RELEVANT, 3) == pytest.approx(0.5)
    # both relevant at ranks 1 and 2 -> AP = (1/1 + 2/2) / 2 = 1.0
    assert average_precision_at_k(["a", "b", "c"], frozenset({"a", "b"}), 3) == pytest.approx(1.0)
    # normalised by |relevant|, so a MISS is charged for: found 'y', missed 'x' -> 1/2 (mutant found this)
    assert average_precision_at_k(["y"], frozenset({"x", "y"}), 3) == pytest.approx(0.5)


def test_ndcg_binary_matches_hand_worked_value():
    # DCG = 1/log2(2+1) at rank 2; IDCG = 1/log2(1+1) = 1 -> NDCG = 1/log2(3)
    assert ndcg_at_k(_RETRIEVED, _RELEVANT, 3) == pytest.approx(1 / math.log2(3))


def test_ndcg_graded_matches_trec_eval_linear_gain_convention():
    retrieved = ["a", "b", "c", "d"]
    relevance = {"a": 3.0, "b": 2.0, "d": 1.0}  # c is irrelevant (gain 0)
    dcg = 3 / math.log2(2) + 2 / math.log2(3) + 0 + 1 / math.log2(5)
    idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)  # ideal order 3,2,1,0
    assert ndcg_at_k(retrieved, relevance, 4) == pytest.approx(dcg / idcg)


def test_ndcg_reproduces_the_published_jarvelin_kekalainen_worked_example():
    """Transcribed literals, not written for this suite: Wikipedia's Discounted Cumulative Gain
    article's worked example, which traces to Jarvelin and Kekalainen 2002, linear gain and a
    log2(rank + 1) discount, the same trec_eval/BEIR convention this module's docstring claims.
    The six retrieved results carry grades (3, 2, 3, 0, 1, 2). Wikipedia's own ideal ranking is
    not scoped to those six ids; it widens the pool with two more graded documents that were
    never retrieved (d7 = 3, d8 = 2), so the ideal order becomes 3, 3, 3, 2, 2, 2 rather than
    3, 3, 2, 2, 1, 0. This fixture reproduces that same widened relevance map so both assertions
    below are byte exact against the article: DCG_6 = 6.861, IDCG_6 = 8.740, nDCG_6 = 0.785. An
    outside, published number pins the convention independently of any formula this codebase
    writes for itself, so a regression the move introduced (say, a silent switch to exponential
    gain) would show up as a published value mismatch, not just the earlier test agreeing with
    its own inline arithmetic."""
    retrieved = ["d1", "d2", "d3", "d4", "d5", "d6"]
    relevance = {
        "d1": 3.0, "d2": 2.0, "d3": 3.0, "d4": 0.0, "d5": 1.0, "d6": 2.0,
        # Never retrieved: present only so IDCG widens to Wikipedia's own ideal ranking
        # instead of scoping to the six retrieved ids.
        "d7": 3.0, "d8": 2.0,
    }
    assert dcg_at_k(retrieved, relevance, 6) == pytest.approx(6.861, abs=5e-4)
    assert ndcg_at_k(retrieved, relevance, 6) == pytest.approx(0.785, abs=5e-4)


def test_perfect_and_empty_edges():
    assert ndcg_at_k(["a", "b"], frozenset({"a", "b"}), 2) == pytest.approx(1.0)
    assert ndcg_at_k([], frozenset({"a"}), 3) == 0.0
    assert recall_at_k(["a"], frozenset(), 1) == 0.0  # no relevant -> guarded, no ZeroDivision
    assert precision_at_k([], frozenset({"a"}), 3) == 0.0


def test_validation_and_degenerate_relevance():
    with pytest.raises(ValueError):
        precision_at_k(["a"], frozenset({"a"}), 0)          # k must be >= 1
    assert average_precision_at_k([], frozenset(), 3) == 0.0  # no relevant docs -> 0.0, not a crash
    assert ndcg_at_k(["a"], frozenset(), 3) == 0.0           # no relevance to normalise against -> 0.0


def _retrieved_ids(retriever: InMemoryRetriever, query: str, k: int) -> list[str]:
    return [c.doc_id for c in retriever.search_chunks(query, k=k, config=RetrievalConfig())]


def test_inmemory_retriever_clears_the_retrieval_floor_on_the_golden_slice():
    """The healthy case: over the labelled slice every gold chunk is found and ranked first."""
    retriever = InMemoryRetriever(RETRIEVAL_CORPUS)
    per_case = [(c, _retrieved_ids(retriever, c.query, 5)) for c in RETRIEVAL_GOLDEN]

    hit = fmean(hit_rate_at_k(r, c.relevant_ids, 3) for c, r in per_case)
    mrr = fmean(reciprocal_rank(r, c.relevant_ids) for c, r in per_case)
    p_at_1 = fmean(precision_at_k(r, c.relevant_ids, 1) for c, r in per_case)
    recall_at_5 = fmean(recall_at_k(r, c.relevant_ids, 5) for c, r in per_case)

    assert hit == 1.0                    # nothing is missed entirely
    assert mrr == 1.0                    # the top hit is always relevant
    assert p_at_1 == 1.0
    assert recall_at_5 >= 0.8            # doc-07's Recall heuristic, comfortably cleared


def test_recall_at_1_exposes_the_missed_second_chunk_on_multi_hop_queries():
    """recall@1 is below the ceiling on the two-relevant queries: a single lookup gets one hop,
    not both. This is the number that makes the case for graph RAG, and it must not read 1.0."""
    retriever = InMemoryRetriever(RETRIEVAL_CORPUS)
    multi = [c for c in RETRIEVAL_GOLDEN if len(c.relevant_ids) > 1]
    recall_at_1 = fmean(
        recall_at_k(_retrieved_ids(retriever, c.query, 5), c.relevant_ids, 1) for c in multi
    )
    assert recall_at_1 == pytest.approx(0.5)


def test_a_retriever_that_misses_the_gold_fails_the_floor_the_gate_has_teeth():
    """A degraded retriever (returns only the wrong chunk) drives hit rate to zero: proof the
    metric would catch a broken retriever a fluent answer would otherwise hide."""
    only_wrong = [hit_rate_at_k(["router-reset"], c.relevant_ids, 3) for c in RETRIEVAL_GOLDEN
                  if "router-reset" not in c.relevant_ids]
    assert fmean(only_wrong) == 0.0


# ---- rank_overlap_at_k: structural agreement between two RANKINGS (the metamorphic primitive) ----


def test_rank_overlap_is_the_fraction_of_k_two_rankings_share():
    # identical top-k -> full agreement
    assert rank_overlap_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    # one of two shared -> half agreement
    assert rank_overlap_at_k(["a", "b"], ["a", "c"], 2) == pytest.approx(0.5)
    # nothing shared -> zero agreement
    assert rank_overlap_at_k(["a", "b"], ["c", "d"], 2) == 0.0


def test_rank_overlap_is_symmetric_and_ignores_order_within_top_k():
    a, b = ["x", "y", "z"], ["z", "y", "w"]
    assert rank_overlap_at_k(a, b, 3) == rank_overlap_at_k(b, a, 3)
    assert rank_overlap_at_k(a, b, 3) == pytest.approx(2 / 3)  # y, z shared of 3


def test_rank_overlap_divides_by_k_trec_style_short_lists_are_not_flattered():
    # only one ranking has anything in the second slot: division is still by k (mirrors
    # precision_at_k's own convention, not by the shorter list's length)
    assert rank_overlap_at_k(["a"], ["a", "b"], 2) == pytest.approx(0.5)


def test_rank_overlap_validates_k():
    with pytest.raises(ValueError):
        rank_overlap_at_k(["a"], ["a"], 0)
