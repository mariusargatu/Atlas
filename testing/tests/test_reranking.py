"""P2 (reranking + fusion, hermetic): make the reranker earn its place (doc 07). The reranker
reorders the first pass; the test is whether it HELPS, measured as a with/without NDCG delta with a
recall guardrail, and whether the lift beats run-to-run noise (a paired test, ``evals.stats``, not a
glance at two averages). RRF is the deterministic hybrid whose job is to never underperform its
inputs. Reranker scores are REPLAYED from a cassette (a real cross-encoder is deferred to dev/prod),
so the comparison stays byte-stable and torch-free.
"""
from __future__ import annotations

from statistics import fmean

import pytest

from atlas.adapters.cassette_reranker import CassetteReranker
from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.ports.knowledge import Chunk
from evals.datasets.retrieval_golden import RETRIEVAL_CORPUS, RETRIEVAL_GOLDEN
from evals.retrieval.fusion import reciprocal_rank_fusion
from evals.retrieval.ir_metrics import ndcg_at_k, recall_at_k
from evals.stats import paired_bootstrap_diff, paired_permutation_test

# Recorded cross-encoder scores per (query, doc_id): higher is better. Hand-set to the values a
# domain-tuned reranker would produce, pushing the plan-specific clause above the generic plan page
# on the data-cap questions. This is the cassette the gate replays; no model runs in the lane.
REPLAYED_RERANK_SCORES: dict[str, dict[str, float]] = {
    "what happens when I exceed my data cap": {"throttling-terms": 0.95, "plan-legacy": 0.80, "plan-current": 0.10},
    "capped plan throttled data": {"throttling-terms": 0.95, "plan-legacy": 0.80, "plan-current": 0.10, "coverage-regional": 0.05},
}


def _ids(chunks: list[Chunk]) -> list[str]:
    return [c.doc_id for c in chunks]


def test_cassette_reranker_reorders_by_recorded_score_deterministically():
    reranker = CassetteReranker({"q": {"b": 0.9, "a": 0.2, "c": 0.5}})
    chunks = [Chunk("a", "a"), Chunk("b", "b"), Chunk("c", "c")]
    assert _ids(reranker.rerank("q", chunks)) == ["b", "c", "a"]
    # an unseen query leaves order untouched (nothing to reorder by); unseen docs sink to the back
    assert _ids(reranker.rerank("other", chunks)) == ["a", "b", "c"]


def test_cassette_reranker_sinks_unscored_docs_to_the_back_in_input_order():
    # a partially-scored query: 'b' is recorded, 'a' and 'c' are not -> b first, then a,c at -inf
    # keeping their input order (pins the _MISSING default and the stable tie-break together)
    reranker = CassetteReranker({"q": {"b": 0.9}})
    chunks = [Chunk("a", "a"), Chunk("b", "b"), Chunk("c", "c")]
    assert _ids(reranker.rerank("q", chunks)) == ["b", "a", "c"]


def test_rrf_never_underperforms_either_single_backend():
    """The hybrid regression guard: RRF of a lexical and a dense ranking is at least as good as the
    better of the two on this slice. Not a promise of a big lift (vanilla RRF is a modest gain), a
    promise that fusing does not make things worse."""
    relevant = frozenset({"d1", "d3"})
    lexical = ["d1", "d2", "d3", "d4"]
    dense = ["d3", "d1", "d5", "d2"]
    fused = reciprocal_rank_fusion([lexical, dense])
    assert fused == ["d1", "d3", "d2", "d5", "d4"]  # deterministic order
    n_lex, n_dense, n_fused = (ndcg_at_k(r, relevant, 5) for r in (lexical, dense, fused))
    assert n_fused >= max(n_lex, n_dense)


def test_rrf_is_order_stable_on_ties():
    # identical single list -> RRF preserves order (all ties broken by original rank then id)
    assert reciprocal_rank_fusion([["a", "b", "c"]]) == ["a", "b", "c"]


def test_rrf_tie_break_is_id_ordered_not_insertion_ordered():
    # b and a get identical scores (1/61) and identical best rank (1); the id tie-break must decide,
    # yielding ['a','b'] and NOT the ['b','a'] a score-only sort would leave in insertion order.
    assert reciprocal_rank_fusion([["b"], ["a"]]) == ["a", "b"]


def test_rrf_rejects_a_non_positive_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)


def _rerank_experiment():
    """Baseline vs reranked NDCG@5 per golden query, with recall@5 for the guardrail."""
    retriever = InMemoryRetriever(RETRIEVAL_CORPUS)
    rer = CassetteReranker(REPLAYED_RERANK_SCORES)
    base_ndcg, rr_ndcg, base_rec, rr_rec = [], [], [], []
    for case in RETRIEVAL_GOLDEN:
        first = retriever.search(case.query, k=5)
        reranked = rer.rerank(case.query, list(first))
        base_ndcg.append(ndcg_at_k(_ids(first), case.relevant_ids, 5))
        rr_ndcg.append(ndcg_at_k(_ids(reranked), case.relevant_ids, 5))
        base_rec.append(recall_at_k(_ids(first), case.relevant_ids, 5))
        rr_rec.append(recall_at_k(_ids(reranked), case.relevant_ids, 5))
    return base_ndcg, rr_ndcg, base_rec, rr_rec


def test_reranker_helps_and_never_hurts_with_a_recall_guardrail():
    base_ndcg, rr_ndcg, base_rec, rr_rec = _rerank_experiment()
    # never hurts: reranked NDCG >= baseline on every query (permanent regression assertion)
    assert all(rr >= b - 1e-12 for rr, b in zip(rr_ndcg, base_ndcg))
    # helps directionally: the mean went up
    assert fmean(rr_ndcg) > fmean(base_ndcg)
    # recall guardrail: reranking dropped no needle it started with
    assert rr_rec == base_rec


def test_the_lift_is_inside_the_noise_at_this_sample_size():
    """'The number went up' is the beginning of the question, not the end. Over seven queries the
    reranker's NDCG lift does not clear a paired permutation test, so it has not yet earned its place
    on this evidence, exactly the paired-test discipline the statistics article insists on."""
    base_ndcg, rr_ndcg, *_ = _rerank_experiment()
    p = paired_permutation_test(rr_ndcg, base_ndcg, seed=7)
    assert p > 0.05                                            # not distinguishable from noise yet
    point, lo, _hi = paired_bootstrap_diff(rr_ndcg, base_ndcg, seed=7)
    assert point > 0.0 and lo <= 0.0                           # positive point, CI still straddles zero
