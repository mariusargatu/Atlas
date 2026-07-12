"""Retrieval domain, hermetic: RRF fusion (D8) is pure score arithmetic over ranked id lists, and
`RetrievalConfig` is a frozen dataclass, no I/O, no client. This is the seam `search_chunks` stands
on (`atlas/ports/knowledge.py`); a tiny fake adapter here pins the port's calling shape independent
of any real adapter.
"""
from __future__ import annotations

import dataclasses
import itertools

import pytest

from atlas.domain.retrieval import RRF_K, RetrievalConfig, rrf_fuse
from atlas.ports.knowledge import Chunk, Retriever


def test_retrieval_config_frozen_defaults():
    config = RetrievalConfig()
    assert config.k_fused == 50
    assert config.k_final == 5
    assert config.rerank_enabled is True
    assert config.exact_scan is False
    assert config.ef_search == 40
    assert config.lexical_only is False  # SP4 task 4: the embedding-down ladder rung's own knob
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.k_final = 10


def test_rrf_k_is_declared_once_as_the_fusion_default():
    """`RRF_K` used to be a `RetrievalConfig` field every caller carried and none varied, plus a
    second copy in a duplicate fusion module. One declaration now: `rrf_fuse`'s own default."""
    import inspect

    assert inspect.signature(rrf_fuse).parameters["k"].default == RRF_K == 60


def test_rrf_fuse_known_example_hand_worked():
    # score(d) = sum over rankings of 1/(k + rank_d), rank is 1-based. k=60.
    # a: rank1 in R1 (1/61) + rank3 in R2 (1/63)
    # b: rank2 in R1 (1/62) + rank1 in R2 (1/61)
    # c: rank3 in R1 (1/63) + rank2 in R2 (1/62)
    # b > a > c
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert [doc_id for doc_id, _score in fused] == ["b", "a", "c"]
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 62)


def test_rrf_fuse_tie_break_is_id_ascending_not_insertion_order():
    # 'z' and 'a' are each rank 1 of their own single-item ranking: identical score (1/61), so the
    # id-ascending tie-break must decide, not insertion order ('z' was listed first).
    fused = rrf_fuse([["z"], ["a"]], k=60)
    assert [doc_id for doc_id, _score in fused] == ["a", "z"]


def test_rrf_fuse_is_invariant_to_the_order_of_the_rankings_sequence():
    # Property: the fused score set depends only on each id's (ranking, rank) memberships, not on
    # what order the outer `rankings` sequence itself is passed in — summation over rankings is
    # commutative, so permuting which ranking comes first/second/... must not perturb the result.
    rankings = [["a", "b", "c"], ["c", "a"], ["b"]]
    baseline = rrf_fuse(rankings)
    for permuted in itertools.permutations(rankings):
        assert rrf_fuse(list(permuted)) == baseline


def test_rrf_fuse_empty_rankings_yields_empty_result():
    assert rrf_fuse([]) == []


def test_search_chunks_port_shape_via_a_fake_adapter():
    """Pins the `search_chunks(query, k, config) -> list[Chunk]` calling convention independent of
    any real adapter: a minimal fake that satisfies the `Retriever` protocol structurally."""

    class FakeRetriever:
        def search_chunks(self, query: str, k: int, config: RetrievalConfig) -> list[Chunk]:
            return [Chunk(chunk_id="c1", doc_id="doc-1", text=query)][:k]

    retriever: Retriever = FakeRetriever()
    results = retriever.search_chunks("hello", 1, RetrievalConfig())
    assert results == [Chunk(chunk_id="c1", doc_id="doc-1", text="hello")]
