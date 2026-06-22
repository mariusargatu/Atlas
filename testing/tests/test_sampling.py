"""Review sampling: which sessions reach the scarcest tier.

Flagged sessions (low-faithfulness, cost overruns, abuse) have priority; a seeded random slice fills
the remaining daily capacity for representativeness. Deterministic given the seed, so the queue is
reproducible in CI, and flagged sessions beyond capacity are reported dropped, never silently lost.
"""
from __future__ import annotations

import pytest

from evals.monitor.sampling import DEFAULT_CAPACITY, build_review_queue

_WINDOW = tuple(f"s{i:03d}" for i in range(100))


def test_flagged_are_always_reviewed_and_random_fills_the_rest():
    q = build_review_queue(_WINDOW, ["s001", "s002"], capacity=10, seed=7)
    assert set(q.flagged) == {"s001", "s002"}
    assert len(q.random) == 8 and len(q.to_review) == 10 and not q.over_capacity


def test_random_is_drawn_only_from_non_flagged():
    q = build_review_queue(_WINDOW, ["s001", "s002"], capacity=10, seed=7)
    assert set(q.random).isdisjoint({"s001", "s002"}) and set(q.random) <= set(_WINDOW)


def test_the_seed_makes_the_random_slice_reproducible():
    a = build_review_queue(_WINDOW, ["s001"], capacity=10, seed=42)
    b = build_review_queue(_WINDOW, ["s001"], capacity=10, seed=42)
    assert a.random == b.random


def test_a_different_seed_moves_the_slice():
    a = build_review_queue(_WINDOW, ["s001"], capacity=10, seed=1)
    b = build_review_queue(_WINDOW, ["s001"], capacity=10, seed=2)
    assert a.random != b.random


def test_flagged_beyond_capacity_are_dropped_not_lost():
    flagged = ["s010", "s011", "s012", "s013", "s014"]
    q = build_review_queue(_WINDOW, flagged, capacity=3, seed=7)
    assert len(q.flagged) == 3 and len(q.dropped_flagged) == 2 and q.over_capacity
    assert q.random == () and set(q.flagged) | set(q.dropped_flagged) == set(flagged)


def test_flagged_and_random_never_overlap():
    q = build_review_queue(_WINDOW, ["s001", "s050"], capacity=20, seed=3)
    assert set(q.flagged).isdisjoint(set(q.random))


def test_a_flag_for_an_unknown_session_is_an_error():
    with pytest.raises(ValueError, match="not in the session window"):
        build_review_queue(_WINDOW, ["s001", "nope"], capacity=5, seed=1)


def test_negative_capacity_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        build_review_queue(_WINDOW, [], capacity=-1, seed=1)


def test_capacity_larger_than_the_window_takes_everything():
    small = ("a", "b", "c")
    q = build_review_queue(small, ["a"], capacity=100, seed=1)
    assert set(q.to_review) == set(small)


def test_zero_capacity_drops_every_flag():
    q = build_review_queue(_WINDOW, ["s001"], capacity=0, seed=1)
    assert q.flagged == () and q.dropped_flagged == ("s001",) and q.over_capacity


def test_omitting_capacity_uses_the_default_daily_budget():
    q = build_review_queue(_WINDOW, ["s001"], seed=1)
    assert len(q.to_review) == min(DEFAULT_CAPACITY, len(_WINDOW)) and not q.over_capacity
