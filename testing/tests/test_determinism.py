"""Determinism kit + canonicalization tests: the meta tests the whole suite rests on.

Pure stdlib, zero network: they prove the cassette key is stable, key order independent,
ignores unrelated kwargs, keeps money exact, and that injected nondeterminism is *caught*
by the digest (the negative meta test from 03-test-architecture.md).
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from canonical import canonical_json, digest, request_digest, serialize_tool_result
from determinism import FrozenClock, IdFactory, SeededRng, SpanSequence, fixture_kit


# --- canonicalization is stable and order independent ---

def test_canonical_json_is_key_order_independent():
    a = {"b": 1, "a": 2, "c": {"y": 1, "x": 2}}
    b = {"c": {"x": 2, "y": 1}, "a": 2, "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_digest_is_stable_across_runs():
    payload = {"plan": "legacy", "price": Decimal("39.00"), "ids": [3, 1, 2]}
    assert digest(payload) == digest(dict(payload))


def test_money_is_exact_not_float():
    assert serialize_tool_result({"bill": Decimal("39.10")}) == '{"bill":"D:39.1"}'
    # value equal money hashes identically regardless of scale (the content addressed key contract)
    assert serialize_tool_result({"bill": Decimal("39.10")}) == serialize_tool_result({"bill": Decimal("39.1000")})


# --- the request digest uses an allow list ---

def test_request_digest_ignores_unlisted_kwargs():
    base = {"model_id": "claude", "messages": [{"role": "user", "content": "hi"}], "temperature": 0}
    noisy = dict(base, run_manager="<obj>", callbacks=["x"], some_kwarg=42)
    assert request_digest(base) == request_digest(noisy)


def test_request_digest_changes_when_a_listed_field_changes():
    base = {"model_id": "claude", "messages": [{"role": "user", "content": "hi"}]}
    changed = {"model_id": "claude", "messages": [{"role": "user", "content": "HELLO"}]}
    assert request_digest(base) != request_digest(changed)


# --- the negative meta test: injected nondeterminism is caught ---

def test_injected_nondeterminism_is_caught_by_the_digest():
    # A stray wall clock epoch that did NOT go through the frozen clock changes the bytes.
    a = serialize_tool_result({"bill": Decimal("39.00")})
    b = serialize_tool_result({"bill": Decimal("39.00"), "_fetched_at_epoch": 1718456400.137})
    assert a != b  # the digest notices the unpinned value


def test_canonical_serializer_normalizes_a_pinned_timestamp():
    clock = FrozenClock(datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc))
    one = serialize_tool_result({"due": clock.now()})
    two = serialize_tool_result({"due": clock.now()})
    assert one == two == '{"due":"2026-06-15T12:00:00+00:00"}'


# --- the kit elements are deterministic ---

def test_frozen_clock_is_constant_and_advanceable():
    clock = FrozenClock(datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc))
    assert clock.now() == clock.now()
    clock.advance(3600)
    assert clock.now() == datetime(2026, 6, 15, 13, 0, tzinfo=timezone.utc)


def test_seeded_rng_repeats():
    assert [SeededRng(0).random() for _ in range(3)] == [SeededRng(0).random() for _ in range(3)]


def test_id_factory_is_monotonic_and_deterministic():
    f = IdFactory("act")
    assert [f.next() for _ in range(3)] == ["act-000001", "act-000002", "act-000003"]


def test_span_sequence_orders_not_the_clock():
    s = SpanSequence()
    assert [s.next() for _ in range(3)] == [0, 1, 2]


def test_fixture_kit_is_reproducible():
    assert fixture_kit().clock.now() == fixture_kit().clock.now()
