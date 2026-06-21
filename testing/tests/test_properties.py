"""Property based + metamorphic tests for the pure cores (stats, canonical).

Where a function is pure math with invariants (a confidence interval, a chance corrected score, a
canonical digest), example tests check a handful of points but leave the spaces between them dark.
Mutation testing found exactly that gap: a tampered Wilson coefficient survived because no test
pinned the interval's *value*, only loose bounds. These tests assert the invariants over a whole
generated space (hypothesis, derandomized in conftest so the lane stays reproducible), plus a few
golden values that nail the arithmetic itself.

The canonical block is *metamorphic*: it asserts relations between inputs and outputs (reorder the
keys → identical digest, add an unlisted field → identical digest, change a listed field → digest
moves) rather than memorised outputs. Those relations ARE the cassette key contract.
"""
from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from determinism.canonical import REQUEST_ALLOW, canonical_json, digest, request_digest
from evals.stats import cohen_kappa, wilson_interval

# JSON ish values with str keys (canonical sorts dict items by key, so keys must be comparable).
_json = st.recursive(
    st.none() | st.booleans() | st.integers(-1000, 1000) | st.text(max_size=8),
    lambda children: st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=6), children, max_size=4),
    max_leaves=12,
)


# ---- stats: wilson_interval ----

@given(n=st.integers(0, 500), frac=st.floats(0, 1))
def test_wilson_interval_stays_within_unit_bounds(n, frac):
    successes = round(frac * n)
    lo, hi = wilson_interval(successes, n)
    assert 0.0 <= lo <= hi <= 1.0


@given(n=st.integers(1, 500), frac=st.floats(0, 1))
def test_wilson_interval_brackets_the_observed_rate(n, frac):
    successes = round(frac * n)
    lo, hi = wilson_interval(successes, n)
    p = successes / n
    eps = 1e-9  # the interval contains p up to float error (at p=0, lo is ~1e-18, not exactly 0)
    assert lo - eps <= p <= hi + eps  # the Wilson interval always contains the point estimate


def test_wilson_interval_golden_values():
    # Pins the arithmetic itself: a tampered coefficient shifts these and is caught (mutation gap).
    lo, hi = wilson_interval(8, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.490, 0.943)
    lo2, hi2 = wilson_interval(50, 100)
    assert (round(lo2, 3), round(hi2, 3)) == (0.404, 0.596)


# ---- stats: cohen_kappa ----

@given(labels=st.lists(st.sampled_from([0, 1]), min_size=1, max_size=60))
def test_kappa_is_symmetric_and_perfect_self_agreement_is_one(labels):
    assert cohen_kappa(labels, labels) == 1.0
    other = [1 - x for x in labels]
    assert math.isclose(cohen_kappa(labels, other), cohen_kappa(other, labels))


@given(
    a=st.lists(st.sampled_from([0, 1]), min_size=2, max_size=60),
    data=st.data(),
)
def test_kappa_never_exceeds_one_in_magnitude(a, data):
    b = data.draw(st.lists(st.sampled_from([0, 1]), min_size=len(a), max_size=len(a)))
    assert -1.0 - 1e-9 <= cohen_kappa(a, b) <= 1.0 + 1e-9


# ---- canonical: metamorphic relations (the cassette key contract) ----

@given(d=st.dictionaries(st.text(max_size=6), _json, max_size=8))
def test_digest_is_independent_of_key_order(d):
    reordered = dict(reversed(list(d.items())))
    assert digest(d) == digest(reordered)
    assert canonical_json(d) == canonical_json(reordered)


@given(d=st.dictionaries(st.text(max_size=6), _json, max_size=8))
def test_digest_is_idempotent_across_repeated_calls(d):
    assert digest(d) == digest(d)


@given(
    messages=st.lists(st.dictionaries(st.text(max_size=4), _json, max_size=3), max_size=4),
    noise_key=st.text(min_size=1, max_size=10).filter(lambda k: k not in REQUEST_ALLOW),
    noise_val=_json,
)
def test_unlisted_request_fields_never_move_the_digest(messages, noise_key, noise_val):
    base = {"model_id": "m", "messages": messages}
    noisy = {**base, noise_key: noise_val}
    assert request_digest(base) == request_digest(noisy)  # allow list shields the key from noise


@given(
    messages=st.lists(st.dictionaries(st.text(max_size=4), _json, max_size=3), min_size=1, max_size=4),
    field=st.sampled_from(["temperature", "top_p", "max_tokens", "tool_choice"]),
    a=st.integers(0, 5),
    b=st.integers(6, 11),
)
def test_changing_a_listed_field_moves_the_digest(messages, field, a, b):
    base = {"model_id": "m", "messages": messages, field: a}
    changed = {"model_id": "m", "messages": messages, field: b}
    assert request_digest(base) != request_digest(changed)  # listed fields DO shape the key
