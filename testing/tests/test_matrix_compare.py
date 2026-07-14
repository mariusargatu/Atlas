"""`matrix.compare`, hermetic: the paired significance wiring (`quality.stats`'s
`paired_bootstrap_diff`/`paired_permutation_test`/`holm_bonferroni`), never a second, reinvented
significance recipe. Pure stdlib arithmetic, no network, no wall clock.
"""
from __future__ import annotations

import pytest

from quality.stats import holm_bonferroni, paired_bootstrap_diff, paired_permutation_test

from matrix.compare import PairwiseDelta, compare_components, delta_to_dict

_SEED = 20260721


def test_zero_or_one_component_returns_no_comparisons():
    assert compare_components({}, seed=_SEED) == []
    assert compare_components({"only": [0.1, 0.2, 0.3]}, seed=_SEED) == []


def test_two_components_produce_exactly_one_pairwise_delta():
    scores = {"a": [1.0, 1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0, 0.0]}
    deltas = compare_components(scores, seed=_SEED)
    assert len(deltas) == 1
    assert isinstance(deltas[0], PairwiseDelta)
    assert deltas[0].a == "a" and deltas[0].b == "b"


def test_pairs_are_built_from_sorted_names_never_dict_order():
    scores = {"zebra": [0.5, 0.5], "alpha": [0.5, 0.5], "mid": [0.5, 0.5]}
    deltas = compare_components(scores, seed=_SEED)
    pairs = [(d.a, d.b) for d in deltas]
    assert pairs == [("alpha", "mid"), ("alpha", "zebra"), ("mid", "zebra")]


def test_delta_matches_the_real_paired_bootstrap_diff_directly():
    scores = {"a": [0.9, 0.8, 0.7, 0.6], "b": [0.1, 0.2, 0.3, 0.4]}
    deltas = compare_components(scores, seed=_SEED)
    diff, lo, hi = paired_bootstrap_diff(scores["a"], scores["b"], seed=_SEED, n_resamples=2000)
    assert deltas[0].diff == diff
    assert deltas[0].ci_lo == lo
    assert deltas[0].ci_hi == hi


def test_p_value_holm_matches_holm_bonferroni_over_the_whole_family():
    scores = {"a": [1.0, 0.0, 1.0, 0.0], "b": [0.0, 1.0, 0.0, 1.0], "c": [1.0, 1.0, 1.0, 1.0]}
    deltas = compare_components(scores, seed=_SEED)
    names = sorted(scores)
    from itertools import combinations

    pairs = list(combinations(names, 2))
    raw_p = [paired_permutation_test(scores[a], scores[b], seed=_SEED) for a, b in pairs]
    expected_adjusted = holm_bonferroni(raw_p)
    assert [d.p_value_holm for d in deltas] == expected_adjusted


def test_holm_adjustment_is_never_smaller_than_the_raw_p_value():
    scores = {"a": [1.0, 0.0, 1.0, 0.0, 1.0], "b": [0.0, 1.0, 0.0, 1.0, 0.0], "c": [0.5, 0.5, 0.5, 0.5, 0.5]}
    for delta in compare_components(scores, seed=_SEED):
        assert delta.p_value_holm >= delta.p_value


def test_paired_score_lists_of_unequal_length_raise():
    with pytest.raises(ValueError):
        compare_components({"a": [1.0, 2.0], "b": [1.0]}, seed=_SEED)


def test_delta_to_dict_is_json_plain():
    import json

    scores = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    delta = compare_components(scores, seed=_SEED)[0]
    json.dumps(delta_to_dict(delta))  # never raises: every field is a JSON scalar
