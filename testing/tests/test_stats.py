"""P2, statistics: a score without an interval is an anecdote."""
from __future__ import annotations

import pytest

from evals.stats import (
    _bca_level,
    _percentile_bounds,
    bootstrap_ci,
    bootstrap_ci_bca,
    cluster_bootstrap_ci,
    cohen_kappa,
    detectable_effect,
    intervals_overlap,
    mcnemar_exact,
    mean_interval,
    paired_bootstrap_diff,
    paired_permutation_test,
    pass_all_k,
    pass_any_k,
    required_n,
    variance_components,
    wilson_interval,
)


def test_close_scores_have_overlapping_intervals():
    # 84% vs 81% on 100 items: you cannot yet claim one beats the other.
    assert intervals_overlap(wilson_interval(84, 100), wilson_interval(81, 100))


def test_clearly_different_scores_do_not_overlap():
    assert not intervals_overlap(wilson_interval(95, 100), wilson_interval(50, 100))


def test_interval_is_within_bounds():
    lo, hi = wilson_interval(84, 100)
    assert 0.0 <= lo < 0.84 < hi <= 1.0


def test_zero_trials_is_the_widest_interval_not_false_certainty():
    # No data cannot claim a 0% pass rate with certainty, so the honest interval is the whole range.
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_invalid_counts_raise():
    with pytest.raises(ValueError):
        wilson_interval(5, 3)  # successes > n is nonsense, not a silent 1.0+ rate


def test_kappa_pins_the_formula_on_an_asymmetric_case():
    # Golden value over UNEQUAL marginals (pa=0.3, pb=0.2), the way wilson is pinned. A mutant that
    # drops the (1-pa)(1-pb) term from the expected agreement reads 0.89 here, so this pin kills it.
    a = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    b = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert round(cohen_kappa(a, b), 4) == 0.7368


def test_kappa_is_one_when_both_raters_always_agree():
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_kappa_rejects_mismatched_or_empty_input():
    with pytest.raises(ValueError):
        cohen_kappa([1, 0], [1])
    with pytest.raises(ValueError):
        cohen_kappa([], [])


# --- mean interval (Wald) ---


def test_mean_interval_brackets_the_point_and_is_symmetric():
    point, lo, hi = mean_interval([0.7, 0.8, 0.9, 0.85, 0.75])
    assert lo < point < hi
    assert round(point - lo, 9) == round(hi - point, 9)  # Wald is symmetric by construction


def test_mean_interval_of_a_single_value_has_no_spread():
    assert mean_interval([0.5]) == (0.5, 0.5, 0.5)


def test_mean_interval_rejects_empty():
    with pytest.raises(ValueError):
        mean_interval([])


# --- bootstrap ---


def test_bootstrap_is_reproducible_under_a_fixed_seed():
    values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_bootstrap_brackets_the_point_estimate():
    values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]  # rate 0.7
    point, lo, hi = bootstrap_ci(values, seed=7)
    assert round(point, 1) == 0.7
    assert lo <= point <= hi
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_rejects_empty_or_zero_resamples():
    with pytest.raises(ValueError):
        bootstrap_ci([], seed=1)
    with pytest.raises(ValueError):
        bootstrap_ci([1, 0], seed=1, n_resamples=0)


def test_bootstrap_rejects_ci_outside_the_open_unit_interval():
    with pytest.raises(ValueError):
        bootstrap_ci([1, 0], seed=1, ci=0.0)
    with pytest.raises(ValueError):
        bootstrap_ci([1, 0], seed=1, ci=1.0)


def test_percentile_bounds_are_not_shifted_by_float_imprecision():
    # ci=0.90 makes (1 - ci) / 2 imprecise as a float (0.049999999999999996, not exactly 0.05). A
    # naive floor/ceil on that silently picks index 49 instead of the correct 50.
    lo, hi = _percentile_bounds(list(range(1000)), ci=0.90, n_resamples=1000)
    assert (lo, hi) == (50, 949)


# --- paired comparison ---


def test_paired_bootstrap_difference_straddles_zero_for_a_tiny_gap():
    # 84 vs 81 on 100 paired items, an 11 pair discordant split (7 where a passes and b
    # fails, 4 the other way, same structure as benchmark/dataset.py): the difference CI contains 0,
    # so you cannot call it.
    a = [1] * 77 + [1] * 7 + [0] * 4 + [0] * 12
    b = [1] * 77 + [0] * 7 + [1] * 4 + [0] * 12
    diff, lo, hi = paired_bootstrap_diff(a, b, seed=0xBEAC04)
    assert round(diff, 3) == 0.030
    assert lo <= 0.0 <= hi


def test_paired_permutation_is_reproducible_and_nonsignificant_for_a_tiny_gap():
    a = [1] * 77 + [1] * 7 + [0] * 4 + [0] * 12
    b = [1] * 77 + [0] * 7 + [1] * 4 + [0] * 12
    p1 = paired_permutation_test(a, b, seed=1, n_resamples=2000)
    p2 = paired_permutation_test(a, b, seed=1, n_resamples=2000)
    assert p1 == p2  # seeded
    assert p1 > 0.05  # the three point gap is inside the noise


def test_paired_permutation_finds_a_real_gap():
    a = [1] * 95 + [0] * 5
    b = [0] * 95 + [1] * 5  # every item flips: a huge, real difference
    assert paired_permutation_test(a, b, seed=1, n_resamples=2000) < 0.05


def test_paired_tests_reject_mismatched_input():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0], [1], seed=1)
    with pytest.raises(ValueError):
        paired_permutation_test([1, 0], [1], seed=1)


def test_paired_tests_reject_zero_resamples():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0], [1, 1], seed=1, n_resamples=0)
    with pytest.raises(ValueError):
        paired_permutation_test([1, 0], [1, 1], seed=1, n_resamples=0)


# --- McNemar ---


def test_mcnemar_golden_value_on_a_known_discordant_split():
    # b=7, c=4: exact two sided binomial tail. Pins the formula against a value computed by hand.
    assert round(mcnemar_exact(7, 4), 3) == 0.549


def test_mcnemar_is_one_when_there_are_no_discordant_pairs():
    assert mcnemar_exact(0, 0) == 1.0


def test_mcnemar_detects_a_lopsided_split():
    # 20 vs 0 discordant: every disagreement favours one system, the clearest possible signal.
    assert mcnemar_exact(20, 0) < 0.001


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 3)


# --- multiple trial reliability shapes ---


def test_pass_all_k_is_the_strict_bar():
    assert pass_all_k(10, 10) is True
    assert pass_all_k(9, 10) is False


def test_pass_any_k_is_the_optimistic_bar():
    assert pass_any_k(1, 10) is True
    assert pass_any_k(0, 10) is False


def test_reliability_shapes_reject_nonsense_counts():
    for fn in (pass_all_k, pass_any_k):
        with pytest.raises(ValueError):
            fn(11, 10)
        with pytest.raises(ValueError):
            fn(1, 0)


# --- BCa bootstrap (statistics crowding a boundary) ---


def test_bca_is_reproducible_under_a_fixed_seed():
    values = [1.0, 1.0, 1.0, 0.9, 1.0, 0.4, 1.0, 0.7, 1.0, 1.0]
    assert bootstrap_ci_bca(values, seed=7) == bootstrap_ci_bca(values, seed=7)


def test_bca_brackets_the_point_estimate():
    values = [1.0, 1.0, 1.0, 0.9, 1.0, 0.4, 1.0, 0.7, 1.0, 1.0]
    point, lo, hi = bootstrap_ci_bca(values, seed=7)
    assert lo <= point <= hi


def test_bca_corrects_the_percentile_interval_on_a_skewed_sample():
    # A metric pressed against its own ceiling (NDCG near one): the plain percentile
    # interval is biased and BCa shifts it. If the two coincide here, no correction
    # happened and the variant is not earning its name.
    values = [1.0] * 16 + [0.3, 0.5, 0.6, 0.8]
    plain = bootstrap_ci(values, seed=11, n_resamples=2000)
    corrected = bootstrap_ci_bca(values, seed=11, n_resamples=2000)
    assert plain[0] == corrected[0]  # same point estimate
    assert (plain[1], plain[2]) != (corrected[1], corrected[2])


def test_bca_collapses_to_the_point_when_the_data_has_no_spread():
    # All resamples equal the point, and the bias correction must not blow up on a
    # degenerate "no resample below the point" proportion.
    assert bootstrap_ci_bca([0.8] * 5, seed=3) == (0.8, 0.8, 0.8)


def test_bca_rejects_samples_too_small_to_jackknife():
    with pytest.raises(ValueError):
        bootstrap_ci_bca([1.0], seed=1)
    with pytest.raises(ValueError):
        bootstrap_ci_bca([], seed=1)


def test_bca_level_pins_to_the_extreme_when_the_adjustment_diverges():
    # A runaway acceleration pushes the denominator to/past zero, where the adjusted z
    # wraps around. The level must pin to the extreme it was heading for instead.
    assert _bca_level(0.975, z0=0.0, accel=1.0) == 1.0
    assert _bca_level(0.025, z0=0.0, accel=-1.0) == 0.0


def test_bca_rejects_bad_resamples_and_ci():
    with pytest.raises(ValueError):
        bootstrap_ci_bca([1.0, 0.0], seed=1, n_resamples=0)
    with pytest.raises(ValueError):
        bootstrap_ci_bca([1.0, 0.0], seed=1, ci=1.0)


def test_bca_ties_do_not_bias_a_symmetric_sample():
    # A fair 0/1 coin's resample means tie the point constantly. Counting ties at half
    # weight keeps the bias term near zero, so BCa agrees with the percentile interval
    # instead of reading the symmetric spread as median biased and collapsing its floor.
    values = [1, 0, 1, 0, 1, 0]
    plain = bootstrap_ci(values, seed=1, n_resamples=2000)
    corrected = bootstrap_ci_bca(values, seed=1, n_resamples=2000)
    assert corrected == plain


# --- cluster bootstrap (conversations with multiple turns) ---


def test_cluster_bootstrap_is_reproducible_under_a_fixed_seed():
    clusters = [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]
    assert cluster_bootstrap_ci(clusters, seed=7) == cluster_bootstrap_ci(clusters, seed=7)


def test_cluster_bootstrap_point_is_the_pooled_mean():
    clusters = [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
    point, lo, hi = cluster_bootstrap_ci(clusters, seed=7)
    assert round(point, 9) == 0.6  # 12 of 20 turns, pooled
    assert lo <= point <= hi


def test_cluster_bootstrap_is_wider_than_the_per_turn_lie():
    # Five conversations whose turns agree perfectly within each conversation: the
    # per turn bootstrap treats 20 correlated turns as 20 independent items and quotes
    # precision that is not there. Resampling whole conversations keeps the honesty.
    clusters = [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
    turns = [t for c in clusters for t in c]
    _, c_lo, c_hi = cluster_bootstrap_ci(clusters, seed=7, n_resamples=2000)
    _, t_lo, t_hi = bootstrap_ci(turns, seed=7, n_resamples=2000)
    assert (c_hi - c_lo) > (t_hi - t_lo)


def test_cluster_bootstrap_rejects_empty_input():
    with pytest.raises(ValueError):
        cluster_bootstrap_ci([], seed=1)
    with pytest.raises(ValueError):
        cluster_bootstrap_ci([[1, 0], []], seed=1)


# --- power: size the test for the effect, not the calendar ---


def test_required_n_grounds_the_high_stakes_heuristic():
    # The ~500-item figure for a high stakes surface is a power calculation, not a round
    # number someone liked: resolving a 5-point paired delta against sd 0.4 at the usual
    # alpha=0.05 / power=0.80 needs 503 items.
    assert required_n(effect=0.05, sd=0.4) == 503


def test_required_n_for_a_standardized_effect_of_one_is_eight():
    # The classic value you can check by hand: effect == sd gives ((1.96 + 0.842))^2 ~ 7.85 -> 8.
    assert required_n(effect=0.2, sd=0.2) == 8


def test_smaller_effects_need_more_items():
    assert required_n(effect=0.02, sd=0.4) > required_n(effect=0.05, sd=0.4)


def test_detectable_effect_names_what_the_suite_can_actually_see():
    # n=100 with sd 0.4 resolves ~11 points, nowhere near a 2-point regression: the suite
    # returns "no difference detectable here", which is not "no difference".
    assert round(detectable_effect(n=100, sd=0.4), 3) == 0.112


def test_sizing_round_trips():
    n = required_n(effect=0.05, sd=0.4)
    assert detectable_effect(n=n, sd=0.4) <= 0.05


def test_power_helpers_reject_nonsense():
    with pytest.raises(ValueError):
        required_n(effect=0.0, sd=0.4)
    with pytest.raises(ValueError):
        required_n(effect=0.05, sd=0.0)
    with pytest.raises(ValueError):
        required_n(effect=0.05, sd=0.4, power=1.0)
    with pytest.raises(ValueError):
        required_n(effect=0.05, sd=0.4, alpha=0.0)
    with pytest.raises(ValueError):
        detectable_effect(n=0, sd=0.4)
    with pytest.raises(ValueError):
        detectable_effect(n=100, sd=-0.4)


# --- variance decomposition (an agent is stochastic, so measure the spread) ---


def test_pure_model_stochasticity_is_all_within_item_variance():
    # Every item is a coin flip with the same mean: the spread is the model's own
    # trial to trial randomness, not item difficulty.
    within, between = variance_components([[0, 1], [1, 0], [0, 1]])
    assert within == 0.5
    assert between == 0.0


def test_pure_item_difficulty_is_all_between_item_variance():
    # Every item is deterministic but some items are simply hard: zero within, all between.
    within, between = variance_components([[1, 1], [0, 0]])
    assert within == 0.0
    assert between == 0.5


def test_equal_difficulty_noisy_items_have_zero_between_variance():
    # Two identical noisy coins: the raw variance of their per item means is inherited
    # within/k noise, not difficulty. The ANOVA correction subtracts it, so phantom
    # difficulty reads as zero instead of a positive "some items are harder" signal.
    within, between = variance_components([[1, 1, 0], [0, 0, 1]])
    assert within == pytest.approx(1 / 3)
    assert between == 0.0


def test_variance_components_are_named_fields():
    # Two same typed floats invite a transposed unpack. The NamedTuple names them.
    vc = variance_components([[1, 1], [0, 0]])
    assert vc.within == 0.0
    assert vc.between == 0.5


def test_variance_components_rejects_underpowered_input():
    with pytest.raises(ValueError):
        variance_components([])
    with pytest.raises(ValueError):
        variance_components([[1, 0]])  # between item needs at least two items
    with pytest.raises(ValueError):
        variance_components([[1, 0], [1]])  # within item needs at least two trials each
