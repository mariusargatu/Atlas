"""Statistics: make eval numbers honest. A score without an interval is
an anecdote. Named `stats` (not `statistics`) so it never shadows the stdlib module.

Pure stdlib by design: the PR lane is hermetic (no network, no scipy/numpy in the
install), so the interval and the resampling math live here in plain Python and run
under the same `task test` as everything else. The heavy frontier the statistics
article names (the NIST GLMM via statsmodels, PPI via ppi-py) is deliberately deferred
to a dev/prod extra rather than dragged into the hermetic core.

Every resampler takes an explicit integer `seed` and builds its own `random.Random`
from it, so the interval reproduces byte for byte when the seed is stamped in
provenance (principles 7/8). A bootstrap whose seed is not recorded does not ship.
"""
from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import Callable, NamedTuple, Sequence

_NORMAL = NormalDist()

_mean: Callable[[Sequence[float]], float] = lambda xs: sum(xs) / len(xs)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% confidence interval for a pass rate (Wilson, well behaved at the edges)."""
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"need 0 <= successes <= n, got successes={successes}, n={n}")
    if n == 0:
        return (0.0, 1.0)  # no data -> the only honest interval is the whole range, never (0, 0)
    return wilson_interval_from_rate(successes / n, n, z)


def wilson_interval_from_rate(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval from a proportion and a sample size, so a caller can pass an
    EFFECTIVE n (e.g. the number of independent clusters) instead of a raw success count.

    Same score interval as `wilson_interval` (which delegates here). It always brackets `rate` and
    stays honest at the boundary (rate 0 or 1), where a variance-based or bootstrap interval collapses
    to zero width, reading a handful of all-agreeing observations as certainty. That boundary honesty
    is why the eval report gates the overall rate on this at the case level, not on a bootstrap.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    if n <= 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denom
    half = (z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """If two metrics' intervals overlap, you cannot yet claim one beats the other.

    Expects plain (lo, hi) pairs, e.g. from `wilson_interval`. `mean_interval` and every
    bootstrap (`bootstrap_ci`, `bootstrap_ci_bca`, `cluster_bootstrap_ci`,
    `paired_bootstrap_diff`) return (point, lo, hi) triples, so unpack to (lo, hi) before
    calling this.
    """
    return not (a[1] < b[0] or b[1] < a[0])


def cohen_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    """Chance corrected agreement between two raters' binary labels.

    Raw percent agreement flatters, but kappa reveals. A judge that agrees with humans only at
    chance scores ~0 even when raw agreement looks high, the lying judge lesson.
    """
    n = len(rater_a)
    if n == 0 or len(rater_b) != n:
        raise ValueError("raters must be equal length and not empty")
    observed = sum(1 for x, y in zip(rater_a, rater_b) if x == y) / n
    pa = sum(rater_a) / n
    pb = sum(rater_b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def cohen_kappa_interval(
    rater_a: list[int], rater_b: list[int], *, z: float = 1.96
) -> tuple[float, float, float]:
    """A large-sample confidence interval for Cohen's kappa, returned as (point, lo, hi).

    Closed form and cheap, the same discipline as `wilson_interval` and `mean_interval`: the
    asymptotic standard error SE = sqrt(p_o (1 - p_o) / (n (1 - p_e)^2)) (Fleiss, Cohen & Everitt
    1969), a normal z either side, clamped to [-1, 1]. Pure `math`, no bootstrap, so it lives
    beside `cohen_kappa` in the same module the judge lane already imports.

    Why it exists: licensing a judge on the POINT kappa is the "gate on the point, not the floor"
    mistake this repo argues against everywhere else (`gate.py`, the benchmark study). The floor
    (`lo`) is the quantity a licence should read. Small n, or a margin pinned at one class, makes
    the normal approximation loose, exactly where a bootstrap on kappa is the honest tool (deferred
    to the statistics article) but the floor is still the right thing to gate on.
    """
    n = len(rater_a)
    if n == 0 or len(rater_b) != n:
        raise ValueError("raters must be equal length and not empty")
    point = cohen_kappa(rater_a, rater_b)
    observed = sum(1 for x, y in zip(rater_a, rater_b) if x == y) / n
    pa = sum(rater_a) / n
    pb = sum(rater_b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:  # a degenerate single-class margin has no spread to build an interval from
        return (point, point, point)
    se = math.sqrt(observed * (1 - observed) / (n * (1 - expected) ** 2))
    half = z * se
    return (point, max(-1.0, point - half), min(1.0, point + half))


def mean_interval(values: Sequence[float], z: float = 1.96) -> tuple[float, float, float]:
    """A 95% Wald interval for the mean of a bounded score (faithfulness, helpfulness).

    Closed form and cheap, allowed only where its assumptions hold: a largish n and a
    distribution that is not pinned against 0 or 1. When the score crowds a boundary or n
    is small, reach for the bootstrap instead, the same way a pass rate reaches for Wilson
    over Wald. Returns (point, lo, hi).

    ponytail: normal z, not a t critical, so no scipy dependency. For the n>=30 batches
    the eval lane reports over, the difference shows up only in the third decimal place.
    Use the bootstrap below when n is genuinely small.
    """
    n = len(values)
    if n == 0:
        raise ValueError("mean_interval needs at least one value")
    point = sum(values) / n
    if n == 1:
        return (point, point, point)  # no spread to estimate from one value, the point is all you have
    var = sum((v - point) ** 2 for v in values) / (n - 1)
    half = z * math.sqrt(var / n)
    return (point, point - half, point + half)


def _level_bounds(
    sorted_values: Sequence[float], lo_level: float, hi_level: float, n_resamples: int
) -> tuple[float, float]:
    """Index selection at two (possibly asymmetric) levels, shared by percentile and BCa.

    Rounds to 9 decimal places before floor/ceil: the levels are not always exact as
    floats (e.g. ci=0.90 gives alpha=0.049999999999999996), and the raw floor/ceil silently
    shifts the index by one at typical n_resamples. Rounding kills that noise while leaving
    already exact boundaries (the default ci=0.95 at n_resamples=1000/2000/10000) unchanged.
    Both indices are clamped into range because a BCa adjusted level can round to 0 or 1.
    """
    lo_idx = min(n_resamples - 1, max(0, math.floor(round(lo_level * n_resamples, 9))))
    hi_idx = min(n_resamples - 1, max(0, math.ceil(round(hi_level * n_resamples, 9)) - 1))
    return sorted_values[lo_idx], sorted_values[hi_idx]


def _percentile_bounds(sorted_values: Sequence[float], ci: float, n_resamples: int) -> tuple[float, float]:
    """Shared percentile index selection for bootstrap_ci/paired_bootstrap_diff."""
    alpha = (1 - ci) / 2
    return _level_bounds(sorted_values, alpha, 1 - alpha, n_resamples)


def _validate_resample_params(n_resamples: int, ci: float) -> None:
    """Shared bounds check for the bootstraps, the `_validate_trial_counts` of resampling."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci}")


def _sorted_resamples(
    vals: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    rng: random.Random,
    n_resamples: int,
) -> list[float]:
    """The one resample engine: n draws with replacement per resample, statistics sorted.

    Both bootstraps call this, so the seeded RNG consumption pattern (one `randrange`
    per element, n per resample) has a single definition and a stamped seed reproduces
    byte for byte across them, and the module docstring's promise lives here, once.
    """
    n = len(vals)
    return sorted(statistic([vals[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = _mean,
    *,
    seed: int,
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for any statistic with no clean closed form SE.

    Resample the values with replacement, recompute the statistic, repeat `n_resamples`
    times. The middle `ci` of that spread is the interval. Assumes nothing about a normal
    curve, which is the whole point for NDCG, a clustered success rate, or a skewed mean.
    Returns (point, lo, hi) where point is the statistic on the original sample.

    Resample at the unit the data is independent on: pass per item values for one shot
    cases, but per conversation aggregates for multiple turns, or the interval invents
    precision the correlated turns do not have.
    """
    if len(values) == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    _validate_resample_params(n_resamples, ci)
    rng = random.Random(seed)
    point = statistic(values)
    resampled = _sorted_resamples(values, statistic, rng, n_resamples)
    lo, hi = _percentile_bounds(resampled, ci, n_resamples)
    return (point, lo, hi)


def _bca_level(nominal: float, z0: float, accel: float) -> float:
    """Map a nominal percentile level to its BCa adjusted level.

    z0 recentres for median bias (the resamples not splitting 50/50 around the point).
    accel rescales for the statistic's variance changing with its own value, the thing
    that happens when a metric crowds a boundary. A runaway acceleration can push the
    denominator to zero or below, where the adjustment diverges, and the level is then pinned
    to the extreme it was heading for rather than allowed to wrap around. At the default
    ci=0.95 the pin is unreachable (a jackknife acceleration is bounded by ~1/6 and the
    pin needs ~0.18 there). It exists because `ci` and `n_resamples` are public API and
    extreme values (ci >= 0.999, very large resample counts) can reach it.
    """
    z = _NORMAL.inv_cdf(nominal)
    denom = 1 - accel * (z0 + z)
    if denom <= 0:
        return 1.0 if (z0 + z) > 0 else 0.0
    return _NORMAL.cdf(z0 + (z0 + z) / denom)


def bootstrap_ci_bca(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = _mean,
    *,
    seed: int,
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bias corrected and accelerated (BCa) bootstrap interval.

    The plain percentile interval assumes the resample spread sits symmetrically around
    the point, which quietly breaks when the statistic is skewed or pressed against its
    own boundary (NDCG pinned near one). BCa reads two corrections off the data itself:
    a bias term from where the point falls inside the resample distribution, and an
    acceleration term from the jackknife (leave one out) skew. Reach for it when the
    metric crowds a ceiling. Elsewhere `bootstrap_ci` is cheaper and agrees.
    Returns (point, lo, hi) like `bootstrap_ci`.
    """
    n = len(values)
    if n < 2:
        raise ValueError("bootstrap_ci_bca needs at least two values (the jackknife drops one)")
    _validate_resample_params(n_resamples, ci)
    vals = list(values)
    rng = random.Random(seed)
    point = statistic(vals)
    resampled = _sorted_resamples(vals, statistic, rng, n_resamples)
    # Bias correction: the share of resamples below the point, ties counted at half
    # weight, a discrete statistic (a 0/1 pass rate hitting the same mean again) ties the
    # point constantly, and counting ties as "not below" reads a symmetric spread as
    # median biased and drags the whole interval down. Clamped off the 0/1 edges so a
    # degenerate no spread sample cannot send inv_cdf to infinity.
    below = sum(1 for r in resampled if r < point) + 0.5 * sum(1 for r in resampled if r == point)
    prop = min(max(below / n_resamples, 1 / (n_resamples + 1)), n_resamples / (n_resamples + 1))
    z0 = _NORMAL.inv_cdf(prop)
    # Acceleration: the skew of the leave one out statistics, zero when they have no spread.
    jack = [statistic(vals[:i] + vals[i + 1 :]) for i in range(n)]
    jack_mean = _mean(jack)
    sum_sq = sum((jack_mean - j) ** 2 for j in jack)
    accel = sum((jack_mean - j) ** 3 for j in jack) / (6 * sum_sq**1.5) if sum_sq > 0 else 0.0
    alpha = (1 - ci) / 2
    lo, hi = _level_bounds(
        resampled, _bca_level(alpha, z0, accel), _bca_level(1 - alpha, z0, accel), n_resamples
    )
    return (point, lo, hi)


def cluster_bootstrap_ci(
    clusters: Sequence[Sequence[float]],
    statistic: Callable[[Sequence[float]], float] = _mean,
    *,
    seed: int,
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap that resamples whole clusters, for conversations of many turns.

    Turns inside one conversation are correlated, so the conversation is the independent
    unit: resample transcripts with replacement and recompute the statistic over the
    pooled turns of each resample. Resampling the turns individually treats correlated
    data as independent and invents precision you do not have, and this is the honest unit.
    `statistic` sees the pooled per turn values (default: their mean). Returns
    (point, lo, hi) where point is the statistic over all turns in the original sample.
    """
    if not clusters or any(len(c) == 0 for c in clusters):
        raise ValueError("cluster_bootstrap_ci needs at least one cluster and no empty clusters")
    return bootstrap_ci(
        [list(c) for c in clusters],
        lambda cs: statistic([v for c in cs for v in c]),
        seed=seed,
        n_resamples=n_resamples,
        ci=ci,
    )


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    seed: int,
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap CI for the per item difference mean(a) - mean(b) on paired data.

    A and B saw the SAME items, so resample item indices once and read both systems at
    that index: the shared resample keeps the pairing intact and the item difficulty
    variance cancels, which is exactly the variance that hides a real delta. The decision
    is whether this interval excludes 0, not whether the two marginal points look apart.
    Returns (diff, lo, hi). Delegates to `bootstrap_ci` over the zipped pairs so the two
    functions share one resample/percentile implementation.
    """
    if len(a) != len(b):
        raise ValueError("paired data must be equal length and not empty")
    return bootstrap_ci(
        list(zip(a, b)),
        lambda pairs: sum(x for x, _ in pairs) / len(pairs) - sum(y for _, y in pairs) / len(pairs),
        seed=seed,
        n_resamples=n_resamples,
        ci=ci,
    )


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    seed: int,
    n_resamples: int = 10000,
) -> float:
    """Two sided paired permutation test on the per item difference (a_i - b_i).

    Distribution free, no normality assumption: under the null that A and B are
    interchangeable, the sign of each per item difference is a coin flip, so the sampling
    distribution of the mean difference comes from flipping those signs at random. The
    p value is the share of sign flipped means at least as extreme as the observed one,
    with the +1 correction so a permutation test never reports an impossible p of exactly 0.
    """
    n = len(a)
    if n == 0 or len(b) != n:
        raise ValueError("paired data must be equal length and not empty")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    diffs = [a[i] - b[i] for i in range(n)]
    observed = abs(sum(diffs) / n)
    rng = random.Random(seed)
    at_least_as_extreme = 0
    for _ in range(n_resamples):
        flipped = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(flipped) >= observed - 1e-12:
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (n_resamples + 1)


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two sided McNemar test for paired binary outcomes (A vs B, pass/fail per item).

    Reads ONLY the discordant pairs: `b` items A passed and B failed, `c` items B passed
    and A failed. The concordant cells (both pass, both fail) carry no information about
    which system is better and are correctly ignored. Under the null the discordant pairs
    split 50/50, so the p value is an exact two sided binomial tail on min(b, c) out of
    b + c. Returns 1.0 when there are no discordant pairs (nothing to tell them apart).
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must not be negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _power_zs(alpha: float, power: float) -> float:
    """The (z_{1-alpha/2} + z_{power}) factor shared by the two sizing directions."""
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0 < power < 1:
        raise ValueError(f"power must be in (0, 1), got {power}")
    return _NORMAL.inv_cdf(1 - alpha / 2) + _NORMAL.inv_cdf(power)


def required_n(effect: float, sd: float, *, alpha: float = 0.05, power: float = 0.80) -> int:
    """Items needed for a paired test to reliably detect `effect` against noise `sd`.

    Size the set to the smallest effect that would change a decision, not to a round
    number someone liked: an eval with too few items does not return "no difference", it
    returns "no difference detectable here", and an underpowered suite fails by finding
    nothing. `sd` is the standard deviation of the per item paired differences. The
    normal approximation formula n = ((z_a + z_b) * sd / effect)^2, rounded up.
    """
    if effect <= 0:
        raise ValueError(f"effect must be positive, got {effect}")
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")
    return math.ceil((_power_zs(alpha, power) * sd / effect) ** 2)


def detectable_effect(n: int, sd: float, *, alpha: float = 0.05, power: float = 0.80) -> float:
    """The smallest paired effect a suite of `n` items can reliably see: `required_n` inverted.

    The honest answer to "our suite of 100 items found nothing" is this number: if it is far
    larger than the regression you care about, the silence is blindness, not good news.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if sd <= 0:
        raise ValueError(f"sd must be positive, got {sd}")
    return _power_zs(alpha, power) * sd / math.sqrt(n)


class VarianceComponents(NamedTuple):
    """Two same typed variances with no ordering invariant: named so a call site cannot
    silently transpose model stochasticity with item difficulty."""

    within: float
    between: float


def variance_components(trials_per_item: Sequence[Sequence[float]]) -> VarianceComponents:
    """Split scores of k trials per item into (within_item, between_item) variance.

    An agent above temperature zero yields different outputs on the same input, so a
    naive average smears two different spreads together: within item variance is the
    model's own stochasticity on a fixed input (the same question answered well on
    Tuesday and badly on Wednesday), between item variance is genuine difficulty (some
    questions harder for everyone). They answer different questions and a release call
    needs both. Within is the mean of the per item sample variances. Between starts from
    the sample variance of the per item means, then subtracts the within item noise those
    means inherit (a k trial mean still carries within/k of the model's own randomness,
    the one way ANOVA correction) and floors at zero. The raw variance of means would
    report phantom difficulty for equally hard noisy items.
    """
    items = [list(t) for t in trials_per_item]
    if len(items) < 2:
        raise ValueError("variance_components needs at least two items to see between-item variance")
    if any(len(t) < 2 for t in items):
        raise ValueError("variance_components needs at least two trials per item to see within-item variance")
    means = [_mean(t) for t in items]
    within = _mean([sum((x - m) ** 2 for x in t) / (len(t) - 1) for t, m in zip(items, means)])
    grand = _mean(means)
    raw_between = sum((m - grand) ** 2 for m in means) / (len(means) - 1)
    noise_in_means = within * _mean([1 / len(t) for t in items])
    between = max(0.0, raw_between - noise_in_means)
    return VarianceComponents(within=within, between=between)


def _validate_trial_counts(passes: int, k: int) -> None:
    if k < 1 or passes < 0 or passes > k:
        raise ValueError(f"need 0 <= passes <= k and k >= 1, got passes={passes}, k={k}")


def pass_all_k(passes: int, k: int) -> bool:
    """Strict reliability: every one of the k trials passed. The bar for irreversible behaviour."""
    _validate_trial_counts(passes, k)
    return passes == k


def pass_any_k(passes: int, k: int) -> bool:
    """Optimistic capability: at least one of the k trials passed. Measures capability, not reliability."""
    _validate_trial_counts(passes, k)
    return passes >= 1


__all__ = [
    "VarianceComponents",
    "bootstrap_ci",
    "bootstrap_ci_bca",
    "cluster_bootstrap_ci",
    "cohen_kappa",
    "cohen_kappa_interval",
    "detectable_effect",
    "intervals_overlap",
    "mcnemar_exact",
    "mean_interval",
    "paired_bootstrap_diff",
    "paired_permutation_test",
    "pass_all_k",
    "pass_any_k",
    "required_n",
    "variance_components",
    "wilson_interval",
    "wilson_interval_from_rate",
]
