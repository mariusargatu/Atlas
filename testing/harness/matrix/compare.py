"""Paired comparison wiring: `quality.stats`'s `paired_bootstrap_diff` (a 95% CI on every reported
delta) plus `holm_bonferroni` (family wise correction across every pairwise comparison run together)
and `paired_permutation_test` (the p value Holm corrects), carried forward from SP7 rather than a
third invented significance recipe -- the digest's own instruction: "carry SP7's honesty forward, do
not re earn it."

Every comparison is PAIRED: the two components compared must have scored the exact SAME case
sequence, in the SAME order (research 14's own citation of Miller 2024 and Smucker et al. 2007, both
require paired per query differences, never independent sample stats over a shared query set). This
module trusts its caller for that; it has no case identity of its own to check against, only score
lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

from quality.stats import holm_bonferroni, paired_bootstrap_diff, paired_permutation_test


@dataclass(frozen=True)
class PairwiseDelta:
    """`a` scored higher than `b` when `diff > 0` (`paired_bootstrap_diff`'s own convention:
    `mean(a) - mean(b)`). `p_value_holm` is the ONE number a caller should read for "is this real
    after correcting for every OTHER pair compared in the same family" -- `p_value` alone
    corrects too little whenever more than one comparison runs together."""

    a: str
    b: str
    diff: float
    ci_lo: float
    ci_hi: float
    p_value: float
    p_value_holm: float


def compare_components(
    scores: Mapping[str, Sequence[float]], *, seed: int, n_resamples: int = 2000
) -> list[PairwiseDelta]:
    """Every pairwise comparison among `scores` (component_id -> per case score list, every list over
    the SAME case order). Pairs are built from `sorted(scores)`, never dict/insertion order, so the
    comparison family (and therefore the Holm adjustment, which depends on the WHOLE family) is
    reproducible independent of how the caller happened to build the `scores` mapping.

    Returns `[]` for zero or one component (nothing to compare, the honest empty case, never a
    manufactured comparison against something absent).
    """
    names = sorted(scores)
    pairs = list(combinations(names, 2))
    if not pairs:
        return []
    raw_p = [paired_permutation_test(scores[a], scores[b], seed=seed) for a, b in pairs]
    adjusted = holm_bonferroni(raw_p)
    out: list[PairwiseDelta] = []
    for (a, b), p, p_holm in zip(pairs, raw_p, adjusted):
        diff, lo, hi = paired_bootstrap_diff(scores[a], scores[b], seed=seed, n_resamples=n_resamples)
        out.append(PairwiseDelta(a=a, b=b, diff=diff, ci_lo=lo, ci_hi=hi, p_value=p, p_value_holm=p_holm))
    return out


def delta_to_dict(delta: PairwiseDelta) -> dict:
    return {
        "a": delta.a,
        "b": delta.b,
        "diff": delta.diff,
        "ci_lo": delta.ci_lo,
        "ci_hi": delta.ci_hi,
        "p_value": delta.p_value,
        "p_value_holm": delta.p_value_holm,
    }


__all__ = ["PairwiseDelta", "compare_components", "delta_to_dict"]
