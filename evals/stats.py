"""Aggregation and paired comparison over question level scores.

`average_by_kind` raises the moment an answerable kind carries a non-finite value,
naming the kind, rather than letting it silently poison a headline figure. It is not on
the recording path, though: every published figure is averaged by
`evals.report.retrieval_summary`, which excludes non-finite values instead. `paired_
comparison` below is the function this module contributes to that path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from atlas.contracts import QuestionName
from atlas.corpus.gold import ANSWERABLE_KINDS

_RESAMPLES = 2000
_Z_95 = 1.96
# Worst-case std of a per-question *difference* between two [0, 1] metrics, which lives
# in [-1, 1]. Not 0.5, the worst case for a single [0, 1] mean: that understates the bar
# a difference between two arms must clear.
_WORST_CASE_DIFFERENCE_STD = 1.0


def average_by_kind(scored: Mapping[tuple[QuestionName, str], float]) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for (_, kind), value in scored.items():
        groups.setdefault(kind, []).append(value)
    result: dict[str, float] = {}
    for kind, values in groups.items():
        if kind in ANSWERABLE_KINDS and any(not math.isfinite(v) for v in values):
            raise ValueError(f"{kind} carries a non finite value, which is a resolver bug, not a score")
        result[kind] = sum(values) / len(values)
    return result


def resample_interval(values: Sequence[float], seed: int, resamples: int = _RESAMPLES) -> tuple[float, float]:
    if not values:
        raise ValueError("resample_interval needs at least one value")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("resample_interval refuses a non finite value")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(resamples)])
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


@dataclass(frozen=True, slots=True)
class PairedComparison:
    low: float
    high: float
    questions: int


def paired_comparison(
    before: Mapping[QuestionName, float], after: Mapping[QuestionName, float], seed: int
) -> PairedComparison:
    """Pairs on question name, never on list length: two different question sets
    of the same size would otherwise pair silently and produce a tight interval
    about nothing."""
    if before.keys() != after.keys():
        raise ValueError("paired_comparison requires before and after to score exactly the same questions")
    names = sorted(before, key=str)
    diffs = [after[name] - before[name] for name in names]
    low, high = resample_interval(diffs, seed)
    return PairedComparison(low=low, high=high, questions=len(names))


def smallest_resolvable_difference(questions: int) -> float:
    """The gap two arms must show before the difference is a result rather than noise.

    Deliberately blunt: assumes the per-question difference is as variable as a bounded
    difference can be. A real paired bootstrap interval (`paired_comparison`) is usually
    much tighter, since pairing removes variance the two arms share. A gap under this bar
    is not shown to be noise; it's only not yet shown to be signal by the cheapest
    possible argument.
    """
    if questions < 1:
        raise ValueError("smallest_resolvable_difference needs at least one question")
    return _Z_95 * _WORST_CASE_DIFFERENCE_STD / math.sqrt(questions)
