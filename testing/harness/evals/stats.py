"""Statistics: make eval numbers honest. A score without an interval is
an anecdote. Named `stats` (not `statistics`) so it never shadows the stdlib module.
"""
from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """A 95% confidence interval for a pass rate (Wilson, well behaved at the edges)."""
    if n < 0 or successes < 0 or successes > n:
        raise ValueError(f"need 0 <= successes <= n, got successes={successes}, n={n}")
    if n == 0:
        return (0.0, 1.0)  # no data -> the only honest interval is the whole range, never (0, 0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """If two metrics' intervals overlap, you cannot yet claim one beats the other."""
    return not (a[1] < b[0] or b[1] < a[0])


def cohen_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    """Chance corrected agreement between two raters' binary labels.

    Raw percent agreement flatters; kappa reveals. A judge that agrees with humans only at
    chance scores ~0 even when raw agreement looks high, the lying judge lesson.
    """
    n = len(rater_a)
    if n == 0 or len(rater_b) != n:
        raise ValueError("raters must be equal-length and non-empty")
    observed = sum(1 for x, y in zip(rater_a, rater_b) if x == y) / n
    pa = sum(rater_a) / n
    pb = sum(rater_b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


__all__ = ["cohen_kappa", "intervals_overlap", "wilson_interval"]
