"""Sample the sessions that reach human review: step one of the living-dataset loop.

The human tier does not scale, so production cannot review every session; it chooses from two
streams. Flagged sessions the monitor already caught are the failures you know about; a seeded
random slice of the rest catches the failures you are not looking for. Flagged sessions go first,
random fills whatever capacity remains, and any flagged session beyond capacity is reported
dropped rather than silently lost. Deterministic given the seed, so the queue replays byte for
byte in CI.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from random import Random

# daily human-review budget; tune per deployment
DEFAULT_CAPACITY = 500


@dataclass(frozen=True)
class ReviewQueue:
    flagged: tuple[str, ...]          # flagged ids selected for review (priority), sorted
    random: tuple[str, ...]           # seeded random-slice ids filling the remaining capacity, sorted
    dropped_flagged: tuple[str, ...]  # flagged ids that did not fit the capacity (over budget), sorted

    @property
    def to_review(self) -> tuple[str, ...]:
        """The full queue a human works through: flagged first, then the random slice."""
        return self.flagged + self.random

    @property
    def over_capacity(self) -> bool:
        """True when flagged sessions alone exceeded the daily capacity."""
        return bool(self.dropped_flagged)


def build_review_queue(
    all_ids: Iterable[str],
    flagged_ids: Iterable[str],
    *,
    capacity: int = DEFAULT_CAPACITY,
    seed: int,
) -> ReviewQueue:
    """Select the day's human-review queue from a window of session ids. Deterministic given the seed.

    Flagged sessions have priority; the random slice fills whatever capacity is left, drawn only
    from the non-flagged sessions so the streams never overlap. When flagged alone exceeds
    capacity, the earliest (sorted) ids fit and the rest are reported dropped, never silently
    discarded. Sorting inputs before sampling keeps the queue reproducible in CI.
    """
    if capacity < 0:
        raise ValueError(f"capacity must be non-negative, got {capacity}")
    universe = set(all_ids)
    flagged = set(flagged_ids)
    unknown = flagged - universe
    if unknown:
        raise ValueError(f"flagged ids not in the session window: {sorted(unknown)}")

    flagged_sorted = sorted(flagged)
    if len(flagged_sorted) > capacity:
        return ReviewQueue(
            flagged=tuple(flagged_sorted[:capacity]),
            random=(),
            dropped_flagged=tuple(flagged_sorted[capacity:]),
        )

    remaining = capacity - len(flagged_sorted)
    pool = sorted(universe - flagged)
    picked = Random(seed).sample(pool, min(remaining, len(pool)))  # seeded + sorted pool -> reproducible
    return ReviewQueue(flagged=tuple(flagged_sorted), random=tuple(sorted(picked)), dropped_flagged=())


__all__ = ["DEFAULT_CAPACITY", "ReviewQueue", "build_review_queue"]
