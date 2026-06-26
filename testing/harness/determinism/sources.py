"""The determinism kit: every nondeterministic source pinned behind an injectable fixture.

Replay pins the model. This pins everything else (principle 7): the clock, the RNG, id/reference
generation, span ordering. The CI lane wires these frozen fixtures, and dev/prod inject their own
(a real clock, a system RNG) at the same call sites, since the consumers only need `.now()`,
`.next()`, etc. Duck typing, no base class required.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence


class FrozenClock:
    """A clock pinned to a fixture instant, advanceable for TTL/timeout/freshness tests."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FrozenClock requires a timezone-aware instant")
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class SeededRng:
    """A reproducible RNG so a CI run repeats exactly."""

    def __init__(self, seed: int = 0) -> None:
        self._r = random.Random(seed)

    def random(self) -> float:
        return self._r.random()

    def choice(self, seq: Sequence):
        return self._r.choice(seq)


class IdFactory:
    """Deterministic, monotonic ids: action references, idempotency keys, checkpoint ids."""

    def __init__(self, prefix: str = "id") -> None:
        self._prefix = prefix
        self._counter = itertools.count(1)

    def next(self) -> str:
        return f"{self._prefix}-{next(self._counter):06d}"


class SpanSequence:
    """A monotonic counter incremented at span open in deterministic graph order.

    Traces order by this, NEVER by the frozen clock (all spans would tie).
    """

    def __init__(self) -> None:
        self._counter = itertools.count(0)

    def next(self) -> int:
        return next(self._counter)


@dataclass
class DeterminismKit:
    """The four pinned sources, bundled so CI seeds determinism in one place."""

    clock: FrozenClock
    rng: SeededRng
    ids: IdFactory
    spans: SpanSequence


def fixture_kit(instant: str = "2026-06-15T12:00:00+00:00", seed: int = 0) -> DeterminismKit:
    """The default CI kit: a fixed instant, seed 0, fresh counters."""
    return DeterminismKit(
        clock=FrozenClock(datetime.fromisoformat(instant)),
        rng=SeededRng(seed),
        ids=IdFactory(),
        spans=SpanSequence(),
    )


__all__ = ["DeterminismKit", "FrozenClock", "IdFactory", "SeededRng", "SpanSequence", "fixture_kit"]
