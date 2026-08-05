"""Concurrent scoring must keep every score with the question it was measured on.

deepeval's `a_measure` writes `self.score` on the metric and then returns it, so
coroutines sharing one metric instance interleave between the write and the return and
hand back each other's scores. The output still looks entirely plausible: only the
pairing is gone, which is what the agreement and kappa columns are computed from.
"""

from __future__ import annotations

import asyncio
import random

from evals.deepeval_suite import _measure_all

CASES = [float(i) for i in range(40)]


class _EchoScorer:
    """Returns the case it was handed the way a real metric returns its own score: by
    writing it to an instance attribute first and reading that attribute back."""

    __name__ = "echo"
    threshold = 0.5

    def __init__(self) -> None:
        self.score: float | None = None

    async def a_measure(self, case: float, **_: object) -> float | None:
        await asyncio.sleep(random.uniform(0, 0.02))
        self.score = case
        await asyncio.sleep(random.uniform(0, 0.02))
        return self.score


def test_every_score_comes_back_paired_with_its_own_question() -> None:
    # A fake that echoes its input means any mismatch is a pairing failure and nothing else.
    assert _measure_all(_EchoScorer(), CASES) == CASES


def test_sharing_one_metric_across_the_concurrent_calls_really_does_corrupt_the_pairing() -> None:
    # Without this, the test above passes for an implementation that never parallelised
    # at all, and the copy inside _measure_concurrently reads as removable tidiness.
    # Measured at 35 of 40 scores landing on the wrong question.
    async def shared_instance() -> list[float | None]:
        limit = asyncio.Semaphore(8)
        scorer = _EchoScorer()

        async def one(case: float) -> float | None:
            async with limit:
                await scorer.a_measure(case)
                return scorer.score  # the read _measure_concurrently must not do

        return list(await asyncio.gather(*(one(c) for c in CASES)))

    got = asyncio.run(shared_instance())
    misplaced = sum(1 for want, have in zip(CASES, got, strict=True) if want != have)
    assert misplaced > len(CASES) // 2, (
        f"only {misplaced} of {len(CASES)} scores were misplaced by a shared metric, so "
        "this fake no longer reproduces the race the copy exists to prevent"
    )
