"""The retrieval measurements.

A measurement receives a sequence and a frozen set, with no question and therefore no
kind, so it cannot tell a probe question apart from a resolver bug; callers that need to
raise on a bad resolve do so before calling in. Contract on the awkward inputs, covered by
a property check:

| Input | Behaviour |
|---|---|
| Empty correct set | NaN, excluded from averaging |
| A limit of zero | raise |
| A limit larger than the ranked list | evaluate over what is there, never pad |
| Duplicate names in the ranked list | raise |
| Empty ranked list | recall, reciprocal rank, ranking quality are zero; precision is NaN |

Precision is NaN on an empty ranking rather than zero: it asks what share of what was
*fetched* was correct, and nothing was fetched, so the question does not apply, the same
reasoning as an empty correct set. `evals.report.retrieval_summary` excludes non-finite
values from published means and carries the surviving count on the row alongside them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from atlas.contracts import ChunkId
from atlas.corpus.gold import Correct


@dataclass(frozen=True, slots=True)
class Bounded:
    """A score together with the largest value it could possibly have taken.

    A metric like recall@10 has a maximum below 1.0 whenever a question's gold set is
    larger than k, and printing the raw value beside metrics whose maximum is always 1.0
    invites reading it as worse than it is. The ceiling travels with the value instead of
    being left for a reader to work out.

    `attained` is the figure worth comparing across configurations: it's the only one
    whose maximum is the same for every question.
    """

    value: float
    ceiling: float

    def __post_init__(self) -> None:
        if math.isnan(self.value) or math.isnan(self.ceiling):
            return
        # Tolerance, not equality: value and ceiling are ratios of the same small
        # integers and arrive with different rounding.
        if self.value > self.ceiling + 1e-9:
            raise ValueError(f"value {self.value} exceeds its own ceiling {self.ceiling}")

    @property
    def attained(self) -> float:
        """The share of what was reachable that was actually reached.

        Not meaningful published alone: a system scoring 1.0 against a ceiling of 0.1
        attains everything available while still returning almost nothing correct.
        """
        if math.isnan(self.value) or math.isnan(self.ceiling) or self.ceiling == 0.0:
            return math.nan
        return self.value / self.ceiling


def _validate(ranked: Sequence[ChunkId], k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked carries a duplicate chunk name")


def recall_at_k(ranked: Sequence[ChunkId], correct: frozenset[ChunkId], k: int) -> float:
    _validate(ranked, k)
    if not correct:
        return math.nan
    fetched = set(ranked[:k])
    return len(correct & fetched) / len(correct)


def precision_at_k(ranked: Sequence[ChunkId], correct: frozenset[ChunkId], k: int) -> float:
    """The share of what was *fetched* that was correct.

    Divides by how many chunks came back, not by k: a ranking of ten chunks was never
    given twenty slots, so scoring it against twenty would measure the depth setting
    rather than the ranking. `precision_ceiling_at_k` uses the same denominator.

    A consequence worth stating: since the shipped pipeline always returns exactly
    `rerank.depth_out = 10` chunks, `precision@10 == precision@20 == precision@50` on any
    run, while recall@k over the same rankings does move with k.
    """
    _validate(ranked, k)
    if not correct:
        return math.nan
    fetched = ranked[:k]
    if not fetched:
        return math.nan
    return len(correct & set(fetched)) / len(fetched)


def reciprocal_rank_at_k(ranked: Sequence[ChunkId], correct: frozenset[ChunkId], k: int) -> float:
    _validate(ranked, k)
    if not correct:
        return math.nan
    for position, name in enumerate(ranked[:k], start=1):
        if name in correct:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[ChunkId], correct: frozenset[ChunkId], k: int) -> float:
    """Normalised discounted cumulative gain at k, over a flat correct set.

    Binary gain: a chunk is correct or it is not. `graded_ndcg_at_k` is the version that
    knows the difference between the document carrying the answer and the ones consulted
    to reach it.
    """
    _validate(ranked, k)
    if not correct:
        return math.nan
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, name in enumerate(ranked[:k], start=1) if name in correct
    )
    ideal_hits = min(k, len(correct))
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / ideal


def success_at_k(ranked: Sequence[ChunkId], correct: frozenset[ChunkId], k: int) -> float:
    """One when anything correct appears in the first k, zero otherwise.

    Averaged over the question set this reads as the share of questions where retrieval
    put something usable in front of the answering stage at all, independent of gold set
    size.
    """
    _validate(ranked, k)
    if not correct:
        return math.nan
    return 1.0 if any(name in correct for name in ranked[:k]) else 0.0



# Deliberately close together (2 and 1, not 10 and 1): supporting documents are
# genuinely wanted, so the answer is worth more, not everything else worth nothing.
PRIMARY_GAIN = 2.0
SUPPORTING_GAIN = 1.0


def graded_ndcg_at_k(ranked: Sequence[ChunkId], correct: Correct, k: int) -> float:
    """nDCG where the chunks carrying the answer outrank the merely required ones.

    The flat measurements can't see the difference between a ranking that puts the
    document holding the answer first versus tenth, as long as both return the same
    documents.

    Returns NaN when the source doesn't distinguish primary documents, rather than
    falling back to treating every correct chunk as primary: that fallback would make
    this silently identical to `ndcg_at_k` on questions where it can't actually grade.
    """
    _validate(ranked, k)
    if not correct.chunks or not correct.primary:
        return math.nan

    def gain(name: ChunkId) -> float:
        if name in correct.primary:
            return PRIMARY_GAIN
        return SUPPORTING_GAIN if name in correct.chunks else 0.0

    dcg = sum(
        gain(name) / math.log2(position + 1)
        for position, name in enumerate(ranked[:k], start=1)
    )
    # Ideal: every primary chunk first, then supporting ones.
    best = sorted(
        (gain(f) for f in correct.chunks), reverse=True,
    )[:k]
    ideal = sum(g / math.log2(position + 1) for position, g in enumerate(best, start=1))
    return dcg / ideal if ideal else math.nan


def ndcg_ceiling_at_k(
    ranked: Sequence[ChunkId], correct: frozenset[ChunkId], k: int
) -> float:
    """The most nDCG@k can reach when the ranking is only so long.

    `ndcg_at_k` divides by an ideal sized from `k`, so a list shorter than k is scored
    against slots it was never given and cannot reach 1.0. The ratio of two ideals: what
    the filled slots could have earned, over what k slots could have.
    """
    if not correct:
        return math.nan
    filled = min(len(ranked[:k]), len(correct))
    slots = min(k, len(correct))
    if not slots:
        return math.nan
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, slots + 1))
    reachable = sum(1.0 / math.log2(position + 1) for position in range(1, filled + 1))
    return reachable / ideal if ideal else math.nan


def graded_ndcg_ceiling_at_k(ranked: Sequence[ChunkId], correct: Correct, k: int) -> float:
    """The graded twin of `ndcg_ceiling_at_k`, over the same two ideals. NaN under the
    same condition as `graded_ndcg_at_k`, so a question the metric can't score doesn't
    acquire a maximum either.
    """
    if not correct.primary or not correct.chunks:
        return math.nan
    gains = sorted(
        (PRIMARY_GAIN if name in correct.primary else SUPPORTING_GAIN)
        for name in correct.chunks
    )[::-1]
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))
    reachable = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:len(ranked[:k])]))
    return reachable / ideal if ideal else math.nan


# recall_at_k, precision_at_k, and both nDCG variants have a maximum that depends on the
# question; each has a ceiling function here, reported through Bounded.
# reciprocal_rank_at_k does not, because one correct chunk anywhere in a non-empty list
# reaches one.


def recall_ceiling_at_k(
    ranked: Sequence[ChunkId], correct: frozenset[ChunkId], k: int
) -> float:
    """Exact: the slots that were actually filled can hold at most that many correct chunks.

    Takes the ranked list rather than dividing by `k` alone: a run whose retriever
    returns fewer than k chunks would otherwise publish a ceiling nobody could reach.
    """
    if not correct:
        return math.nan
    return min(len(ranked[:k]), len(correct)) / len(correct)


def precision_ceiling_at_k(
    ranked: Sequence[ChunkId], correct: frozenset[ChunkId], k: int
) -> float:
    """Exact: with fewer correct chunks than slots, the surplus slots must be wrong.

    Takes the ranked list, unlike every other ceiling here, because `precision_at_k`
    divides by how many chunks were actually fetched rather than by k.
    """
    fetched = len(ranked[:k])
    if not correct or fetched == 0:
        return math.nan
    return min(fetched, len(correct)) / fetched



def bounded_ndcg_at_k(
    ranked: Sequence[ChunkId], correct: frozenset[ChunkId], k: int
) -> Bounded:
    return Bounded(ndcg_at_k(ranked, correct, k), ndcg_ceiling_at_k(ranked, correct, k))


def bounded_graded_ndcg_at_k(ranked: Sequence[ChunkId], correct: Correct, k: int) -> Bounded:
    return Bounded(graded_ndcg_at_k(ranked, correct, k),
                   graded_ndcg_ceiling_at_k(ranked, correct, k))


def bounded_recall_at_k(
    ranked: Sequence[ChunkId], correct: frozenset[ChunkId], k: int
) -> Bounded:
    return Bounded(recall_at_k(ranked, correct, k), recall_ceiling_at_k(ranked, correct, k))


def bounded_precision_at_k(
    ranked: Sequence[ChunkId], correct: frozenset[ChunkId], k: int
) -> Bounded:
    return Bounded(precision_at_k(ranked, correct, k), precision_ceiling_at_k(ranked, correct, k))

