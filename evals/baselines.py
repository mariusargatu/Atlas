"""The rankings that never read the question: what "no system at all" scores on this
corpus. `evals.validity` asks whether the corpus can tell systems apart; these give the
floor every comparison there is measured against. Every published table carries both
because `RunRow` refuses to be built without them.
"""

from __future__ import annotations

import random
from collections import Counter

from atlas.contracts import Chunk, ChunkId, Question, QuestionName
from atlas.corpus.gold import ANSWERABLE_KINDS, GoldIndex

# A default only: `null_rankings` builds the constant list at the requested `k` rather
# than slicing a list fixed at this size, so a larger k doesn't hand the real system more
# slots than its question-ignoring opponent. See docs/checking-the-benchmark.md.
FIXED_LIST_SIZE = 10


def best_constant_ranking(
    chunks: tuple[Chunk, ...], questions: tuple[Question, ...], size: int = FIXED_LIST_SIZE
) -> tuple[ChunkId, ...]:
    """The `size` chunks correct for the most questions, tie broken by name.

    The strongest list a system that never reads the question can produce here. A weaker
    opponent would flatter the real system, which is the failure this baseline guards
    against. Fitted on the question set it is scored against deliberately: this is meant as
    an upper bound on question-blind ranking, not a held-out baseline.
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, not {size}: an empty baseline scores zero")
    gold = GoldIndex.build(chunks)
    frequency: Counter[ChunkId] = Counter()
    for question in questions:
        if question.kind not in ANSWERABLE_KINDS:
            continue
        frequency.update(gold.correct(question).chunks)
    return tuple(sorted(frequency, key=lambda name: (-frequency[name], name))[:size])


def null_rankings(
    chunks: tuple[Chunk, ...], questions: tuple[Question, ...], k: int, seed: int = 0
) -> dict[str, dict[QuestionName, tuple[ChunkId, ...]]]:
    """Both rankings that ignore the query, per question, returned together so a caller
    cannot publish one and forget the other.

    `rng.sample` is drawn fresh per question rather than once and reused: a single draw
    published as the expected value understates true random-chance variance by an order of
    magnitude on this corpus. Expected recall under sampling without replacement is exactly
    k/N regardless of gold set size, since the gold set cancels out of the per-chunk draw
    probability.
    """
    names = [c.name for c in chunks]
    rng = random.Random(seed)
    constant = best_constant_ranking(chunks, questions, size=k)
    return {
        "null: random": {q.name: tuple(rng.sample(names, k)) for q in questions},
        "null: best constant": {q.name: constant for q in questions},
    }
