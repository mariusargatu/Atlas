"""The rankings that ignore the question, and the one direction they must never move.

A baseline is the only thing here whose weakening makes every other check greener:
`fixed_list_check` asks whether the real arms beat this floor, so lowering it turns a
merge gate green while measuring less, and the comparison it feeds cannot police it.
`best_constant_ranking` already shipped weakened once, scoring recall@10 0.042 against
the 0.107 the real strongest list scores, with nothing going red.
"""

from __future__ import annotations

import pytest

from evals.baselines import FIXED_LIST_SIZE, best_constant_ranking, null_rankings


def test_the_constant_list_is_the_strongest_question_ignoring_opponent(chunks, large):
    # An alphabetical list is what the collapsed sort produced, and it is the weakest
    # opponent available.
    from atlas.corpus.gold import GoldIndex

    gold = GoldIndex.build(chunks)
    frequency: dict[str, int] = {}
    for question in large:
        for name in gold.correct(question).chunks:
            frequency[name] = frequency.get(name, 0) + 1

    best = best_constant_ranking(chunks, large)
    alphabetical = tuple(sorted(frequency)[:FIXED_LIST_SIZE])

    chosen = sum(frequency.get(name, 0) for name in best)
    weakest = sum(frequency.get(name, 0) for name in alphabetical)
    assert chosen > weakest, (
        f"the constant list covers {chosen} question-chunk pairs against the alphabetical "
        f"list's {weakest}, so the check it feeds is facing the weakest opponent available"
    )


def test_the_constant_list_grows_with_the_depth_it_is_scored_at(chunks, large):
    # It did not, and that was silent: the list was built at FIXED_LIST_SIZE and then
    # sliced with `[:k]`, so at `--k 20` the real arms were scored over twenty slots and
    # the floor they must beat over ten.
    for k in (5, 10, 20, 50):
        rankings = null_rankings(chunks, large, k)
        for label, per_question in rankings.items():
            for ranked in per_question.values():
                assert len(ranked) == k, f"{label} returned {len(ranked)} chunks at k={k}"


def test_a_deeper_constant_list_extends_the_shallower_one(chunks, large):
    # Without this the depth sweep would be comparing different baselines rather than the
    # same one at two depths.
    ten = best_constant_ranking(chunks, large, size=10)
    twenty = best_constant_ranking(chunks, large, size=20)
    assert twenty[:10] == ten


def test_an_empty_baseline_is_refused_rather_than_scored(chunks, large):
    # A floor of zero chunks scores zero on every metric, which any system clears, so it
    # is the absence of a floor rather than a lenient one.
    with pytest.raises(ValueError, match="at least 1"):
        best_constant_ranking(chunks, large, size=0)
