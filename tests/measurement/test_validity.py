"""The benchmark's own checks: is this corpus capable of telling systems apart?

A gold set can be correct and still useless. If a retriever that ignores the question
scores what the real one scores, every comparison downstream is a null result wearing
a decimal point. These checks run on the benchmark before any number it produces is
believed.

The corpus is a fixed third party artefact, so variation comes from bootstrap
resamples of the question set rather than from regenerating the collection. That
measures the sampling variability of the headline number directly, which is the
quantity the threshold wanted in the first place.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace

import pytest

from evals.baselines import best_constant_ranking
from evals.ir_metrics import recall_at_k
from evals.validity import (
    build_benchmark,
    chance_check,
    correct_answers_exist,
    fixed_list_check,
    headline_recall,
    headroom_check,
)

RESAMPLE_SEEDS = (1, 2, 3, 4)


@pytest.fixture(scope="module")
def benchmark():
    return build_benchmark(0)


@pytest.fixture(scope="module")
def resamples():
    return tuple(build_benchmark(seed) for seed in RESAMPLE_SEEDS)


def test_the_question_set_is_large_enough_to_average_over(benchmark) -> None:
    # tau2 ships 97 tasks, the ceiling on how finely anything here can discriminate. A
    # partial checkout would otherwise silently narrow every interval in the repository.
    assert len(benchmark.questions) == 97


def test_every_question_resolves_to_at_least_one_correct_chunk(benchmark) -> None:
    result = correct_answers_exist(benchmark.questions, benchmark.chunks)
    assert result.passed, result.detail


def test_the_check_goes_red_when_a_required_document_reached_no_chunk(benchmark) -> None:
    # The check above passes vacuously for an implementation that never raises. The fault
    # it exists to catch is a document the gold set requires that the chunker produced no
    # chunk for, and the message has to name the question so a real failure is diagnosable.
    referenced = benchmark.questions[0]
    dropped = referenced.required[0]
    thinned = tuple(f for f in benchmark.chunks if f.document != dropped)
    assert len(thinned) < len(benchmark.chunks), "the referenced document was cut into nothing"

    result = correct_answers_exist(benchmark.questions, thinned)
    assert not result.passed
    assert referenced.name in result.detail


def test_the_headline_number_barely_moves_across_resamples(resamples) -> None:
    # If the headline number swings wildly between resamples of the same corpus, the
    # question set is too small to support any comparison drawn from it, and every
    # threshold derived from that variation is measuring noise.
    figures = [headline_recall(b) for b in resamples]
    assert all(not math.isnan(f) for f in figures)
    assert max(figures) - min(figures) < 0.15, f"headline recall swings across {figures}"


def test_the_fixed_list_gate_is_an_interval_and_has_nothing_to_loosen(benchmark) -> None:
    """Two properties: the gate is the interval excluding zero, so an arm that ties fails
    however the seed is chosen, and the seed is not a bar, so moving it cannot turn a real
    gap unreal.
    """
    outcome = fixed_list_check(benchmark)
    assert outcome.passed, outcome.message
    assert outcome.low > 0.0 and outcome.high > outcome.low

    # A tied arm: every per question difference is exactly zero, so the interval is
    # degenerate at zero and `low > 0.0` is false.
    tied = replace(benchmark, real_scorer=benchmark.fixed_list_scorer)
    assert not fixed_list_check(tied).passed

    # No seed makes the real gap disappear: the bar does not move with the draw.
    for seed in (0, 1, 7, 99):
        assert fixed_list_check(benchmark, seed=seed).low > 0.0


def test_a_question_ignoring_fixed_list_does_not_beat_the_real_system(benchmark) -> None:
    # The saturation finding made executable: a constant list of passages that never reads
    # the question scored a perfect result on a related project, and this is the only check
    # in the suite that would see it happen here.
    outcome = fixed_list_check(benchmark)
    assert outcome.passed, outcome.message
    assert outcome.real_score > outcome.fixed_list_score
    assert outcome.low > 0.0, outcome.message


def test_the_benchmark_has_room_above_chance_and_below_the_ceiling(benchmark) -> None:
    # The saturation trap has two ends and the fixed list check only watches one. A
    # benchmark pinned at the ceiling has no room left to show an improvement, and one
    # sitting at chance has nothing to show an improvement over.
    headroom = headroom_check(benchmark)
    assert headroom.passed, headroom.message

    # Strictly above the interval's upper bound, not above the point estimate. A real
    # system that merely overlaps a random ranking's interval has not been shown to beat
    # it, and reading the two means alone is how that gets missed.
    chance = chance_check(benchmark, seed=0)
    assert headline_recall(benchmark) > chance.high, (
        f"recall {headline_recall(benchmark):.4f} does not clear a random ranking's "
        f"interval [{chance.low:.4f}, {chance.high:.4f}]: this corpus cannot show a gain"
    )


def test_the_fixed_list_check_goes_red_when_the_fixed_list_ties_the_real_system(
    benchmark,
) -> None:
    # The saturation trap arriving exactly rather than approximately: an arm that scores
    # what the fixed list scores, question for question, so every paired difference is zero
    # and the interval is degenerate at zero.
    tied = replace(benchmark, real_scorer=benchmark.fixed_list_scorer)
    outcome = fixed_list_check(tied)
    assert not outcome.passed


def test_the_question_ignoring_baseline_gets_as_many_slots_as_the_arm_it_is_the_floor_for(
    benchmark,
) -> None:
    """A fixed list built at ten slots but scored at `benchmark.limit` flatters the real
    system by exactly the extra depth, silently: `recall_at_k` slices with `[:k]`, which
    can only shrink a list already cut to ten. At `limit=50` that reads +0.3591 against a
    true +0.1187, in the one check this module exists to keep honest.

    Latent rather than live, since every caller today builds at ten, where the two lists
    are the same list and nothing else in the suite asks for another depth.
    """
    deeper = build_benchmark(0, limit=20)
    assert len(deeper.fixed_list) == 20

    outcome = fixed_list_check(deeper)
    assert outcome.passed, outcome.message

    # Against the list as it used to be built: ten entries, scored over twenty slots.
    handicapped = best_constant_ranking(deeper.chunks, deeper.questions)
    assert len(handicapped) == 10
    over_ten_slots = statistics.fmean(
        recall_at_k(handicapped, deeper.gold.correct(q).chunks, deeper.limit)
        for q in deeper.questions if deeper.gold.correct(q).chunks
    )
    assert outcome.fixed_list_score > over_ten_slots, (
        f"the baseline scores {outcome.fixed_list_score:.4f} at depth {deeper.limit} and "
        f"{over_ten_slots:.4f} cut to ten, so it is still being handicapped by the depth")

    # A floor that does not rise with the depth is not a floor.
    assert outcome.fixed_list_score > fixed_list_check(benchmark).fixed_list_score
