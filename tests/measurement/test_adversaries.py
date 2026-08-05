from __future__ import annotations

import statistics
from dataclasses import replace

import pytest

from atlas.config import EmbeddingSettings
from evals import EXEMPT_BY_DESIGN, MEASUREMENTS
from evals.adversaries import (
    _PRECISION_THRESHOLD,
    ALL_DETECTORS,
    BROKEN_SYSTEMS,
    BROKEN_SYSTEMS_BY_NAME,
    build_broken,
    build_healthy,
    fixed_list_outcome,
    resolve_detector,
    run_detector,
)
from evals.ir_metrics import precision_at_k
from evals.systems import System, _build_healthy
from evals.validity import build_benchmark

# How deep the regression adversary below returns. Any depth above _LIMIT reproduces the
# defect; a hundred is the figure the defect was found at.
_DEEP = 100


def _nudged_fixed_list(healthy: System, promote_every: int, demote_every: int) -> System:
    """The question-ignoring fixed list, with a known, small, real change applied to it.

    Every `promote_every`-th question gains a missing gold chunk at the front; every
    `demote_every`-th of the rest loses one to a chunk correct for nothing. Pass zero for
    neither. Built from the gold sets and the fixed list alone, so no change under
    src/atlas can move the numbers this produces.
    """
    benchmark = healthy.benchmark
    fixed = benchmark.fixed_list
    corpus = tuple(sorted(f.name for f in benchmark.chunks))
    retrieved = {}
    for position, question in enumerate(benchmark.questions):
        correct = benchmark.gold.correct(question).chunks
        ranked = fixed
        if promote_every and position % promote_every == 0:
            missing = sorted(correct - set(fixed))
            if missing:
                ranked = (missing[0], *fixed[:len(fixed) - 1])
        elif demote_every and position % demote_every == 0:
            present = sorted(correct & set(fixed))
            if present:
                filler = next(c for c in corpus if c not in correct and c not in fixed)
                ranked = (*(c for c in fixed if c != present[0]), filler)
        retrieved[question.name] = ranked
    return replace(healthy, name="nudged_fixed_list", retrieved=retrieved)


def test_the_table_holds_all_seven_broken_systems():
    # A short table collects fewer parametrised cases and the job stays green having run
    # less than it claims, so the count is asserted rather than trusted.
    assert len(BROKEN_SYSTEMS) == 7 and len({c.name for c in BROKEN_SYSTEMS}) == 7


@pytest.mark.parametrize("case", BROKEN_SYSTEMS, ids=lambda c: c.name)
def test_every_broken_system_is_caught_by_something(case):
    fired = [d for d in case.caught_by if run_detector(d, build_broken(case.name))]
    assert fired, f"{case.name} passes the entire suite, so the suite has a hole."


def test_no_detector_fires_on_the_healthy_system():
    # The control, without which the whole table is worthless: a detector that returns true
    # for everything satisfies every case above.
    #
    # The margin here depends on the fixed list being the strongest question-ignoring list
    # available, not merely an alphabetical one. Ordering it by how many questions each
    # chunk is correct for narrowed the healthy system's lead from 0.1601 to 0.0899.
    healthy = build_healthy()
    fired = [name for name in ALL_DETECTORS if run_detector(name, healthy)]
    assert fired == [], f"these detectors fire on the unbroken system: {fired}"


def test_no_detector_fires_on_the_healthy_system_under_an_alternate_embedding_size():
    # The control above holds at exactly one point in the configuration space, since
    # build_healthy() is cached from the default settings. A threshold tuned against those
    # numbers could stay silent under embedding-1536.json, which `just compare` treats as
    # a legitimate change rather than a broken system.
    system = _build_healthy(build_benchmark(seed=0, limit=10, embedding=EmbeddingSettings(size=1536)))
    fired = [name for name in ALL_DETECTORS if run_detector(name, system)]
    assert fired == [], f"these detectors fire on a healthy system under a legitimate embedding change: {fired}"


def test_no_detector_fires_on_the_healthy_system_at_a_depth_other_than_ten():
    """Depth is the second axis a legitimate configuration moves along.

    Retrieval cuts at `benchmark.limit`, which this asserts first. The detectors
    deliberately do not follow: `_LIMIT` stays ten because the thresholds beside it were
    measured at ten. Scored at `benchmark.limit` instead, this control's precision would
    fall 0.3639 to 0.2758 against a bar that did not move, purely because a caller asked
    for a deeper run.
    """
    system = _build_healthy(build_benchmark(seed=0, limit=20))
    assert {len(ranked) for ranked in system.retrieved.values()} == {20}

    fired = [name for name in ALL_DETECTORS if run_detector(name, system)]
    assert fired == [], f"these detectors fire on a healthy system run at a greater depth: {fired}"

    at_the_detectors_depth = statistics.fmean(
        precision_at_k(system.retrieved[q.name], system.benchmark.gold.correct(q).chunks, 10)
        for q in system.benchmark.questions if system.benchmark.gold.correct(q).chunks
    )
    assert at_the_detectors_depth == pytest.approx(
        statistics.fmean(
            precision_at_k(build_healthy().retrieved[q.name],
                           build_healthy().benchmark.gold.correct(q).chunks, 10)
            for q in build_healthy().benchmark.questions
            if build_healthy().benchmark.gold.correct(q).chunks
        )
    )


@pytest.mark.parametrize("case", BROKEN_SYSTEMS, ids=lambda c: c.name)
def test_every_named_detector_actually_exists(case):
    # Without this, a mistyped detector name makes a broken system look covered
    # while nothing at all ran against it.
    for name in case.caught_by:
        assert resolve_detector(name) is not None, f"{case.name} names unknown detector {name}"


def test_every_registered_measurement_is_mapped_or_exempt_with_a_reason():
    # Walks an explicit registry rather than everything the package exports, since an
    # exemption for every exported helper fills the list with noise inside a week.
    for measurement in MEASUREMENTS:
        name = measurement.__name__
        assert any(name in c.caught_by for c in BROKEN_SYSTEMS) or EXEMPT_BY_DESIGN.get(name), (
            f"{name} catches no broken system and carries no written exemption reason.")


def test_only_the_fixed_list_check_catches_the_same_ten_passages_system():
    # The central claim of the repository, asserted as behaviour rather than read back out
    # of the table, which would only prove the table says what it says.
    broken = build_broken("same_ten_passages")
    fired = [name for name in ALL_DETECTORS if run_detector(name, broken)]
    assert fired == ["fixed_list_validity_check"]
    assert BROKEN_SYSTEMS_BY_NAME["same_ten_passages"].caught_by == tuple(fired)


def test_the_fixed_list_check_reads_what_the_system_retrieved_and_not_which_scorer_it_holds():
    """The verdict has to move when the retrieval moves, and only then.

    A detector that reads `system.benchmark` instead of `system.retrieved` fires identically
    on a perfect retriever wearing the same benchmark, and the test above cannot see the
    difference. Both swaps below fail against such a detector.
    """
    healthy = build_healthy()
    broken = build_broken("same_ten_passages")

    honest_retrieval = replace(broken, retrieved=healthy.retrieved)
    assert not run_detector("fixed_list_validity_check", honest_retrieval)

    question_ignoring_retrieval = replace(healthy, retrieved=broken.retrieved)
    assert run_detector("fixed_list_validity_check", question_ignoring_retrieval)

    outcome = fixed_list_outcome(broken)
    assert outcome.real_score == outcome.fixed_list_score
    assert (outcome.low, outcome.high) == (0.0, 0.0)


def test_a_real_gain_over_the_fixed_list_can_still_fail_to_clear_zero_at_97_questions():
    """What the check above costs: at 97 questions it cannot resolve small real gains.

      helps 10 questions, hurts none ....... +0.0115, [+0.0033, +0.0252], resolved
      helps 17, hurts 17 ................... +0.0121, [-0.0041, +0.0348], not resolved

    The larger mean gain is the one the interval cannot separate from zero, because a paired
    bootstrap's width comes from the spread of per-question differences rather than their
    average. Every real retrieval change has the second shape. A fired detector here says
    97 questions cannot show the gain, not that no gain exists.
    """
    healthy = build_healthy()

    helps_only = _nudged_fixed_list(healthy, promote_every=8, demote_every=0)
    resolved = fixed_list_outcome(helps_only)
    assert resolved.real_score > resolved.fixed_list_score
    assert resolved.low > 0.0, resolved.message
    assert not run_detector("fixed_list_validity_check", helps_only)

    both_ways = _nudged_fixed_list(healthy, promote_every=5, demote_every=3)
    unresolved = fixed_list_outcome(both_ways)
    assert unresolved.real_score > unresolved.fixed_list_score, unresolved.message
    assert unresolved.low < 0.0 < unresolved.high, unresolved.message
    assert run_detector("fixed_list_validity_check", both_ways)

    resolved_gap = resolved.real_score - resolved.fixed_list_score
    unresolved_gap = unresolved.real_score - unresolved.fixed_list_score
    assert unresolved_gap > resolved_gap
    assert unresolved_gap < 0.02


def test_the_precision_detector_scores_every_system_at_the_same_depth():
    """A retriever that returns a hundred chunks with every gold chunk ranked last.

    Scoring at whatever depth a system happens to return makes depth buy immunity: this
    system reads 0.2081 at its own hundred and 0.0000 at the ten the pipeline runs at, so
    it would clear the threshold meant to catch exactly it. No higher threshold could catch
    it without also firing on the healthy control at 0.3639.
    """
    healthy = build_healthy()
    benchmark = healthy.benchmark
    corpus = tuple(sorted(f.name for f in benchmark.chunks))
    retrieved = {}
    for question in benchmark.questions:
        correct = benchmark.gold.correct(question).chunks
        # Every gold chunk it can carry, all of it below the depth anything reads.
        tail = tuple(sorted(correct))[:_DEEP]
        head = tuple(c for c in corpus if c not in correct)[:_DEEP - len(tail)]
        retrieved[question.name] = head + tail
    deep = replace(healthy, name="ranks_every_gold_chunk_last", retrieved=retrieved)

    at_its_own_depth = statistics.fmean(
        precision_at_k(retrieved[q.name], benchmark.gold.correct(q).chunks, _DEEP)
        for q in benchmark.questions
    )
    assert at_its_own_depth > _PRECISION_THRESHOLD, (
        f"scored at the {_DEEP} chunks it returns, this system reads {at_its_own_depth:.4f} "
        f"and clears the {_PRECISION_THRESHOLD} bar meant to catch it")
    assert run_detector("precision_at_k", deep)
    assert run_detector("ndcg_at_k", deep) and run_detector("success_at_k", deep)
