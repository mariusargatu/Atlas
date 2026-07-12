"""Semantic mutation testing over the IR metrics: every REALISTIC mutant (a plausible human bug, not
a syntactic operator flip) must be killed by a Phase-1 assertion. A survivor here is a bug the suite
would ship. This is the deterministic, gate-safe core of "did I measure my measuring instrument?" —
the classical mutation-testing loop, run on committed mutants with no LLM.
"""
from __future__ import annotations

from evals.mutation.mutants import IR_METRIC_MUTANTS, kills


def test_every_realistic_mutant_is_killed_by_the_metric_suite():
    survivors = [m.name for m in IR_METRIC_MUTANTS if not kills(m)]
    assert not survivors, (
        "surviving realistic bugs the IR-metric suite would ship (add a witness assertion to "
        f"test_ir_metrics.py to kill each): {survivors}"
    )


def test_the_registry_is_non_trivial_and_documented():
    assert len(IR_METRIC_MUTANTS) >= 5
    for mutant in IR_METRIC_MUTANTS:
        assert mutant.realistic_bug and mutant.metric and mutant.witness


def _mutant(name: str):
    return next(m for m in IR_METRIC_MUTANTS if m.name == name)


def test_mutant_bodies_handle_relevant_and_non_relevant_positions():
    # a list with a non-relevant doc exercises both loop branches inside the mutant bodies
    assert _mutant("precision_shrinks_denominator").fn(["a", "b"], frozenset({"a"}), 2) == 0.5
    assert _mutant("map_divides_by_hits").fn(["a", "b"], frozenset({"a"}), 2) == 1.0
    assert _mutant("reciprocal_rank_takes_last").fn(["z", "y"], frozenset({"y"})) == 0.5  # z not relevant
