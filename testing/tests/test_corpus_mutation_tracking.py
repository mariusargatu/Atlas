"""corpus_mutation.tracking, hermetic (SP8 task 7): the answer-tracks-mutated-truth assertion logic,
exercised here over a STUB pair of answers, never a real generation call. This is the check that
actually makes this a MUTATION lane rather than Task 6's metamorphic one: metamorphic asks whether
the SAME truth survives DIFFERENT wording; this asks whether the answer CHANGES to track a
DIFFERENT truth under the SAME wording. An agent answering from parametric/training knowledge (or a
stale cache or index) instead of the freshly re-indexed retrieved context fails this check by
repeating the STALE, pre mutation value instead of the new one -- exactly what
`test_an_answer_repeating_the_stale_pre_mutation_value_fails_to_track_the_new_truth` below proves.
"""
from __future__ import annotations

from corpus_mutation.selection import FactMutation
from corpus_mutation.tracking import AnswerTrackingResult, answer_tracks_mutated_truth

_MUTATION = FactMutation(
    contradiction_id="conflict-daniel-contract",
    fact_ref="contract_term-daniel-2025:contract_months",
    old_value=12,
    new_value=24,
    question="Is my plan contract free?",
)

_TRACKS_NEW_TRUTH_ANSWER = (
    "According to Daniel's 2025 Contract Term, you're on a 24 month contract, so it isn't contract free."
)
_REPEATS_STALE_TRUTH_ANSWER = (
    "According to Daniel's 2025 Contract Term, you're on a 12 month contract, so it isn't contract free."
)
_UNRELATED_ANSWER = "I couldn't find any contract information on file for your account."


def test_an_answer_stating_the_new_mutated_value_tracks_the_new_truth():
    result = answer_tracks_mutated_truth(_MUTATION, _TRACKS_NEW_TRUTH_ANSWER)
    assert result.tracks_new_truth is True
    assert result.repeats_stale_truth is False
    assert result.holds is True


def test_an_answer_repeating_the_stale_pre_mutation_value_fails_to_track_the_new_truth():
    result = answer_tracks_mutated_truth(_MUTATION, _REPEATS_STALE_TRUTH_ANSWER)
    assert result.tracks_new_truth is False
    assert result.repeats_stale_truth is True
    assert result.holds is False


def test_an_unrelated_answer_tracks_neither_the_new_nor_the_stale_value():
    result = answer_tracks_mutated_truth(_MUTATION, _UNRELATED_ANSWER)
    assert result.tracks_new_truth is False
    assert result.repeats_stale_truth is False
    assert result.holds is False


def test_answer_tracking_dereferences_a_tool_result_too_not_only_response_text():
    tool_results = [{"contract_term-daniel-2025": {"contract_months": 24}}]
    result = answer_tracks_mutated_truth(_MUTATION, "no contract detail in my reply", tool_results)
    assert result.tracks_new_truth is True


def test_answer_tracking_result_records_the_fact_ref_and_both_values():
    result = answer_tracks_mutated_truth(_MUTATION, _TRACKS_NEW_TRUTH_ANSWER)
    assert result.fact_ref == _MUTATION.fact_ref
    assert result.new_value == 24
    assert result.old_value == 12


def test_answer_tracking_result_is_a_dataclass_instance():
    assert isinstance(answer_tracks_mutated_truth(_MUTATION, _TRACKS_NEW_TRUTH_ANSWER), AnswerTrackingResult)
