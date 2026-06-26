"""`judge.provisional`, hermetic (SP8 task 3): the manufactured failure generator (ground truth by
construction from SP7's registry contradictions), the registry truth agreement number, the judge vs
judge kappa number, and the ONE provisional calibration artifact that reports both, each stamped
with its actual source. See test_judge_provisional_honesty.py for the KAPPA HONESTY machine
enforced tests (neither number may ever be compared against the 0.6 deployment bar).
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from corpus_tools.registry import load_registry

from judge.contract import JudgeContract
from judge.provisional import (
    CORE_REGISTRY,
    JudgeVsJudgeAgreement,
    ManufacturedCase,
    ProvisionalCalibrationArtifact,
    RegistryTruthAgreement,
    judge_vs_judge_kappa,
    manufactured_cases,
    provisional_calibration_artifact,
    registry_truth_agreement,
)

_CONTRACT_A = JudgeContract("openai:gpt-5.4-nano", "groundedness-v1", "tmpl-a")
_CONTRACT_B = JudgeContract("anthropic:claude-haiku-4-5-20251001", "groundedness-v1", "tmpl-b")
_CLOCK = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)


def _reg():
    return load_registry([CORE_REGISTRY])


def _cases():
    return manufactured_cases(_reg())


# ---- manufactured failures: ground truth BY CONSTRUCTION, zero invented corruption logic ----------


def test_manufactured_cases_cover_every_registry_contradiction_exactly_twice():
    reg = _reg()
    cases = manufactured_cases(reg)
    assert len(cases) == 2 * len(reg.contradictions)
    assert {c.contradiction_id for c in cases} == {c.id for c in reg.contradictions}


def test_each_contradiction_yields_exactly_one_true_and_one_false_case():
    reg = _reg()
    cases = manufactured_cases(reg)
    by_contradiction: dict[str, list[ManufacturedCase]] = {}
    for c in cases:
        by_contradiction.setdefault(c.contradiction_id, []).append(c)
    for contradiction_id, pair in by_contradiction.items():
        assert sorted(c.ground_truth for c in pair) == [0, 1], (
            f"{contradiction_id} must yield exactly one true (1) and one false (0) manufactured case"
        )


def test_the_true_case_states_the_winning_fact_value_the_false_case_states_the_losing_fact_value():
    reg = _reg()
    cases = manufactured_cases(reg)
    for contradiction in reg.contradictions:
        pair = [c for c in cases if c.contradiction_id == contradiction.id]
        true_case = next(c for c in pair if c.ground_truth == 1)
        false_case = next(c for c in pair if c.ground_truth == 0)
        winning_entity_id, _, winning_field = contradiction.winning_fact.partition(":")
        losing_entity_id, _, losing_field = contradiction.losing_fact.partition(":")
        winning_value = str(reg.entity(winning_entity_id).fields[winning_field])
        losing_value = str(reg.entity(losing_entity_id).fields[losing_field])
        assert winning_value in true_case.answer
        assert losing_value in false_case.answer
        assert true_case.answer != false_case.answer


def test_the_cited_context_is_the_same_for_both_cases_and_only_names_the_winning_entity():
    reg = _reg()
    cases = manufactured_cases(reg)
    for contradiction in reg.contradictions:
        pair = [c for c in cases if c.contradiction_id == contradiction.id]
        true_case = next(c for c in pair if c.ground_truth == 1)
        false_case = next(c for c in pair if c.ground_truth == 0)
        assert true_case.context == false_case.context  # both cited against the SAME chunk
        winning_entity_id = contradiction.winning_fact.partition(":")[0]
        losing_entity_id = contradiction.losing_fact.partition(":")[0]
        assert winning_entity_id in true_case.context
        assert losing_entity_id not in true_case.context  # the losing entity never rides along


def test_manufactured_case_ids_are_unique_and_deterministic_across_calls():
    reg = _reg()
    first = manufactured_cases(reg)
    second = manufactured_cases(reg)
    assert [c.case_id for c in first] == [c.case_id for c in second]
    assert len({c.case_id for c in first}) == len(first)


def test_manufactured_cases_defaults_to_loading_the_committed_registry_when_none_is_given():
    cases = manufactured_cases()
    assert len(cases) >= 4  # at least the two committed contradictions, true + false each


# ---- registry truth agreement: known ground truth, plain accuracy, never Cohen's kappa -------------


def test_registry_truth_agreement_is_one_when_the_judge_gets_every_case_right():
    cases = _cases()
    report = registry_truth_agreement(
        _CONTRACT_A, cases, [c.ground_truth for c in cases], generated_at=_CLOCK
    )
    assert report.agreement == 1.0
    assert report.n == len(cases)


def test_registry_truth_agreement_is_zero_when_the_judge_gets_every_case_wrong():
    cases = _cases()
    inverted = [1 - c.ground_truth for c in cases]
    report = registry_truth_agreement(_CONTRACT_A, cases, inverted, generated_at=_CLOCK)
    assert report.agreement == 0.0


def test_registry_truth_agreement_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        registry_truth_agreement(_CONTRACT_A, _cases(), [1], generated_at=_CLOCK)


def test_registry_truth_agreement_rejects_an_empty_set():
    with pytest.raises(ValueError):
        registry_truth_agreement(_CONTRACT_A, [], [], generated_at=_CLOCK)


def test_registry_truth_agreement_renders_its_source_and_contract():
    cases = _cases()
    report = registry_truth_agreement(
        _CONTRACT_A, cases, [c.ground_truth for c in cases], generated_at=_CLOCK
    )
    text = report.render()
    assert RegistryTruthAgreement.SOURCE in text
    assert _CONTRACT_A.judge_model_id in text


# ---- judge vs judge kappa: agreement between two judges, no ground truth at all --------------------


def test_judge_vs_judge_kappa_is_perfect_when_both_judges_agree_on_everything():
    cases = _cases()
    labels = [c.ground_truth for c in cases]
    result = judge_vs_judge_kappa(
        _CONTRACT_A, _CONTRACT_B, [c.case_id for c in cases], labels, labels, generated_at=_CLOCK
    )
    assert result.kappa == 1.0
    assert result.raw_agreement == 1.0


def test_judge_vs_judge_kappa_reads_negative_when_the_two_judges_disagree_on_everything():
    cases = _cases()
    labels_a = [c.ground_truth for c in cases]
    labels_b = [1 - c.ground_truth for c in cases]
    result = judge_vs_judge_kappa(
        _CONTRACT_A, _CONTRACT_B, [c.case_id for c in cases], labels_a, labels_b, generated_at=_CLOCK
    )
    assert result.kappa < 0
    assert result.raw_agreement == 0.0


def test_judge_vs_judge_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        judge_vs_judge_kappa(_CONTRACT_A, _CONTRACT_B, ["a", "b"], [1], [1, 0], generated_at=_CLOCK)


def test_judge_vs_judge_kappa_rejects_an_empty_set():
    with pytest.raises(ValueError):
        judge_vs_judge_kappa(_CONTRACT_A, _CONTRACT_B, [], [], [], generated_at=_CLOCK)


def test_judge_vs_judge_renders_both_contracts_and_its_source():
    cases = _cases()
    labels = [c.ground_truth for c in cases]
    result = judge_vs_judge_kappa(
        _CONTRACT_A, _CONTRACT_B, [c.case_id for c in cases], labels, labels, generated_at=_CLOCK
    )
    text = result.render()
    assert JudgeVsJudgeAgreement.SOURCE in text
    assert _CONTRACT_A.judge_model_id in text
    assert _CONTRACT_B.judge_model_id in text


# ---- ONE artifact, BOTH numbers, EACH labeled by its own real source -------------------------------


def _artifact() -> ProvisionalCalibrationArtifact:
    cases = _cases()
    labels = [c.ground_truth for c in cases]
    rt = registry_truth_agreement(_CONTRACT_A, cases, labels, generated_at=_CLOCK)
    jvj = judge_vs_judge_kappa(
        _CONTRACT_A, _CONTRACT_B, [c.case_id for c in cases], labels, labels, generated_at=_CLOCK
    )
    return provisional_calibration_artifact(rt, jvj, generated_at=_CLOCK)


def test_the_artifact_carries_both_readings_unmodified():
    artifact = _artifact()
    assert artifact.registry_truth.agreement == 1.0
    assert artifact.judge_vs_judge.kappa == 1.0


def test_the_artifact_render_is_byte_reproducible_under_the_same_frozen_instant():
    assert _artifact().render() == _artifact().render()


def test_the_artifact_render_labels_each_number_by_its_own_source():
    text = _artifact().render()
    assert RegistryTruthAgreement.SOURCE in text
    assert JudgeVsJudgeAgreement.SOURCE in text
    assert RegistryTruthAgreement.SOURCE != JudgeVsJudgeAgreement.SOURCE
