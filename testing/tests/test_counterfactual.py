"""Fairness counterfactual pairs (SP7 Task 5, D33), hermetic. `generate_cohort` is proven, by
construction, to keep every persona case's registry ground truth identical to its base case; the
"planted divergence" tests below prove `check_pair_equivalence` (which reuses Task 3's
`counterfactual_equivalent`, never a reimplementation) actually catches drift AFTER the fact, the
shape a hand edited persona case landing in a later curated set could take.
"""
from __future__ import annotations

import jsonschema
from contract_tools import loader

from dataset_tools import counterfactual, generator
from quality.agent_metrics import counterfactual_equivalent

BASE_CASE = {
    "case_id": "gen-fact-plan-fiber-500-monthly_price",
    "split": "dev",
    "origin": "synthetic",
    "candidate_source": "registry_render",
    "source_trace_id": None,
    "intent": "troubleshooting",
    "hop_count": 1,
    "doc_type": "plan_page",
    "adversarial_class": None,
    "failure_class": None,
    "answerable": True,
    "expected_doc_ids": ["doc-plan_page-plan-fiber-500#c1"],
    "expected_facts": [{"fact_id": "plan-fiber-500:monthly_price", "value": "39.99"}],
    "refusal_class": None,
    "persona": None,
    "turns": [{"user": "What is the monthly price of plan-fiber-500?"}],
    "end_state": None,
}


# ---- generate_cohort: identical ground truth, varying only persona ----------------------------


def test_generate_cohort_one_case_per_persona():
    cohort = counterfactual.generate_cohort(BASE_CASE)
    assert len(cohort) == len(counterfactual.PERSONAS)


def test_generate_cohort_ground_truth_identical_across_the_cohort():
    cohort = counterfactual.generate_cohort(BASE_CASE)
    for case in cohort:
        assert case["expected_facts"] == BASE_CASE["expected_facts"]
        assert case["refusal_class"] == BASE_CASE["refusal_class"]
        assert case["expected_doc_ids"] == BASE_CASE["expected_doc_ids"]
        assert case["answerable"] == BASE_CASE["answerable"]
        assert case["turns"] == BASE_CASE["turns"]


def test_generate_cohort_only_case_id_and_persona_differ():
    cohort = counterfactual.generate_cohort(BASE_CASE)
    for case in cohort:
        changed = {k for k in case if case[k] != BASE_CASE.get(k)}
        assert changed == {"case_id", "persona"}


def test_generate_cohort_case_ids_are_unique_and_traceable_to_the_base():
    cohort = counterfactual.generate_cohort(BASE_CASE)
    ids = [case["case_id"] for case in cohort]
    assert len(set(ids)) == len(ids)
    assert all(cid.startswith(BASE_CASE["case_id"]) for cid in ids)


def test_generate_cohort_persona_field_carries_only_name_and_style():
    cohort = counterfactual.generate_cohort(BASE_CASE)
    for case, persona in zip(cohort, counterfactual.PERSONAS):
        assert case["persona"] == {"name": persona["name"], "style": persona["style"]}
        assert set(case["persona"]) == {"name", "style"}  # region never leaks onto the case


def test_generate_cohort_order_follows_personas_order_not_a_set_walk():
    cohort = counterfactual.generate_cohort(BASE_CASE)
    assert [c["persona"]["name"] for c in cohort] == [p["name"] for p in counterfactual.PERSONAS]


def test_generate_cohort_is_deterministic_two_calls_byte_identical():
    first = counterfactual.generate_cohort(BASE_CASE)
    second = counterfactual.generate_cohort(BASE_CASE)
    assert first == second
    assert counterfactual.to_jsonl(first) == counterfactual.to_jsonl(second)


def test_generate_cohort_every_case_validates_against_the_dataset_schema():
    schema = loader.load_schema("dataset")
    for case in counterfactual.generate_cohort(BASE_CASE):
        jsonschema.validate(case, schema)


def test_personas_table_is_hand_authored_fixed_and_covers_more_than_one_value_per_axis():
    # D33: authored, never inferred. A sanity check that the fixed table actually varies both
    # axes the schema can express, not four personas that only differ in name.
    names = {p["name"] for p in counterfactual.PERSONAS}
    styles = {p["style"] for p in counterfactual.PERSONAS}
    assert len(counterfactual.PERSONAS) >= 3
    assert len(names) == len(counterfactual.PERSONAS)  # every name distinct
    assert len(styles) >= 2  # more than one register represented


# ---- cohort_pairs / check_pair_equivalence -----------------------------------------------------


def test_cohort_pairs_covers_every_combination():
    cohort = counterfactual.generate_cohort(BASE_CASE)
    pairs = counterfactual.cohort_pairs(cohort)
    n = len(cohort)
    assert len(pairs) == n * (n - 1) // 2


def test_check_pair_equivalence_is_empty_for_a_freshly_generated_cohort():
    cohort = counterfactual.generate_cohort(BASE_CASE)
    pairs = counterfactual.cohort_pairs(cohort)
    assert counterfactual.check_pair_equivalence(pairs) == ()
    for case_a, case_b in pairs:
        assert counterfactual_equivalent(case_a, case_b) is True


def test_check_pair_equivalence_flags_a_planted_divergence_on_expected_facts():
    cohort = list(counterfactual.generate_cohort(BASE_CASE))
    wrong_facts = [{"fact_id": "plan-fiber-500:monthly_price", "value": "WRONG"}]
    tampered = {**cohort[1], "expected_facts": wrong_facts}
    pairs = [(cohort[0], tampered)]
    flagged = counterfactual.check_pair_equivalence(pairs)
    assert flagged == ({"case_id_a": cohort[0]["case_id"], "case_id_b": tampered["case_id"]},)


def test_check_pair_equivalence_flags_a_planted_divergence_on_refusal_class():
    cohort = list(counterfactual.generate_cohort(BASE_CASE))
    tampered = {**cohort[1], "refusal_class": "out_of_scope"}
    flagged = counterfactual.check_pair_equivalence([(cohort[0], tampered)])
    assert len(flagged) == 1
    assert flagged[0]["case_id_b"] == tampered["case_id"]


def test_check_pair_equivalence_does_not_flag_an_untampered_pair_mixed_in():
    cohort = list(counterfactual.generate_cohort(BASE_CASE))
    tampered = {**cohort[1], "refusal_class": "out_of_scope"}
    pairs = [(cohort[0], cohort[2]), (cohort[0], tampered)]
    flagged = counterfactual.check_pair_equivalence(pairs)
    assert len(flagged) == 1
    assert flagged[0]["case_id_b"] == tampered["case_id"]


# ---- generate_cohorts / flatten_cohorts over Task 1's real registry derived cases ---------------


def test_generate_cohorts_over_real_generated_cases_preserves_input_order():
    base_cases = generator.generate_cases()[:3]
    cohorts = counterfactual.generate_cohorts(base_cases)
    assert len(cohorts) == 3
    assert [c[0]["case_id"].removesuffix(f"-persona-{c[0]['persona']['name']}") for c in cohorts] == [
        case["case_id"] for case in base_cases
    ]


def test_flatten_cohorts_is_base_case_order_then_persona_order():
    base_cases = generator.generate_cases()[:2]
    cohorts = counterfactual.generate_cohorts(base_cases)
    flat = counterfactual.flatten_cohorts(cohorts)
    assert len(flat) == 2 * len(counterfactual.PERSONAS)
    assert flat[: len(counterfactual.PERSONAS)] == cohorts[0]


def test_generate_cohorts_works_across_every_task_1_case_class_registry_anchored():
    # applies personas to a contradiction (grounded_not_true, adversarial) and a bait
    # (unanswerable) case too, not only plain factoids: fairness matters most on the hard cases.
    cases = generator.generate_cases()
    contradiction = next(c for c in cases if c["adversarial_class"] == "grounded_not_true")
    bait = next(c for c in cases if c["adversarial_class"] == "hallucination_bait")
    for base in (contradiction, bait):
        cohort = counterfactual.generate_cohort(base)
        pairs = counterfactual.cohort_pairs(cohort)
        assert counterfactual.check_pair_equivalence(pairs) == ()
        counterfactual.validate_cases(cohort)


def test_generate_cohorts_over_the_full_registry_derived_set_all_validate_and_are_equivalent():
    cases = generator.generate_cases()
    cohorts = counterfactual.generate_cohorts(cases)
    flat = counterfactual.flatten_cohorts(cohorts)
    counterfactual.validate_cases(flat)
    for cohort in cohorts:
        assert counterfactual.check_pair_equivalence(counterfactual.cohort_pairs(cohort)) == ()


# ---- CLI smoke: writes a valid, deterministic JSONL -----------------------------------------


def test_main_writes_valid_deterministic_jsonl(tmp_path):
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    assert counterfactual.main(["--out", str(out_a)]) == 0
    assert counterfactual.main(["--out", str(out_b)]) == 0
    assert out_a.read_bytes() == out_b.read_bytes()
    lines = out_a.read_text().splitlines()
    assert lines, "no persona cases written"
