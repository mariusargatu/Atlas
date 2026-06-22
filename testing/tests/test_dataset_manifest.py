"""dataset_manifest.json, deterministic splits, and the two direction contamination lint
(SP7 Task 4). Consumes `dataset_tools.generator` read only (its `generate_cases()` output is the
input this module builds a manifest over); never edits it.

Independent recomputation, the same style `test_dataset_generator.py` already uses: the lint and
overlap tests read the committed corpus docs directly rather than trusting
`dataset_tools.manifest`'s own internals, so a real regression in the join or the lint logic is
caught, not tautologically confirmed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from dataset_tools import generator, manifest

CORPUS_DIR = Path("corpus/rendered/corpus-0.1.1")


@pytest.fixture(scope="module")
def real_cases() -> tuple[dict, ...]:
    return generator.generate_cases()


# --- reproducibility: byte identical across repeat builds, and independent of input order --------


def test_build_manifest_twice_is_byte_identical(real_cases) -> None:
    first, first_cases = manifest.build_manifest(real_cases)
    second, second_cases = manifest.build_manifest(real_cases)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first_cases == second_cases


def test_build_manifest_is_independent_of_input_case_order(real_cases) -> None:
    shuffled = tuple(reversed(real_cases))
    forward, _ = manifest.build_manifest(real_cases)
    backward, _ = manifest.build_manifest(shuffled)
    assert json.dumps(forward, sort_keys=True) == json.dumps(backward, sort_keys=True)


def test_write_manifest_two_writes_byte_identical(tmp_path: Path, real_cases) -> None:
    built, _ = manifest.build_manifest(real_cases)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    manifest.write_manifest(built, first_path)
    manifest.write_manifest(built, second_path)
    assert first_path.read_bytes() == second_path.read_bytes()


def test_different_seed_changes_at_least_one_assignment(real_cases) -> None:
    a = manifest.assign_splits(real_cases, seed=manifest.DEFAULT_SEED)
    b = manifest.assign_splits(real_cases, seed=manifest.DEFAULT_SEED + 1)
    assert a != b


def test_same_seed_is_stable_across_calls(real_cases) -> None:
    a = manifest.assign_splits(real_cases, seed=7)
    b = manifest.assign_splits(real_cases, seed=7)
    assert a == b


# --- stratification: every stratum accounted for, exact counts pinned against the committed corpus ---


def test_case_slice_classifies_every_generated_case_not_other(real_cases) -> None:
    slices = {c["case_id"]: manifest.case_slice(c) for c in real_cases}
    assert set(slices.values()) <= {
        "factoid_one_hop",
        "factoid_two_hop",
        "grounded_not_true",
        "hallucination_bait",
    }


def test_split_assignment_covers_every_case_exactly_once(real_cases) -> None:
    assignment = manifest.assign_splits(real_cases)
    assert set(assignment) == {c["case_id"] for c in real_cases}
    assert all(split in ("dev", "test") for split in assignment.values())


def test_stratified_split_counts_pinned_against_committed_generated_set(real_cases) -> None:
    # 65 one hop, 19 two hop, 2 contradiction, 2 bait (Task 1's own pinned counts), one stratum
    # each since every generated case shares intent "troubleshooting". test_fraction 0.2 rounds
    # each stratum independently: round(65*0.2)=13, round(19*0.2)=4, round(2*0.2)=0, round(2*0.2)=0.
    assignment = manifest.assign_splits(real_cases, seed=manifest.DEFAULT_SEED, test_fraction=0.2)
    by_slice: dict[str, dict[str, int]] = {}
    for case in real_cases:
        s = manifest.case_slice(case)
        split = assignment[case["case_id"]]
        by_slice.setdefault(s, {"dev": 0, "test": 0})[split] += 1
    assert by_slice["factoid_one_hop"] == {"dev": 52, "test": 13}
    assert by_slice["factoid_two_hop"] == {"dev": 15, "test": 4}
    assert by_slice["grounded_not_true"] == {"dev": 2, "test": 0}
    assert by_slice["hallucination_bait"] == {"dev": 2, "test": 0}


def test_manifest_reports_per_split_and_per_slice_counts(real_cases) -> None:
    built, _ = manifest.build_manifest(real_cases, test_fraction=0.2)
    assert built["splits"]["dev"]["count"] == 71
    assert built["splits"]["test"]["count"] == 17
    assert built["splits"]["dev"]["count"] + built["splits"]["test"]["count"] == len(real_cases)
    assert built["splits"]["dev"]["slices"]["factoid_one_hop"] == 52
    assert built["splits"]["test"]["slices"]["factoid_one_hop"] == 13


def test_manifest_declares_zero_count_slices_explicitly_never_omits_them(real_cases) -> None:
    # At test_fraction 0.2 the two case grounded_not_true and two case hallucination_bait strata
    # both round to zero test coverage (round(2 * 0.2) equals 0). The manifest must still declare
    # both keys at 0 in the test split's slices map, not omit them, the same "declared, never
    # silent" convention fact_overlap already follows for its own count.
    built, _ = manifest.build_manifest(real_cases, test_fraction=0.2)
    test_slices = built["splits"]["test"]["slices"]
    assert "grounded_not_true" in test_slices
    assert test_slices["grounded_not_true"] == 0
    assert "hallucination_bait" in test_slices
    assert test_slices["hallucination_bait"] == 0
    # every known case_slice class appears in both splits, present even when the count is zero
    for split in ("dev", "test"):
        assert set(built["splits"][split]["slices"]) >= set(manifest.CASE_SLICES)


def test_manifest_pins_dataset_and_corpus_version(real_cases) -> None:
    built, _ = manifest.build_manifest(real_cases)
    corpus_manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    assert built["dataset_version"] == manifest.DATASET_VERSION
    assert built["corpus_version"] == corpus_manifest["corpus_version"]


def test_final_cases_carry_the_assigned_split_not_the_generators_placeholder(real_cases) -> None:
    _, final_cases = manifest.build_manifest(real_cases)
    assignment = manifest.assign_splits(real_cases)
    for case in final_cases:
        assert case["split"] == assignment[case["case_id"]]
    # the generator's own placeholder was "dev" for every case; a real split was computed, not copied
    assert any(case["split"] == "test" for case in final_cases)


# --- content addressing -----------------------------------------------------------------------------


def test_content_hash_changes_when_a_single_case_changes(real_cases) -> None:
    built, _ = manifest.build_manifest(real_cases)
    mutated = list(real_cases)
    mutated[0] = {**mutated[0], "turns": [{"user": "a different question entirely"}]}
    mutated_built, _ = manifest.build_manifest(tuple(mutated))
    assert built["content_hash"] != mutated_built["content_hash"]


def test_case_list_is_content_addressed_per_case(real_cases) -> None:
    built, _ = manifest.build_manifest(real_cases)
    for case_id, entry in built["cases"].items():
        assert isinstance(entry["content_hash"], str)
        assert len(entry["content_hash"]) == 64  # sha256 hex digest


def test_duplicate_case_ids_raise_a_clear_error(real_cases) -> None:
    dupe = (*real_cases, real_cases[0])
    with pytest.raises(ValueError, match=real_cases[0]["case_id"]):
        manifest.build_manifest(dupe)


# --- contamination lint direction 1: verbatim case text leaking an undeclared doc -------------------


def _daniel_contract_case(real_cases: tuple[dict, ...]) -> dict:
    return next(c for c in real_cases if c["case_id"] == "gen-fact-contract_term-daniel-2025-contract_months")


def test_lint_passes_when_verbatim_text_is_from_a_declared_doc(real_cases) -> None:
    case = _daniel_contract_case(real_cases)
    doc_text = (CORPUS_DIR / "docs" / "doc-contract_terms-contract_term-daniel-2025.txt").read_text()
    quoted = doc_text[10:70]
    assert len(quoted) >= manifest.MIN_LINT_TEXT_LEN
    declared_case = {**case, "turns": [{"user": quoted}]}
    violations = manifest.lint_verbatim_leakage((declared_case,), corpus_dir=CORPUS_DIR)
    assert violations == ()


def test_lint_flags_verbatim_text_from_an_undeclared_doc(real_cases) -> None:
    case = _daniel_contract_case(real_cases)
    other_doc_text = (CORPUS_DIR / "docs" / "doc-policy-policy-fair-use.txt").read_text()
    leaked = other_doc_text[10:70]
    assert len(leaked) >= manifest.MIN_LINT_TEXT_LEN
    planted_case = {**case, "turns": [{"user": leaked}]}
    violations = manifest.lint_verbatim_leakage((planted_case,), corpus_dir=CORPUS_DIR)
    assert len(violations) == 1
    assert violations[0]["case_id"] == case["case_id"]
    assert violations[0]["doc_id"] == "doc-policy-policy-fair-use"


def test_short_text_below_threshold_is_never_flagged(real_cases) -> None:
    case = _daniel_contract_case(real_cases)
    other_doc_text = (CORPUS_DIR / "docs" / "doc-policy-policy-fair-use.txt").read_text()
    short = other_doc_text[10 : 10 + manifest.MIN_LINT_TEXT_LEN - 1]
    planted_case = {**case, "turns": [{"user": short}]}
    violations = manifest.lint_verbatim_leakage((planted_case,), corpus_dir=CORPUS_DIR)
    assert violations == ()


def test_committed_generated_set_passes_the_lint_clean(real_cases) -> None:
    assert manifest.lint_verbatim_leakage(real_cases, corpus_dir=CORPUS_DIR) == ()


def test_build_manifest_raises_on_a_planted_leak(real_cases) -> None:
    case = _daniel_contract_case(real_cases)
    other_doc_text = (CORPUS_DIR / "docs" / "doc-policy-policy-fair-use.txt").read_text()
    leaked = other_doc_text[10:70]
    mutated = tuple(
        {**c, "turns": [{"user": leaked}]} if c["case_id"] == case["case_id"] else c for c in real_cases
    )
    with pytest.raises(manifest.ContaminationLintError):
        manifest.build_manifest(mutated)


def test_build_manifest_reports_clean_lint_status_for_the_committed_set(real_cases) -> None:
    built, _ = manifest.build_manifest(real_cases)
    assert built["contamination_lint"]["status"] == "pass"
    assert built["contamination_lint"]["checked_cases"] == len(real_cases)


# --- contamination lint direction 2: dev/test fact coverage overlap, declared never silent ----------


def test_fact_overlap_field_is_always_present_even_when_zero() -> None:
    cases = (
        {
            "case_id": "c1",
            "split": "placeholder",
            "origin": "synthetic",
            "intent": "troubleshooting",
            "answerable": True,
            "turns": [{"user": "question one about something long enough to pass the lint"}],
            "expected_facts": [{"fact_id": "fact-a", "value": "1"}],
        },
    )
    overlap = manifest._fact_overlap(
        tuple({**c, "split": "dev"} for c in cases)
    )
    assert overlap["declared"] is True
    assert overlap["count"] == 0
    assert overlap["fact_ids"] == []


def test_fact_overlap_is_declared_when_a_fact_spans_both_splits() -> None:
    shared_fact = {"fact_id": "fact-shared", "value": "1"}
    cases = (
        {"case_id": "c1", "split": "dev", "expected_facts": [shared_fact]},
        {"case_id": "c2", "split": "test", "expected_facts": [shared_fact]},
        {"case_id": "c3", "split": "dev", "expected_facts": [{"fact_id": "fact-only-dev", "value": "2"}]},
    )
    overlap = manifest._fact_overlap(cases)
    assert overlap["declared"] is True
    assert overlap["count"] == 1
    assert overlap["fact_ids"] == ["fact-shared"]


def test_committed_generated_set_overlap_matches_independent_recomputation(real_cases) -> None:
    built, final_cases = manifest.build_manifest(real_cases)
    fact_splits: dict[str, set[str]] = {}
    for case in final_cases:
        for fact in case.get("expected_facts") or []:
            fact_splits.setdefault(fact["fact_id"], set()).add(case["split"])
    expected_overlap = sorted(fid for fid, splits in fact_splits.items() if len(splits) > 1)
    assert built["fact_overlap"]["fact_ids"] == expected_overlap
    assert built["fact_overlap"]["count"] == len(expected_overlap)
    assert built["fact_overlap"]["declared"] is True


# --- JSONL round trip: consuming the generator's own JSONL output as input --------------------------


def test_load_cases_from_jsonl_round_trips_generator_output(tmp_path: Path, real_cases) -> None:
    path = tmp_path / "cases.jsonl"
    generator.write_jsonl(real_cases, path)
    loaded = manifest.load_cases_from_jsonl(path)
    assert loaded == real_cases


# --- CLI / gate integration: the pipeline Taskfile's dataset:build target actually runs -------------


def test_cli_main_end_to_end_generates_splits_and_manifest(tmp_path: Path) -> None:
    cases_out = tmp_path / "dataset_cases.jsonl"
    manifest_out = tmp_path / "dataset_manifest.json"
    exit_code = manifest.main(
        ["--cases-out", str(cases_out), "--manifest-out", str(manifest_out)]
    )
    assert exit_code == 0
    assert cases_out.exists()
    assert manifest_out.exists()

    written_manifest = json.loads(manifest_out.read_text())
    written_cases = manifest.load_cases_from_jsonl(cases_out)
    assert written_manifest["case_count"] == len(written_cases)
    assert written_manifest["contamination_lint"]["status"] == "pass"
    assert set(written_manifest["cases"]) == {c["case_id"] for c in written_cases}


def test_cli_main_with_explicit_cases_jsonl_input(tmp_path: Path, real_cases) -> None:
    cases_in = tmp_path / "in.jsonl"
    generator.write_jsonl(real_cases, cases_in)
    cases_out = tmp_path / "out.jsonl"
    manifest_out = tmp_path / "manifest.json"
    exit_code = manifest.main(
        [
            "--cases",
            str(cases_in),
            "--cases-out",
            str(cases_out),
            "--manifest-out",
            str(manifest_out),
        ]
    )
    assert exit_code == 0
    written_manifest = json.loads(manifest_out.read_text())
    assert written_manifest["case_count"] == len(real_cases)


def test_cli_main_returns_nonzero_and_writes_nothing_on_a_planted_leak(tmp_path: Path, real_cases) -> None:
    case = _daniel_contract_case(real_cases)
    other_doc_text = (CORPUS_DIR / "docs" / "doc-policy-policy-fair-use.txt").read_text()
    leaked = other_doc_text[10:70]
    mutated = tuple(
        {**c, "turns": [{"user": leaked}]} if c["case_id"] == case["case_id"] else c for c in real_cases
    )
    cases_in = tmp_path / "in.jsonl"
    generator.write_jsonl(mutated, cases_in)
    cases_out = tmp_path / "out.jsonl"
    manifest_out = tmp_path / "manifest.json"

    exit_code = manifest.main(
        [
            "--cases",
            str(cases_in),
            "--cases-out",
            str(cases_out),
            "--manifest-out",
            str(manifest_out),
        ]
    )
    assert exit_code == 1
    assert not cases_out.exists()
    assert not manifest_out.exists()
