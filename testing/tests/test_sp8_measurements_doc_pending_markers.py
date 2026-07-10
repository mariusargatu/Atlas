"""SP8 Task 8's measurements doc, guarded: three live lanes (the judge live provisional sweep, the
metamorphic live lane, the corpus mutation live lane) are deferred, none of them run as part of this
docs only task. This test asserts the doc says so honestly -- every section that needs a live run
carries the literal `PENDING LIVE CAPTURE` marker, never a filled in number nobody can trace back to
a real run -- and that the exact, copy pasteable recompute command for each lane is present, plus the
KAPPA HONESTY source labeling, the small n disclosures (n=4 manufactured set, 76 not 200 real
labels), and the two folded in carries (the judge_label live content flattening caveat, the registry
versus catalog id space distinction) the plan requires as measurement doc content, not a side note.

This test reads the committed file only (no network, no live process); it is exactly the hermetic
half of the SP6 freeze evidence pattern and the SP7 measurements doc guard this one follows
(`testing/tests/test_sp7_measurements_doc_pending_markers.py`).
"""
from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs/measurements/sp8-judge-human-loop.md")


def _text() -> str:
    return DOC_PATH.read_text()


def test_doc_exists_and_is_nonempty():
    assert DOC_PATH.exists()
    assert _text().strip()


def test_pending_live_capture_marker_present_for_every_live_dependent_lane():
    text = _text()
    # The judge live provisional sweep, the metamorphic live lane, the corpus mutation live lane:
    # at least 4 occurrences, one per deferred lane section (a fifth, incidental prose mention in
    # Section 2.4, is not counted on here -- see the per section anchored tests below, which are
    # the ones that actually catch a single section's marker being swapped for a fabricated number).
    assert text.count("PENDING LIVE CAPTURE") >= 4


def test_registry_truth_agreement_section_carries_its_own_pending_marker_and_recompute():
    # SP7's own guard (test_sp7_measurements_doc_pending_markers.py) anchors markers per
    # section/table row rather than counting loosely across the whole document; this test follows
    # that pattern for Section 2.2 specifically, so replacing ONLY this section's live number with a
    # fabricated reading (e.g. "agreement 0.97, excellent") fails here even though the document
    # still carries 4 other PENDING LIVE CAPTURE occurrences elsewhere.
    text = _text()
    section = text.split("### 2.2 ")[1].split("### 2.3 ")[0]
    assert "PENDING LIVE CAPTURE" in section
    assert "task judge-live" in section


def test_judge_vs_judge_kappa_section_carries_its_own_pending_marker_and_recompute():
    # Section 2.3's own anchor: replacing this section's live number with a fabricated kappa (e.g.
    # "kappa 0.91, excellent") must fail here, independent of Section 2.2's own marker.
    text = _text()
    section = text.split("### 2.3 ")[1].split("### 2.4 ")[0]
    assert "PENDING LIVE CAPTURE" in section
    assert "task judge-live" in section


def test_metamorphic_live_section_carries_its_own_pending_marker_and_recompute():
    text = _text()
    section = text.split("## 4. ")[1].split("## 5. ")[0]
    assert "PENDING LIVE CAPTURE" in section
    assert "docker compose up postgres tei-embed tei-rerank" in section
    assert "testing/tests/test_metamorphic_live.py" in section


def test_corpus_mutation_live_section_carries_its_own_pending_marker_and_recompute():
    text = _text()
    section = text.split("## 5. ")[1].split("## 6. ")[0]
    assert "PENDING LIVE CAPTURE" in section
    assert "python -m corpus_mutation" in section
    assert "testing/tests/test_corpus_mutation_live.py" in section


def test_the_reason_for_deferral_is_stated_for_each_lane_not_just_the_marker():
    text = _text()
    # judge live: keys present but deliberately not spent by a docs only task.
    assert "ANTHROPIC_API_KEY" in text and "OPENAI_API_KEY" in text
    assert "docs only" in text
    # metamorphic and corpus mutation live lanes: the deleted fastlane node, the compose stack.
    assert "fastlane" in text
    assert "docker compose up postgres tei-embed tei-rerank" in text


def test_judge_live_recompute_command_is_present_and_copy_pasteable():
    text = _text()
    assert "task judge-live" in text


def test_metamorphic_live_recompute_command_is_present_and_copy_pasteable():
    text = _text()
    assert "testing/tests/test_metamorphic_live.py" in text


def test_corpus_mutation_live_recompute_command_is_present_and_copy_pasteable():
    text = _text()
    assert "python -m corpus_mutation" in text
    assert "testing/tests/test_corpus_mutation_live.py" in text


def test_provisional_numbers_are_each_labeled_by_their_own_source():
    text = _text()
    assert "registry_truth_manufactured_ground_truth_by_construction" in text
    assert "judge_vs_judge_no_ground_truth" in text


def test_honesty_statement_names_the_only_number_that_licenses_deployment():
    text = _text()
    assert "does not license" in text or "never licenses" in text
    assert "AUTOMATION_BAR" in text
    assert "0.6" in text
    assert "CalibrationReport.licensed" in text


def test_fixture_derived_numbers_are_labeled_as_such_not_a_real_distribution_claim():
    text = _text()
    assert "fixture derived" in text
    assert "not a real distribution claim" in text


def test_no_dedicated_taskfile_target_for_human_calibration_is_disclosed_honestly():
    text = _text()
    assert "judge:calibrate" in text
    assert "does not exist" in text


def test_small_n_honesty_is_stated_for_both_the_manufactured_set_and_the_label_reality():
    text = _text()
    assert "n=4" in text
    assert "76" in text and "200" in text


def test_judge_label_content_flattening_caveat_is_named():
    text = _text()
    assert "judge_label" in text
    assert "AttributeError" in text
    assert "FAILED" in text


def test_registry_versus_catalog_id_space_distinction_is_named():
    text = _text()
    assert "plan-fiber-100" in text
    assert "contract_term-daniel-2025" in text
    assert "search_knowledge" in text
    assert "plan_legacy_value" in text


def test_carries_section_names_sp9_and_sp11():
    text = _text()
    section = text.split("## 8. Carries forward")[1]
    assert "SP9" in section and "panel_vote" in section and "D28" in section
    assert "SP11" in section and "CalibrationReport.licensed" in section


def test_contract_versioning_section_states_no_new_contract_touch():
    text = _text()
    section = text.split("## 9.")[1]
    assert "No CHANGELOG entry" in section
