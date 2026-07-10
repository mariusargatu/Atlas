"""SP7 Task 7's measurements doc, guarded: the live capture (recall@k/MRR/nDCG, the flagship
reranked position reproduction, the generation half) is deferred (the fastlane node went
unreachable mid measurement). This test asserts the doc says so honestly -- every section that
needs live data carries the literal `PENDING LIVE CAPTURE` marker, never a filled in number nobody
can trace back to a real run -- and that the exact, copy pasteable rerun command an operator needs
is actually present, so a future reader (or a merge) cannot mistake an unfilled placeholder for a
real captured measurement, and cannot be stuck without knowing how to fill it in either.

This test reads the committed file only (no network, no live process); it is exactly the hermetic
half of the SP6 freeze evidence pattern this doc follows (`contract_tools.freeze_check` reads a
committed evidence file and never reruns the live capture itself either).
"""
from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs/measurements/sp7-datasets-metrics.md")


def _text() -> str:
    return DOC_PATH.read_text()


def test_doc_exists_and_is_nonempty():
    assert DOC_PATH.exists()
    assert _text().strip()


def test_pending_live_capture_marker_present_for_every_live_dependent_section():
    text = _text()
    # Section 2 (retrieval half), Section 4 (flagship reproduction), Section 5 (generation half),
    # plus the top level callout: at least 4 occurrences, one per place a live number would land.
    assert text.count("PENDING LIVE CAPTURE") >= 4


def test_the_reason_for_deferral_is_stated_not_just_the_marker():
    text = _text()
    assert "OOM" in text
    assert "fastlane" in text
    assert "reboot" in text


def test_the_rerun_command_is_present_and_copy_pasteable():
    text = _text()
    assert "docker compose up -d postgres" in text
    assert "source .env.fastlane" in text
    assert "-m live" in text
    assert "testing/tests/test_sp7_retrieval_metrics_live.py" in text


def test_retrieval_metrics_table_cells_are_pending_not_a_fabricated_number():
    text = _text()
    for metric in ("hit_rate@3", "recall@3", "MRR", "nDCG@3"):
        row = next(line for line in text.splitlines() if line.strip().startswith(f"| {metric} "))
        assert row.count("PENDING LIVE CAPTURE") == 2, row  # point column and CI column


def test_flagship_reranked_position_is_pending_not_a_fabricated_rank():
    text = _text()
    assert "reranked rank / score: PENDING LIVE CAPTURE" in text


def test_original_sp3_flagship_numbers_are_cited_not_reproduced_as_new():
    # The SP3 finding (already measured, already committed in sp3-rag-spine.md) is fine to CITE;
    # this test's job is only to make sure it is framed as a citation of a past measurement, never
    # silently presented as this task's own fresh live result.
    text = _text()
    assert "Already measured once, SP3, cited here" in text
    assert "fused rank 5 of 45, reranked rank 14" in text
    assert "score 0.00136" in text


def test_honest_interval_width_section_is_real_not_pending():
    # Section 3 needs no live call (a power sizing bound from n alone): it must NOT carry the
    # pending marker, and must show the real, hermetically computed detectable_effect/required_n
    # numbers this test independently recomputes and cross checks below.
    text = _text()
    section_3 = text.split("## 3. Honest interval width")[1].split("## 4.")[0]
    assert "PENDING LIVE CAPTURE" not in section_3
    assert "0.1889" in section_3
    assert "0.1607" in section_3
    assert "2181" in section_3


def test_honest_interval_width_numbers_reproduce_against_the_real_stats_module():
    from quality import stats

    text = _text()
    section_3 = text.split("## 3. Honest interval width")[1].split("## 4.")[0]
    for n, expected in ((55, 0.1889), (76, 0.1607)):
        recomputed = stats.detectable_effect(n, 0.5)
        assert round(recomputed, 4) == expected
        assert f"{expected}" in section_3
    assert stats.required_n(0.03, 0.5) == 2181


def test_generation_half_section_names_the_key_presence_and_the_reused_smoke_function():
    text = _text()
    section_5 = text.split("## 5. Generation half")[1].split("## 6.")[0]
    assert "PENDING LIVE CAPTURE" in section_5
    assert "ANTHROPIC_API_KEY" in section_5 and "OPENAI_API_KEY" in section_5
    assert "rag_tools.smoke" in section_5


def test_carries_forward_section_names_sp8_and_sp9_and_the_t5_t6_informational_carries():
    text = _text()
    section_6 = text.split("## 6. Carries forward")[1].split("## 7.")[0]
    assert "SP8" in section_6 and "judge" in section_6
    assert "SP9" in section_6 and "matrix" in section_6
    assert "plan_legacy_value" in section_6  # registry vs catalog id space distinction
    assert "region axis" in section_6 and "deferred" in section_6


def test_dataset_contract_stays_unversioned_by_this_task():
    text = _text()
    section_7 = text.split("## 7. Dataset contract versioning")[1]
    assert "0.1.0" in section_7
    assert "No CHANGELOG entry is required" in section_7
