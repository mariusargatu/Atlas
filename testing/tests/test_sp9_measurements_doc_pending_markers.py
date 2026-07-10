"""SP9 Task 8's measurements doc, guarded: four lanes (the staged matrix sweep, the real Neo4j LLM
extraction run, the load lane's real k6 burst sweep, the real Ollama daemon call plus its human
spot check sample) are deferred, none of them run as part of this docs only task. This test asserts
the doc says so honestly -- every section that needs a live run carries the literal `PENDING LIVE
CAPTURE` marker, never a filled in number nobody can trace back to a real run -- and that the exact,
copy pasteable rerun command (or, where none exists yet, an honestly disclosed composition) is
present for each lane, plus the honest interval width carried forward from SP7 (cited, not
recomputed), the flagship SP3 conflict slice cited rather than re announced, and the pareto point
kept to prose only (no chart, no fabricated table).

Every section anchors its own assertions to that section's own text (`text.split("## N. ")[1]
.split("## N+1. ")[0]`), following `test_sp8_measurements_doc_pending_markers.py`'s own corrected
pattern (SP8's own I1 finding: a single, whole document occurrence count can pass even when a
tamper replaces one specific section's real marker with a fabricated number, so long as enough
OTHER sections still carry the marker). A tamper that swaps out any one section's own marker or
rerun command must fail here on its own, independent of every other section.

This test reads the committed file only (no network, no live process); it is exactly the hermetic
half of the SP6 freeze evidence pattern and the SP7/SP8 measurements doc guards this one follows.
"""
from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs/measurements/sp9-variants-matrix.md")


def _text() -> str:
    return DOC_PATH.read_text()


def _section(text: str, start_marker: str, end_marker: str) -> str:
    return text.split(start_marker, 1)[1].split(end_marker, 1)[0]


def _flat(text: str) -> str:
    """Collapse whitespace (including a markdown line wrap's own newline) so a multi word phrase
    that happens to wrap across two source lines still matches as one contiguous phrase."""
    return " ".join(text.split())


def test_doc_exists_and_is_nonempty():
    assert DOC_PATH.exists()
    assert _text().strip()


def test_pending_live_capture_marker_present_for_every_deferred_lane():
    text = _text()
    # The intro callout (two of its four items), the matrix sweep, the flagship citation's own
    # restatement of SP7's still pending reproduction, the load lane, and the ollama lane: at least
    # 6 occurrences. The per section anchored tests below are the ones that actually catch a single
    # lane's marker being swapped for a fabricated number; this is only the floor.
    assert text.count("PENDING LIVE CAPTURE") >= 6


def test_the_reason_for_deferral_is_stated_for_the_matrix_sweep_not_just_the_marker():
    text = _text()
    assert "atlas-fastlane" in text
    assert "2026-07-21" in text
    assert "no live matrix CLI was built" in text
    assert "__main__.py`" in text
    assert "does not exist" in _flat(text)


def test_matrix_sweep_section_carries_its_own_pending_marker_and_composition():
    text = _text()
    section = _section(text, "## 2. ", "## 3. ")
    assert "PENDING LIVE CAPTURE" in section
    # This section's OWN heading also carries the phrase (Section title), so a tamper that swaps
    # only the body claim for a fabricated number could still leave a bare count check green; this
    # specific sentence is the body's own real claim and must survive independent of the heading.
    assert "no real embedder, reranker, or generator has swept this dataset yet" in section
    assert "run_matrix" in section
    assert "build_ollama_generator_component" in section
    assert "docker compose up -d postgres" in section
    assert "source .env.fastlane" in section


def test_neo4j_extraction_is_marked_pending_with_a_real_rerun_command():
    text = _text()
    # Anchored in Section 1 (where the fixture derived numbers live) and the intro callout, not a
    # whole document count: replacing only this lane's marker with a fabricated "real" reading must
    # fail here independent of every other lane.
    section = _section(text, "## 1. ", "## 2. ")
    assert "PENDING LIVE CAPTURE" in section
    assert "task graph-up" in section
    assert "FIXTURE DERIVED" in section


def test_neo4j_extraction_numbers_reproduce_against_the_real_study_module():
    # The fixture numbers cited in Section 1 are not merely quoted from a past report; they must
    # still match the real, committed evals.graphrag study's own output exactly (rounded the same
    # way the module itself already rounds for its printed report).
    from evals.graphrag.registry_graph import EXTRACTED_ENTITY_CLUSTERS, EXTRACTED_TRIPLES, GOLD_ENTITY_CLUSTERS, GOLD_TRIPLES
    from quality.graph_metrics import bcubed_prf, pairwise_prf, triple_prf

    text = _text()
    section = _section(text, "## 1. ", "## 2. ")

    p, r, f1 = triple_prf(EXTRACTED_TRIPLES, GOLD_TRIPLES)
    assert (round(p, 4), round(r, 4), round(f1, 4)) == (0.9375, 0.7895, 0.8571)
    assert "0.9375" in section and "0.7895" in section and "0.8571" in section

    bp, br, bf1 = bcubed_prf(EXTRACTED_ENTITY_CLUSTERS, GOLD_ENTITY_CLUSTERS)
    assert round(bp, 2) == 0.75
    assert round(br, 2) == 1.0
    assert "0.75" in section

    assert pairwise_prf(EXTRACTED_ENTITY_CLUSTERS, GOLD_ENTITY_CLUSTERS) == (0.0, 0.0, 0.0)


def test_honest_interval_width_section_is_real_not_pending():
    # This section needs no live call (a power sizing bound cited from SP7, not recomputed): it
    # must NOT carry the pending marker.
    text = _text()
    section = _section(text, "## 3. ", "## 4. ")
    assert "PENDING LIVE CAPTURE" not in section
    assert "0.1607" in section
    assert "2181" in section
    assert "0.1889" in section


def test_honest_interval_width_numbers_reproduce_against_the_real_stats_module_and_are_cited_not_recomputed():
    from quality import stats

    text = _text()
    section = _section(text, "## 3. ", "## 4. ")
    assert round(stats.detectable_effect(76, 0.5), 4) == 0.1607
    assert round(stats.detectable_effect(55, 0.5), 4) == 0.1889
    assert stats.required_n(0.03, 0.5) == 2181
    # The section must say plainly it is a citation of SP7's own already computed numbers, never a
    # recomputed rosier one.
    assert "already computed and disclosed this" in section
    assert "cites the SAME numbers rather than recomputing a rosier version" in section


def test_flagship_conflict_slice_is_cited_not_re_announced():
    text = _text()
    section = _section(text, "## 4. ", "## 5. ")
    assert "fused rank 5 of 45, reranked rank 14" in section
    assert "0.00136" in section
    assert "cited here, never re derived or re announced as a fresh" in section
    # The SP7 reproduction itself is still pending; this document does not attempt a third one.
    assert "PENDING LIVE CAPTURE" in section
    assert "does not attempt a third" in section


def test_load_lane_section_carries_its_own_pending_marker_and_real_taskfile_targets():
    text = _text()
    section = _section(text, "## 5. ", "## 6. ")
    assert "PENDING LIVE CAPTURE" in section
    # Same teeth requirement as the matrix sweep section above: this section's own heading also
    # carries the phrase, so the body's own specific claim must be checked independently.
    assert "the real burst sweep has not run" in section
    assert "k6 run testing/harness/load/k6/chat_sse_load.js" in section
    assert "uv run python -m load --iterations" in section
    assert "task load:k6" in section and "task load:join" in section


def test_ollama_section_cites_adr030_and_carries_its_own_pending_marker():
    text = _text()
    section = _section(text, "## 6. ", "## 7. ")
    assert "ADR-030" in section
    assert "PENDING LIVE CAPTURE" in section
    assert "build_ollama_generator_component" in section
    assert "build_ollama_spot_check_items" in section
    # The ADR's own numbers are cited, never restated as a second table.
    assert "does not restate or re derive those numbers a second time" in section


def test_pareto_section_is_prose_only_no_chart_no_fabricated_table():
    text = _text()
    section = _section(text, "## 7. ", "## 8. ")
    assert "SP11 builds the actual rendered chart" in section
    assert "No chart, no table of fabricated numbers" in section
    assert "because" in section.lower()
    # No markdown table (a pipe-delimited row) and no embedded chart markup anywhere in this
    # section: the pareto point stays prose only, per the plan's own instruction.
    assert "|" not in section
    assert "```" not in section
    assert "<svg" not in section.lower()
    assert "mermaid" not in section.lower()


def test_pareto_section_names_all_three_variants_and_the_reranker_and_local_generator_tradeoffs():
    text = _text()
    section = _section(text, "## 7. ", "## 8. ")
    assert "naive" in section.lower()
    assert "agentic" in section.lower()
    assert "graph" in section.lower()
    assert "reranker" in section.lower()
    assert "qwen2.5" in section


def test_contract_versioning_section_states_no_new_contract_touch_by_this_task():
    text = _text()
    section = _section(text, "## 8. ", "## 9. ")
    assert "1.2.0 to 1.3.0" in section
    assert "No CHANGELOG entry is required" in _flat(section)


def test_carries_section_names_sp10_and_sp11_with_their_own_reasons():
    text = _text()
    section = text.split("## 9. Carries forward", 1)[1]
    assert "SP10" in section and "burst benchmark" in section.lower() or "SP10" in section
    assert "tofu apply" in section
    assert "SP11" in section
    assert "because clause" in section
    assert "interval bars" in section


def test_judge_label_content_flattening_caveat_is_cited_not_silently_repeated():
    text = _text()
    section = text.split("## 9. Carries forward", 1)[1]
    assert "judge_label" in section
    assert "AttributeError" in section
    assert "sp8-judge-human-loop.md" in section


def test_no_em_dash_character_anywhere_in_the_document():
    # This repo's own convention (per SP9 task 6's own fix round): no literal em dash character
    # anywhere, "--" used as the dash substitute instead.
    text = _text()
    assert "—" not in text
