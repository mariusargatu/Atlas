"""`matrix.cases`, hermetic: loading the retrieval slice of the real, committed 76 case seed set
(`testing/harness/dataset_tools/seed_cases.jsonl`) -- a file read, no network, no retriever. Mirrors
`test_sp7_retrieval_metrics_live.py`'s own `_retrieval_relevant_cases` filter (55 of 76 cases), pinned
here as a hermetic property of the loader rather than only ever checked live.
"""
from __future__ import annotations

from pathlib import Path

from dataset_tools import manifest as dataset_manifest

from matrix.cases import MatrixCase, RETRIEVAL_SLICES, load_matrix_cases

SEED_PATH = Path(__file__).resolve().parents[1] / "harness" / "dataset_tools" / "seed_cases.jsonl"


def test_seed_path_exists_and_is_the_76_case_committed_set():
    all_cases = dataset_manifest.load_cases_from_jsonl(SEED_PATH)
    assert len(all_cases) == 76


def test_load_matrix_cases_keeps_exactly_the_55_case_retrieval_relevant_slice():
    cases = load_matrix_cases(SEED_PATH)
    assert len(cases) == 55
    assert all(isinstance(c, MatrixCase) for c in cases)


def test_every_loaded_case_has_a_nonempty_relevant_set():
    for case in load_matrix_cases(SEED_PATH):
        assert case.relevant_doc_ids, f"{case.case_id} loaded into the retrieval slice with no relevant docs"


def test_case_id_and_query_round_trip_from_the_first_turn():
    cases = load_matrix_cases(SEED_PATH)
    by_id = {c.case_id: c for c in cases}
    assert "gen-fact-plan-fiber-500-name" in by_id
    assert by_id["gen-fact-plan-fiber-500-name"].query == "What is the name of plan-fiber-500?"


def test_a_narrower_slice_selection_returns_fewer_cases():
    narrow = load_matrix_cases(SEED_PATH, slices=frozenset({"factoid_two_hop"}))
    wide = load_matrix_cases(SEED_PATH, slices=RETRIEVAL_SLICES)
    assert 0 < len(narrow) < len(wide)


# ---- query_entity_ids: the registry derived golden set T1/T2 named as the real, deferred supplier -


def test_query_entity_ids_is_derived_from_the_registry_ids_expected_facts_carry():
    """T1's own report names "the real supplier of query_entity_ids from the registry derived
    golden set"; this IS that supplier. No new data: `expected_facts[].fact_id` is already the
    registry's own `entity_id:field` shape (`dataset_tools.generator`'s own convention), so this is
    the exact `quality.agent_metrics.expected_entity_ids` extraction, reused rather than
    re-derived."""
    cases = load_matrix_cases(SEED_PATH)
    by_id = {c.case_id: c for c in cases}
    one_hop = by_id["gen-fact-plan-fiber-500-name"]
    assert one_hop.query_entity_ids == frozenset({"plan-fiber-500"})


def test_query_entity_ids_carries_every_entity_named_by_a_two_hop_cases_facts():
    cases = load_matrix_cases(SEED_PATH)
    by_id = {c.case_id: c for c in cases}
    two_hop = by_id["gen-edge-available_in-plan-fiber-500-region-north"]
    assert two_hop.query_entity_ids == frozenset({"plan-fiber-500", "region-north"})


def test_query_entity_ids_is_never_empty_for_the_retrieval_slice():
    """Every case in `RETRIEVAL_SLICES` carries `expected_facts` (factoid_one_hop/two_hop/
    grounded_not_true all dereference at least one registry fact); `grade_documents` in
    `agentic_rag.py` is vacuous only when this is empty, so a real supplier means real cases in
    this slice now actually exercise CRAG grading, not just the hermetic no-op path."""
    for case in load_matrix_cases(SEED_PATH):
        assert case.query_entity_ids, f"{case.case_id} loaded with no query_entity_ids"


def test_file_order_is_preserved_never_resorted():
    all_cases = dataset_manifest.load_cases_from_jsonl(SEED_PATH)
    file_order = [c["case_id"] for c in all_cases if dataset_manifest.case_slice(c) in RETRIEVAL_SLICES]
    loaded_order = [c.case_id for c in load_matrix_cases(SEED_PATH)]
    assert loaded_order == file_order
