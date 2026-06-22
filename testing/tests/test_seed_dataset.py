"""SP7 Task 6: the hand curated seed dataset (`testing/harness/dataset_tools/seed_cases.jsonl`).

Every case class Task 1's generator can mechanically derive (`factoid_one_hop`, `factoid_two_hop`,
`grounded_not_true`, `hallucination_bait`, per `dataset_tools.manifest.CASE_SLICES`) is present in
the curated set, plus three hand authored classes the mechanical generator cannot produce at all
(action/write cases, multi turn trajectories, and the D33 fairness persona cohort, all bucketing as
`manifest.case_slice`'s own declared "other" overflow). The set is small by design (registry
coverage, not padding, D16's 50-150 range): every fact is either the generator's own mechanically
derived ground truth (reused verbatim, never hand transcribed) or a hand authored action/multi turn
scenario anchored to the REAL backend account/catalog domain
(`atlas.domain.accounts`/`atlas.domain.catalog`), never an invented entity.

T4's hard requirement (the comment under `DEFAULT_TEST_FRACTION` in `dataset_tools/manifest.py`):
Task 1's raw mechanical output left the test split with ZERO adversarial (contradiction or bait)
coverage at the default 0.2 test fraction; this curated set fixes that in BOTH directions, verified
below by actually running the real `manifest.build_manifest` over the committed file, never merely
asserted in prose.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from contract_tools import loader
from dataset_tools import manifest
from .fixtures import corpus_expectations

SEED_PATH = Path("testing/harness/dataset_tools/seed_cases.jsonl")


@pytest.fixture(scope="module")
def schema() -> dict:
    return loader.load_schema("dataset")


@pytest.fixture(scope="module")
def seed_cases() -> tuple[dict, ...]:
    return manifest.load_cases_from_jsonl(SEED_PATH)


# ---- the file itself: git native JSONL, one case per line, byte round trippable -------------------


def test_seed_file_is_committed_and_non_empty():
    assert SEED_PATH.exists()
    assert SEED_PATH.read_text().strip()


def test_every_line_is_standalone_valid_json():
    lines = SEED_PATH.read_text().splitlines()
    assert lines, "no cases in the seed file"
    for line in lines:
        json.loads(line)


# ---- sizing (D16: 50 to 150), honest against real registry coverage, not padded --------------------


def test_seed_set_size_is_within_the_d16_range(seed_cases):
    assert 50 <= len(seed_cases) <= 150


def test_every_case_id_is_unique(seed_cases):
    ids = [c["case_id"] for c in seed_cases]
    assert len(ids) == len(set(ids))


# ---- schema validity: every case, no exceptions -----------------------------------------------------


def test_every_seed_case_validates_against_the_dataset_schema(schema, seed_cases):
    for case in seed_cases:
        jsonschema.validate(case, schema)


# ---- every mechanically derivable case class from Task 1 is present, curated not blindly dumped ------


def test_every_generator_case_class_is_represented(seed_cases):
    slices = {manifest.case_slice(c) for c in seed_cases}
    assert slices >= {"factoid_one_hop", "factoid_two_hop", "grounded_not_true", "hallucination_bait"}


def test_hand_authored_classes_beyond_the_generator_are_present(seed_cases):
    # action (write) cases and multi turn trajectories are NOT one of Task 1's three mechanical
    # classes (the generator's own docstring: expected_tool_calls is never derivable); both
    # legitimately fall into manifest.case_slice's declared "other" overflow bucket. Distinguished
    # here by their own real properties, not by slice alone.
    action_single_turn = [c for c in seed_cases if c["intent"] == "action" and len(c["turns"]) == 1]
    multi_turn = [c for c in seed_cases if len(c["turns"]) > 1]
    persona = [c for c in seed_cases if c.get("persona") is not None]
    assert len(action_single_turn) >= 5
    assert len(multi_turn) >= 3
    assert len(persona) >= 8


# ---- expected_tool_calls: hand filled with REAL tool names only (never a ghost tool) -----------------

_REAL_DOTTED_TOOLS = {
    "account.get_account_summary", "account.get_usage", "account.get_bill",
    "account.get_equipment", "account.list_tickets",
    "catalog.list_plans", "catalog.get_plan", "catalog.compute_price", "catalog.check_eligibility",
    "actions.change_plan", "actions.add_addon", "actions.remove_addon",
    "actions.reset_modem", "actions.open_ticket", "actions.book_engineer",
    "knowledge.search_knowledge",
}


def _every_tool_call(case: dict):
    for call in case.get("expected_tool_calls") or ():
        yield call["tool"]
    for turn in case["turns"]:
        checkpoint = turn.get("checkpoint") or {}
        for call in checkpoint.get("expected_tool_calls") or ():
            yield call["tool"]


def test_every_expected_tool_call_names_a_real_mcp_tool(seed_cases):
    for case in seed_cases:
        for tool in _every_tool_call(case):
            assert tool in _REAL_DOTTED_TOOLS, (case["case_id"], tool)


def test_write_tool_args_use_real_backend_catalog_ids_never_the_rag_registry(seed_cases):
    """`actions.change_plan`'s `plan_id` must resolve against `atlas.domain.catalog.CATALOG`
    (`plan_current_fast`/`plan_legacy_value`), never a RAG corpus registry entity id
    (`plan-fiber-500` and friends): the two are disjoint id spaces in this reference system (SP7
    Task 6 step zero finding), and a registry id here would raise `KeyError` if this case were ever
    actually run, exactly the gc-0002 defect this task exists to fix everywhere in the seed set."""
    from atlas.domain import catalog

    for case in seed_cases:
        for call in case.get("expected_tool_calls") or ():
            if call["tool"] == "actions.change_plan":
                assert call["args"]["plan_id"] in catalog.CATALOG
        for turn in case["turns"]:
            for call in (turn.get("checkpoint") or {}).get("expected_tool_calls") or ():
                if call["tool"] in ("catalog.get_plan", "actions.change_plan"):
                    assert call["args"]["plan_id"] in catalog.CATALOG


# ---- intent: only real classify_intent outputs, never an invented label -----------------------------


def test_every_case_intent_is_a_real_classify_intent_output(seed_cases):
    for case in seed_cases:
        assert case["intent"] in ("action", "troubleshooting")
        for turn in case["turns"]:
            checkpoint = turn.get("checkpoint")
            if checkpoint is not None and "expected_intent" in checkpoint:
                assert checkpoint["expected_intent"] in ("action", "troubleshooting")


def test_multi_turn_case_top_level_intent_matches_classify_intent_of_its_own_turn_zero(seed_cases):
    """The multi turn cases' top level `intent` field (what `manifest.py`'s split stratification
    keys on, distinct from the per turn `checkpoint.expected_intent` the previous test checks)
    follows the convention "equals `classify_intent` of the case's own turn 0 text", confirmed here
    for all 4 hand authored multi turn seed cases (SP7 Task 6 review, Minor #1: true by hand
    inspection but previously unpinned by any test, so a future edit could silently violate it)."""
    from atlas.domain.binding import classify_intent

    multi_turn = [c for c in seed_cases if len(c["turns"]) > 1]
    assert len(multi_turn) >= 3
    for case in multi_turn:
        assert classify_intent(case["turns"][0]["user"]) == case["intent"], case["case_id"]


def test_every_turns_expected_intent_matches_the_real_deterministic_classifier(seed_cases):
    """`checkpoint.expected_intent` is graded against `out.get("intent")`, which the real graph
    sets from `classify_intent(question)` alone (the multi turn runner's session carries no pinned
    intent). A checkpoint that disagrees with the real classifier would fail its own diagnostic
    forever, not because the agent is wrong but because the case's own ground truth is: this is the
    exact class of defect T5 found in the pre correction gc-0002."""
    from atlas.domain.binding import classify_intent

    for case in seed_cases:
        for turn in case["turns"]:
            checkpoint = turn.get("checkpoint")
            if checkpoint is not None and "expected_intent" in checkpoint:
                assert classify_intent(turn["user"]) == checkpoint["expected_intent"], case["case_id"]


# ---- the flagship baseline row (SP3 carry): pinned, fused SET membership, never a rank assertion -----


def test_flagship_baseline_row_is_pinned_and_asserts_set_membership_only(seed_cases):
    """The SP3 flagship finding (docs/measurements/sp3-rag-spine.md): the BGE reranker demotes the
    conflict-daniel-contract truth chunk from fused rank 5 of 45 to reranked rank 14, score
    0.00136. Confirming that live is Task 7's job (live lane); this only pins the case here, per
    the repo's "measured, not gated" doctrine: `expected_doc_ids` is a plain list an ID based
    recall/precision grader (`quality.ir_metrics`) checks by SET membership, never a ranked
    position, so this test only has to prove the row exists and carries the real, mechanically
    derived grounding chunk id, never a rank number (there is none to assert)."""
    flagship = next(c for c in seed_cases if c["case_id"] == "seed-flagship-daniel-contract-free")
    assert flagship["adversarial_class"] == "grounded_not_true"
    assert flagship["expected_facts"] == [
        {"fact_id": "contract_term-daniel-2025:contract_months", "value": 12}
    ]
    assert flagship["expected_doc_ids"] == ["2514487e4633b47b"]  # the real, provenance derived chunk id
    assert len(flagship["expected_doc_ids"]) == 1  # a plain membership list, no rank/position field anywhere


# ---- fairness persona cohort (D33): registry anchored equivalence holds, never inferred -------------


def test_persona_cohort_pairs_are_equivalent_by_the_real_d33_check(seed_cases):
    from itertools import combinations

    from quality.agent_metrics import counterfactual_equivalent

    persona_cases = [c for c in seed_cases if c.get("persona") is not None]
    bases: dict[str, list[dict]] = {}
    for case in persona_cases:
        base_id = case["case_id"].rsplit("-persona-", 1)[0]
        bases.setdefault(base_id, []).append(case)
    assert len(bases) >= 3, "fewer than 3 base cases carried a persona cohort"
    for base_id, cohort in bases.items():
        assert len(cohort) >= 3
        for case_a, case_b in combinations(cohort, 2):
            assert counterfactual_equivalent(case_a, case_b), (base_id, case_a["case_id"], case_b["case_id"])


def test_persona_cohort_covers_more_than_one_case_class_registry_anchored(seed_cases):
    # D33 matters most on the hard cases (contradiction, bait), not only plain factoids.
    persona_cases = [c for c in seed_cases if c.get("persona") is not None]
    slices = {manifest.case_slice(c) for c in persona_cases}
    assert "grounded_not_true" in slices
    assert "hallucination_bait" in slices


def test_persona_field_never_carries_a_region_key_region_axis_deferred(seed_cases):
    """Region decision (SP7 Task 6, reaffirming SP7 Task 5's own disposition): D33 names the
    customer name, dialect/register, AND region as candidate varying axes, but the dataset
    contract's persona block (v0.1.0) declares only `name`/`style`
    (`additionalProperties: false`); no consumer (`quality.agent_metrics.counterfactual_equivalent`,
    `dataset_tools.counterfactual`) reads a region axis anywhere. Deferred, not added: a schema MINOR
    bump plus the CHANGELOG gate plus golden regeneration is a real cost this task declines to pay
    for a field nothing yet reads. This test pins the deferral so a future case cannot silently
    smuggle a third persona key back in without a schema change actually authorizing it.
    """
    for case in seed_cases:
        if case.get("persona") is not None:
            assert set(case["persona"]) == {"name", "style"}


# ---- the manifest build: reproducible, splits assign, lint passes, BOTH splits carry adversarial -----


def test_manifest_build_over_the_seed_set_is_byte_reproducible(seed_cases):
    first, first_cases = manifest.build_manifest(seed_cases)
    second, second_cases = manifest.build_manifest(seed_cases)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first_cases == second_cases


def test_manifest_build_over_the_seed_set_passes_the_contamination_lint(seed_cases):
    built, _ = manifest.build_manifest(seed_cases)
    assert built["contamination_lint"]["status"] == "pass"
    assert built["contamination_lint"]["checked_cases"] == len(seed_cases)


def test_both_splits_carry_adversarial_coverage_the_t4_hard_requirement(seed_cases):
    """The exact requirement `dataset_tools/manifest.py`'s own comment under `DEFAULT_TEST_FRACTION`
    names for Task 6 by name: Task 1's raw 88 case mechanical output left the test split with ZERO
    grounded_not_true and ZERO hallucination_bait coverage (both round to 0 at test_fraction 0.2
    with only 2 cases each); the curated set MUST fix this in both directions. Verified here by
    actually running the real `manifest.build_manifest`, never merely asserted in prose."""
    built, _ = manifest.build_manifest(seed_cases)
    for split in ("dev", "test"):
        slices = built["splits"][split]["slices"]
        assert slices["grounded_not_true"] > 0, f"{split} split has zero contradiction coverage"
        assert slices["hallucination_bait"] > 0, f"{split} split has zero hallucination bait coverage"


def test_manifest_reports_every_known_slice_and_every_case_is_placed(seed_cases):
    built, final_cases = manifest.build_manifest(seed_cases)
    assert built["case_count"] == len(seed_cases)
    assert built["splits"]["dev"]["count"] + built["splits"]["test"]["count"] == len(seed_cases)
    for split in ("dev", "test"):
        assert set(built["splits"][split]["slices"]) >= set(manifest.CASE_SLICES)
    assert {c["case_id"] for c in final_cases} == {c["case_id"] for c in seed_cases}


def test_manifest_declares_fact_overlap_honestly_small_registry_visible(seed_cases):
    # The homogeneity risk the plan names twice by design: a small registry (21 entities, 2
    # contradictions) forces some dev/test fact coverage overlap. Declared, never gated, never hidden.
    built, _ = manifest.build_manifest(seed_cases)
    assert built["fact_overlap"]["declared"] is True
    assert isinstance(built["fact_overlap"]["count"], int)


def test_manifest_pins_the_real_corpus_version(seed_cases):
    built, _ = manifest.build_manifest(seed_cases)
    corpus_manifest = json.loads(Path("corpus/rendered/corpus-0.1.1/manifest.json").read_text())
    assert built["corpus_version"] == corpus_manifest["corpus_version"]


# ---- CLI / gate integration: task dataset:build CASES=... over the committed seed file ---------------


def test_cli_main_builds_over_the_committed_seed_file(tmp_path):
    cases_out = tmp_path / "cases.jsonl"
    manifest_out = tmp_path / "manifest.json"
    exit_code = manifest.main(
        [
            "--cases", str(SEED_PATH),
            "--cases-out", str(cases_out),
            "--manifest-out", str(manifest_out),
        ]
    )
    assert exit_code == 0
    written = json.loads(manifest_out.read_text())
    assert written["contamination_lint"]["status"] == "pass"
    for split in ("dev", "test"):
        assert written["splits"][split]["slices"]["grounded_not_true"] > 0
        assert written["splits"][split]["slices"]["hallucination_bait"] > 0


def _rows() -> list[dict]:
    return [json.loads(line) for line in SEED_PATH.read_text().splitlines() if line.strip()]


# --- the golden set must actually resolve against the corpus --------------------------------------


def test_every_expected_doc_id_is_a_chunk_that_exists_in_the_committed_corpus() -> None:
    """`expected_doc_ids` are `chunker.ChunkRecord.chunk_id` values, content addressed over
    (corpus_version, doc_id, doc_version, chunker_version, span). Any corpus change that alters a
    doc's text, or removes it, silently strips its chunk id out of existence and leaves these cases
    grading retrieval against ids nothing can ever return.

    Nothing checked this before. A corpus rebuild left 12 chunk ids across 5 cases pointing at
    nothing, and the full 2252 test suite stayed green.
    """
    from rag_tools import ingest

    live = {record.chunk_id for record in ingest.chunk_corpus(corpus_expectations.CORPUS_VERSION)}
    stale = {
        (row["case_id"], chunk_id)
        for row in _rows()
        for chunk_id in row.get("expected_doc_ids", [])
        if chunk_id not in live
    }
    assert not stale, f"seed cases reference chunk ids absent from the corpus: {sorted(stale)}"


def test_expected_doc_ids_equal_what_the_provenance_join_derives_today() -> None:
    """Stronger than existence: each case's ids must be exactly the chunks its own
    `expected_facts` ground into, per `provenance_join.chunk_ids_for_fact` (the SAME derivation
    `dataset_tools.generator` used to write them). This catches a fact that moved to a different
    document, which an existence check alone would miss."""
    from dataset_tools import provenance_join

    index = provenance_join.load_corpus_index(corpus_expectations.COMMITTED_CORPUS_DIR)
    drifted = []
    for row in _rows():
        facts = row.get("expected_facts") or []
        if not facts:
            continue
        derived = sorted(
            {cid for fact in facts for cid in provenance_join.chunk_ids_for_fact(index, fact["fact_id"])}
        )
        if sorted(row.get("expected_doc_ids", [])) != derived:
            drifted.append(row["case_id"])
    assert not drifted, f"expected_doc_ids no longer match the provenance join for: {drifted}"


# ---- identity is case data, not a test side lookup table ----------------------------------------


def _needs_identity(row: dict) -> bool:
    """A case whose OUTCOME depends on who is signed in. Two shapes qualify: it asserts account
    state afterwards, or it expects a write tool to run, anywhere: a top level
    `expected_tool_calls` entry OR nested in a `turns[].checkpoint.expected_tool_calls` (a multi
    turn case's only write signal can live entirely in a checkpoint, with a null top level
    `end_state`). Reuses `_every_tool_call`, the same recursive traversal
    `test_every_expected_tool_call_names_a_real_mcp_tool` already walks, rather than a second,
    top level only loop that would silently miss the nested shape. "Is a write" is read from
    `atlas.domain.binding.WRITE_TOOLS`, the runtime's own tool binding, never a hand list here."""
    from atlas.domain.binding import WRITE_TOOLS

    if (row.get("end_state") or {}).get("account_assertions"):
        return True
    for tool in _every_tool_call(row):
        if str(tool).rsplit(".", 1)[-1] in WRITE_TOOLS:
            return True
    return False


def test_cases_whose_outcome_depends_on_identity_declare_a_customer_id() -> None:
    """This replaced a hand kept customer id lookup table in test_seed_dataset_multi_turn.py, which
    could drift from the cases it indexed."""
    from atlas.domain.accounts import SEED

    for row in _rows():
        if _needs_identity(row):
            assert row.get("customer_id"), f"{row['case_id']}: identity-dependent case needs a customer_id"
            assert row["customer_id"] in SEED, f"{row['case_id']}: unknown customer_id"


def test_declared_customer_ids_are_seeded_accounts() -> None:
    from atlas.domain.accounts import SEED

    for row in _rows():
        if row.get("customer_id"):
            assert row["customer_id"] in SEED, f"{row['case_id']}: {row['customer_id']!r} not seeded"
