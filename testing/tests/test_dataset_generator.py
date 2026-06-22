"""Mechanical registry to case generator (SP7 Task 1): every ground truth field is derived BY
CONSTRUCTION from the committed registry and corpus, zero LLM. Three case classes: one hop and
two hop factoid (`Registry.entities` / `Registry.edges`), `grounded_not_true` adversarial cases
(`Registry.contradictions`), and answerable false hallucination bait (the never rendered pool).

The provenance join tests recompute their expected chunk ids independently, the same way
`test_chunker.py` verifies chunker behavior: via `rag_tools.chunker.chunk_document` directly over
the committed corpus text and provenance sidecar, never by copying a value out of `generator.py`
itself. That is what makes them a real regression on the join, not a tautology.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
from contract_tools import loader
from dataset_tools import generator, provenance_join
from rag_tools import chunker

CORPUS_DIR = Path("corpus/rendered/corpus-0.1.1")


@pytest.fixture(scope="module")
def schema() -> dict:
    return loader.load_schema("dataset")


@pytest.fixture(scope="module")
def cases() -> tuple[dict, ...]:
    return generator.generate_cases()


@pytest.fixture(scope="module")
def by_class(cases: tuple[dict, ...]) -> dict[str, list[dict]]:
    one_hop = [c for c in cases if c.get("hop_count") == 1 and c["adversarial_class"] is None]
    two_hop = [c for c in cases if c.get("hop_count") == 2 and c["adversarial_class"] is None]
    contradiction = [c for c in cases if c["adversarial_class"] == "grounded_not_true"]
    bait = [c for c in cases if c["adversarial_class"] == "hallucination_bait"]
    return {"one_hop": one_hop, "two_hop": two_hop, "contradiction": contradiction, "bait": bait}


def _independent_chunk_ids(doc_id: str, fact_ref: str) -> tuple[str, ...]:
    """Recompute, straight from the committed corpus files and `rag_tools.chunker`, which chunk
    ids of `doc_id` ground `fact_ref`. Independent of `dataset_tools.provenance_join`."""
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    sidecar = json.loads((CORPUS_DIR / "provenance" / f"{doc_id}.json").read_text())
    text = (CORPUS_DIR / "docs" / f"{doc_id}.txt").read_text()
    placements = sidecar["placements"]
    target_span = next(p["span"] for p in placements if p["fact_ref"] == fact_ref)
    chunks = chunker.chunk_document(
        doc_id=doc_id,
        doc_type=sidecar["doc_type"],
        text=text,
        doc_version=manifest["docs"][doc_id],
        corpus_version=manifest["corpus_version"],
        placements=placements,
    )
    start, end = target_span
    return tuple(
        sorted(
            chunk.chunk_id
            for chunk in chunks
            if start < chunk.char_span[1] and chunk.char_span[0] < end
        )
    )


# --- shape validation, all three case classes ----------------------------------------------------


def test_one_hop_cases_validate_against_schema(schema, by_class) -> None:
    assert by_class["one_hop"], "no one hop cases generated"
    for case in by_class["one_hop"]:
        jsonschema.validate(case, schema)


def test_two_hop_cases_validate_against_schema(schema, by_class) -> None:
    assert by_class["two_hop"], "no two hop cases generated"
    for case in by_class["two_hop"]:
        jsonschema.validate(case, schema)


def test_contradiction_cases_validate_against_schema(schema, by_class) -> None:
    assert by_class["contradiction"], "no contradiction cases generated"
    for case in by_class["contradiction"]:
        jsonschema.validate(case, schema)


def test_bait_cases_validate_against_schema(schema, by_class) -> None:
    assert by_class["bait"], "no bait cases generated"
    for case in by_class["bait"]:
        jsonschema.validate(case, schema)


def test_every_generated_case_validates_against_schema(schema, cases) -> None:
    for case in cases:
        jsonschema.validate(case, schema)


# --- case counts, pinned against the committed corpus (corpus-0.1.1, core.yaml only) --------------


def test_case_counts_per_class(by_class) -> None:
    # core.yaml: 21 entities, 19 edges, 2 contradictions, 2 never rendered. 65 of the 19 renderable
    # entities' fields have a placement somewhere in the corpus (the mechanically answerable
    # one hop pool); every edge's two endpoints are each groundable by at least one field.
    assert len(by_class["one_hop"]) == 65
    assert len(by_class["two_hop"]) == 19
    assert len(by_class["contradiction"]) == 2
    assert len(by_class["bait"]) == 2


def test_bait_covers_exactly_the_never_rendered_pool(by_class) -> None:
    bait_ids = {c["case_id"] for c in by_class["bait"]}
    assert bait_ids == {"gen-bait-plan-quantum-5g", "gen-bait-fee-teleport-setup"}


# --- provenance join correctness: two known facts, byte exact -------------------------------------


def test_provenance_join_single_doc_fact_is_byte_exact(cases) -> None:
    # contract_term-daniel-2025:contract_months is winning_fact of conflict-daniel-contract and is
    # placed in exactly one committed doc.
    case = next(c for c in cases if c["case_id"] == "gen-fact-contract_term-daniel-2025-contract_months")
    expected = _independent_chunk_ids("doc-contract_terms-contract_term-daniel-2025", "contract_term-daniel-2025:contract_months")
    assert expected, "fixture assumption broken: the fact must be placed in the committed corpus"
    assert case["expected_doc_ids"] == list(expected)
    assert case["doc_type"] == "contract_terms"
    assert case["expected_facts"] == [{"fact_id": "contract_term-daniel-2025:contract_months", "value": 12}]


def test_provenance_join_multi_doc_fact_is_byte_exact(cases) -> None:
    # plan-fiber-100:monthly_price is placed in three docs (its own plan page plus two
    # troubleshooting docs that also quote the price).
    case = next(c for c in cases if c["case_id"] == "gen-fact-plan-fiber-100-monthly_price")
    doc_ids = (
        "doc-plan_page-plan-fiber-100",
        "doc-troubleshooting-device-modem-d3--plan-fiber-100",
        "doc-troubleshooting-device-router-ax2--plan-fiber-100",
    )
    expected: set[str] = set()
    for doc_id in doc_ids:
        found = _independent_chunk_ids(doc_id, "plan-fiber-100:monthly_price")
        assert found, f"fixture assumption broken: {doc_id} must place this fact"
        expected.update(found)
    assert case["expected_doc_ids"] == sorted(expected)
    assert len(case["expected_doc_ids"]) == 3, "one chunk per doc on corpus-0.1.1 (single chunk per doc)"


# --- contradiction cases: winning fact only, registry anchored -------------------------------------


def test_contradiction_case_matches_registry_winning_fact(by_class) -> None:
    case = next(c for c in by_class["contradiction"] if c["case_id"] == "gen-contradiction-conflict-daniel-contract")
    assert case["adversarial_class"] == "grounded_not_true"
    assert case["hop_count"] == 1
    assert case["answerable"] is True
    assert case["expected_facts"] == [{"fact_id": "contract_term-daniel-2025:contract_months", "value": 12}]
    assert case["turns"][0]["user"].lower().startswith("is my plan contract free")


def test_second_contradiction_case_is_inter_doc_two_hop(by_class) -> None:
    case = next(c for c in by_class["contradiction"] if c["case_id"] == "gen-contradiction-conflict-promo-price-north")
    assert case["hop_count"] == 2
    assert case["expected_facts"] == [
        {"fact_id": "region-north:equipment_rental_override_amount", "value": "5.00"}
    ]


# --- gc-0001 vs the registry: bound so the two can never silently drift again (SP7 T1 review,
# --- Important 2). gc-0001 (contracts/dataset/examples/single_turn_case.json) is SP1 era, pre
# --- registry, and once disagreed with `conflict-daniel-contract`'s registry truth on hop_count,
# --- fact_id naming, and value representation. gc-0001 was updated to match; this test is what
# --- keeps it matched against a future registry edit to the same scenario. -------------------------


def test_committed_example_matches_the_registry_derived_case_for_the_same_scenario(by_class) -> None:
    generated = next(
        c for c in by_class["contradiction"] if c["case_id"] == "gen-contradiction-conflict-daniel-contract"
    )
    example = loader.load_examples("dataset")["single_turn_case"]
    assert example["case_id"] == "gc-0001"
    assert example["hop_count"] == generated["hop_count"]
    assert example["expected_facts"] == generated["expected_facts"]


# --- gc-0002 vs the real backend catalog (SP7 Task 6 correction). gc-0002 used to bind a RAG
# --- registry plan id ("fiber-500") to catalog.get_plan/actions.change_plan, an intent
# --- ("plan_change") classify_intent cannot emit, and a turn 1 with no action cue: three
# --- independent ways the case was behaviorally impossible against the real graph (SP7 Task 5's
# --- multi turn runner proved it: dataset_tools.multi_turn.run_multi_turn_case never reaches
# --- catalog.get_plan on a troubleshooting turn, and CATALOG[plan_id] raises KeyError for a
# --- registry plan id in the first place, since the RAG corpus registry (corpus/registry/core.yaml,
# --- plan-fiber-*) and the account/catalog backend (atlas.domain.catalog.CATALOG,
# --- plan_current_fast/plan_legacy_value) are two disjoint id spaces in this reference system).
# --- gc-0002 is rebased off the backend catalog: a real classify_intent output ("action", earned by
# --- turn 1's own "switch my" cue), a real dotted tool name, and a real, offered plan id. This test
# --- is what keeps it bound to the real catalog against a future edit reintroducing an unreal id. -


def test_committed_multi_turn_example_is_anchored_to_the_real_backend_catalog_not_the_registry() -> None:
    from atlas.domain import catalog

    example = loader.load_examples("dataset")["multi_turn_case"]
    assert example["case_id"] == "gc-0002"
    assert example["intent"] == "action"  # a real classify_intent output, never a made up label
    get_plan_call, change_plan_call = example["expected_tool_calls"]
    assert get_plan_call["tool"] == "catalog.get_plan"
    assert change_plan_call["tool"] == "actions.change_plan"
    plan_id = change_plan_call["args"]["plan_id"]
    assert get_plan_call["args"]["plan_id"] == plan_id
    assert plan_id in catalog.CATALOG  # a real, offered plan id, not a registry entity id
    assertions = {a["path"]: a["equals"] for a in example["end_state"]["account_assertions"]}
    assert assertions["plan_id"] == plan_id
    assert assertions["bill.amount"] == str(catalog.compute_price(plan_id))
    # turn 1's own phrasing earns the "action" binding by itself (a real cue from
    # atlas.domain.binding._ACTION_CUES), never assumed: catalog.get_plan is unreachable on a
    # troubleshooting turn (domain.binding.INTENT_TOOLS), so without this cue the checkpoint below
    # would be unreachable, exactly the SP7 Task 5 finding this correction closes.
    assert "switch my" in example["turns"][0]["user"].lower()
    assert example["turns"][0]["checkpoint"]["expected_tool_calls"] == [get_plan_call]


# --- hallucination bait: unanswerable, no grounding doc --------------------------------------------


def test_bait_cases_are_unanswerable_with_empty_doc_ids(by_class) -> None:
    for case in by_class["bait"]:
        assert case["answerable"] is False
        assert case["expected_doc_ids"] == []
        assert "expected_facts" not in case


# --- expected_tool_calls: never derivable, never emitted -------------------------------------------


def test_expected_tool_calls_is_never_emitted(cases) -> None:
    for case in cases:
        assert "expected_tool_calls" not in case


# --- determinism: two runs byte identical -----------------------------------------------------------


def test_generate_cases_is_deterministic_two_runs_byte_identical() -> None:
    first = generator.generate_cases()
    second = generator.generate_cases()
    assert generator.to_jsonl(first) == generator.to_jsonl(second)
    assert first == second


def test_write_jsonl_two_writes_byte_identical(tmp_path) -> None:
    cases = generator.generate_cases()
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    generator.write_jsonl(cases, first_path)
    generator.write_jsonl(cases, second_path)
    assert first_path.read_bytes() == second_path.read_bytes()
    # Git native: one case object per line, newline terminated, no trailing blank line noise.
    lines = first_path.read_text().splitlines()
    assert len(lines) == len(cases)
    for line in lines:
        json.loads(line)  # every line is standalone valid JSON


def test_generate_cases_is_deterministic_across_separate_processes(tmp_path: Path) -> None:
    """The two same process byte identical tests above cannot detect hash order instability: both
    calls share one interpreter, hence one PYTHONHASHSEED, hence the same (possibly wrong) set/dict
    iteration order twice in a row. This spawns `python -m dataset_tools.generator` as two genuinely
    separate OS processes, each with its own explicit, different PYTHONHASHSEED, and byte compares
    the JSONL each writes. Closes the exact blind spot the SP7 planning digest names by name (a walk
    that iterates a bare set/dict without a stable sort looks deterministic in a same process test
    but reorders across a real process boundary). Proven against the reviewer's tamper (a guarded
    `set(entity.fields)` that only reorders entities after the first) in a disposable worktree: red
    with the tamper present, green once reverted, while all other determinism tests in this file
    stayed green throughout the tamper (SP7 T1 review, Important 1)."""
    repo_root = Path(__file__).resolve().parents[2]
    out_a = tmp_path / "hash_seed_a.jsonl"
    out_b = tmp_path / "hash_seed_b.jsonl"
    python_path = os.pathsep.join(
        [str(repo_root / "backend"), str(repo_root / "testing" / "harness"), str(repo_root)]
    )
    for seed, out_path in (("1", out_a), ("2", out_b)):
        env = {**os.environ, "PYTHONPATH": python_path, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-m", "dataset_tools.generator", "--out", str(out_path)],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    assert out_a.read_bytes() == out_b.read_bytes()


def test_case_ordering_follows_registry_file_order(cases) -> None:
    # core.yaml's first entity is plan-fiber-500, whose first declared field is "name": the walk
    # order is registry file order, never a set/dict reordering (D16's determinism contract).
    assert cases[0]["case_id"] == "gen-fact-plan-fiber-500-name"


# --- split is provisional, Task 4 owns final stratified assignment ---------------------------------


def test_split_defaults_to_dev_for_every_generated_case(cases) -> None:
    assert all(case["split"] == "dev" for case in cases)


# --- origin: promoted is accepted, even though nothing here produces one ---------------------------


def test_origin_promoted_case_is_accepted_by_the_schema(schema, cases) -> None:
    promoted = {**cases[0], "origin": "promoted", "source_trace_id": "trace-live-0001"}
    jsonschema.validate(promoted, schema)


# --- provenance_join module: direct unit coverage of the join primitives ---------------------------


def test_provenance_join_docs_for_fact_is_sorted_and_exact() -> None:
    index = provenance_join.load_corpus_index(CORPUS_DIR)
    docs = provenance_join.docs_for_fact(index, "plan-fiber-100:monthly_price")
    assert docs == tuple(sorted(docs))
    assert set(docs) == {
        "doc-plan_page-plan-fiber-100",
        "doc-troubleshooting-device-modem-d3--plan-fiber-100",
        "doc-troubleshooting-device-router-ax2--plan-fiber-100",
    }


def test_provenance_join_unplaced_fact_returns_empty() -> None:
    index = provenance_join.load_corpus_index(CORPUS_DIR)
    assert provenance_join.docs_for_fact(index, "plan-quantum-5g:name") == ()
    assert provenance_join.chunk_ids_for_fact(index, "plan-quantum-5g:name") == ()
    assert provenance_join.doc_type_for_fact(index, "plan-quantum-5g:name") is None


# --- span overlap rule cross check: two production copies bound on shared fixtures (SP7 T1 review,
# --- Minor 1). `provenance_join._span_overlaps` and `chunker._entity_ids_for_span` independently
# --- implement the identical inequality (`placement_start < chunk_end and chunk_start <
# --- placement_end`), a deliberate mirror per the module docstring ("import or mirror, never a
# --- third variant"). Nothing previously asserted the two agree over arbitrary spans; this binds
# --- them so a future change to one (e.g. a `<=` bugfix) that is not mirrored to the other fails
# --- here, loudly, instead of silently drifting. A third copy of the same inequality lives in this
# --- test file's own `_independent_chunk_ids` helper above; consolidating all three into one shared
# --- helper is a future review candidate, not done here (other SP7 rails are live). -----------------

_SPAN_OVERLAP_FIXTURES: tuple[tuple[tuple[int, int], tuple[int, int], bool], ...] = (
    ((0, 100), (10, 20), True),      # placement fully inside the chunk
    ((0, 100), (90, 150), True),     # placement overlaps the chunk's tail
    ((50, 100), (0, 60), True),      # placement overlaps the chunk's head
    ((0, 100), (0, 100), True),      # identical spans
    ((10, 20), (10, 20), True),      # identical, single point sized
    ((0, 50), (50, 100), False),     # touching at the boundary: exclusive per the strict "<"
    ((50, 100), (0, 50), False),     # touching at the boundary, reversed
    ((0, 100), (100, 200), False),   # disjoint, adjacent
    ((0, 100), (200, 300), False),   # disjoint, far apart
)


@pytest.mark.parametrize("chunk_span, placement_span, expect_overlap", _SPAN_OVERLAP_FIXTURES)
def test_provenance_join_and_chunker_agree_on_span_overlap_membership(
    chunk_span: tuple[int, int], placement_span: tuple[int, int], expect_overlap: bool
) -> None:
    via_provenance_join = provenance_join._span_overlaps(chunk_span, placement_span)
    entity_ids = chunker._entity_ids_for_span(
        chunk_span, [{"fact_ref": "spancheck:field", "span": list(placement_span)}]
    )
    via_chunker = bool(entity_ids)
    assert via_provenance_join == expect_overlap
    assert via_chunker == expect_overlap
    assert via_provenance_join == via_chunker
