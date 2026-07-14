"""`matrix.ollama_spot_check`, hermetic (SP9 task 7): a small sample of Ollama qwen2.5:7b matrix
answers as ONE MORE label items JSONL, reusing SP8's existing HITL adjudication surface end to end
-- `labeling.generate_label_set.write_label_items` (the writer) and
`atlas.label_routes.build_label_router`/`atlas.adapters.label_store.LabelStore` (the page's own
backend), never a second labeling surface. No keys, no network, no live Ollama daemon anywhere:
`GenerationCell`/`RetrievalConfigResult` are hand built fixtures, the SAME shapes
`matrix.generators.run_generation_cell`/`matrix.select.all_retrieval_configs` already produce.
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from determinism.sources import FrozenClock

from atlas.adapters.label_store import LabelStore
from atlas.label_routes import build_label_router

from labeling.generate_label_set import write_label_items

from matrix.cases import MatrixCase
from matrix.generators import GenerationCell
from matrix.ollama_spot_check import SOURCE, build_ollama_spot_check_items
from matrix.select import RetrievalConfigResult

_OLLAMA_ID = "ollama-qwen2.5:7b"

_CASE_A = MatrixCase(
    "case-a", "Is my plan contract free?", frozenset({"d1"}), ({"fact_id": "a:contract", "value": "0"},),
)
_CASE_B = MatrixCase(
    "case-b", "What is the data cap?", frozenset({"d2"}), ({"fact_id": "b:cap", "value": "unlimited"},),
)
_CASE_C_UNANSWERED = MatrixCase("case-c", "Never answered by this cell", frozenset(), ())

_CASES = (_CASE_A, _CASE_B, _CASE_C_UNANSWERED)


def _config() -> RetrievalConfigResult:
    candidates = {
        "case-a": (
            {
                "chunk_id": "d1c1", "parent_id": None, "doc_id": "d1", "doc_version": "v1", "doc_type": "faq",
                "heading_path": [], "char_span": [0, 10], "text": "Your plan has no minimum term.",
                "entity_ids": [], "score": 0.9,
            },
        ),
        "case-b": (
            {
                "chunk_id": "d2c1", "parent_id": None, "doc_id": "d2", "doc_version": "v1", "doc_type": "faq",
                "heading_path": [], "char_span": [0, 10], "text": "Data is unlimited on this plan.",
                "entity_ids": [], "score": 0.8,
            },
        ),
    }
    return RetrievalConfigResult(config_id="bge-m3-local", candidates=candidates, ndcg_point=0.9)


def _cell(*, empty_answer_for_b: bool = False) -> GenerationCell:
    per_case = {
        "case-a": {"answer": "Your plan is contract free.", "correctness": 1.0, "panel_label": "pass", "panel_disagreed": False, "panel_votes": [1, 1, 1]},
        "case-b": {"answer": "" if empty_answer_for_b else "Your data cap is unlimited.", "correctness": 1.0, "panel_label": "pass", "panel_disagreed": False, "panel_votes": [1, 1, 1]},
    }
    return GenerationCell(config_id="bge-m3-local", generator_component_id=_OLLAMA_ID, per_case=per_case, prompt_hash="ph")


# ---- well formed items ----------------------------------------------------------------------------


def test_build_items_produces_one_well_formed_item_per_answered_case():
    items = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-1")
    assert len(items) == 2  # case-c has no answer at all: never fabricated
    for item in items:
        assert set(item) == {"case_id", "trace_id", "question", "answer", "retrieved_chunks", "registry_facts", "source"}
        assert item["source"] == SOURCE
        assert item["trace_id"]
        assert item["answer"]


def test_items_carry_the_real_question_and_answer():
    items = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-1")
    by_case = {i["case_id"]: i for i in items}
    assert by_case["case-a"]["question"] == "Is my plan contract free?"
    assert by_case["case-a"]["answer"] == "Your plan is contract free."


def test_items_carry_retrieved_chunks_in_the_label_route_shape():
    items = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-1")
    by_case = {i["case_id"]: i for i in items}
    chunks = by_case["case-a"]["retrieved_chunks"]
    assert chunks == [{"doc_id": "d1", "chunk_id": "d1c1", "text": "Your plan has no minimum term.", "score": 0.9}]


def test_items_carry_registry_facts_stringified():
    items = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-1")
    by_case = {i["case_id"]: i for i in items}
    assert by_case["case-a"]["registry_facts"] == [{"fact_id": "a:contract", "value": "0"}]


# ---- never fabricates: an unanswered or empty answer case is skipped -----------------------------


def test_a_case_absent_from_the_cell_is_skipped_never_fabricated():
    items = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-1")
    assert "case-c" not in {i["case_id"] for i in items}


def test_a_case_with_an_empty_answer_is_skipped_never_fabricated():
    items = build_ollama_spot_check_items(_config(), _cell(empty_answer_for_b=True), _CASES, run_id="run-1")
    assert {i["case_id"] for i in items} == {"case-a"}


# ---- fixed seed order, never shuffled -------------------------------------------------------------


def test_items_preserve_the_cases_own_file_order():
    reordered = (_CASE_B, _CASE_A, _CASE_C_UNANSWERED)
    items = build_ollama_spot_check_items(_config(), _cell(), reordered, run_id="run-1")
    assert [i["case_id"] for i in items] == ["case-b", "case-a"]


# ---- limit: a small sample, first N produced, never a random sample ------------------------------


def test_limit_caps_the_sample_to_the_first_n_answered_cases():
    items = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-1", limit=1)
    assert len(items) == 1
    assert items[0]["case_id"] == "case-a"


def test_limit_none_returns_every_answered_case():
    items = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-1", limit=None)
    assert len(items) == 2


# ---- trace_id: deterministic, unique, and honestly NOT a live trace_root --------------------------


def test_trace_id_is_deterministic_for_the_same_inputs():
    items1 = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-1")
    items2 = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-1")
    assert [i["trace_id"] for i in items1] == [i["trace_id"] for i in items2]


def test_trace_id_differs_across_runs_never_colliding():
    items1 = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-1")
    items2 = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-2")
    by_case_1 = {i["case_id"]: i["trace_id"] for i in items1}
    by_case_2 = {i["case_id"]: i["trace_id"] for i in items2}
    for case_id in by_case_1:
        assert by_case_1[case_id] != by_case_2[case_id]


def test_trace_id_differs_across_cases_within_one_run():
    items = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-1")
    assert len({i["trace_id"] for i in items}) == len(items)


# ---- the produced JSONL is well formed and loads in the EXISTING HITL label store -----------------


@pytest.mark.asyncio
async def test_sample_jsonl_is_well_formed_and_loads_in_the_existing_label_store(tmp_path):
    items = build_ollama_spot_check_items(_config(), _cell(), _CASES, run_id="run-1")
    out_path = tmp_path / "label_items.ollama_spot_check.jsonl"
    write_label_items(items, out_path)  # the SAME writer generate_label_items's own real path uses

    # Well formed: one JSON object per line, every one loadable.
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)

    # Loads through the EXISTING HITL backend route, no new labeling surface.
    store = LabelStore(tmp_path / "labels.jsonl", FrozenClock(datetime.fromisoformat("2026-06-15T12:00:00+00:00")))
    app = FastAPI()
    app.include_router(build_label_router(items_path=out_path, store=store))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/labels/items")
        assert r.status_code == 200
        body = r.json()
        assert body["progress"] == {"labeled": 0, "total": 2}
        assert [i["case_id"] for i in body["items"]] == ["case-a", "case-b"]
        assert body["items"][0]["source"] == SOURCE

        trace_id = body["items"][0]["trace_id"]
        post = await client.post(
            "/labels", json={"trace_id": trace_id, "verdict": "pass", "critique": "Grounded in the cited passage."}
        )
        assert post.status_code == 200
        assert post.json()["progress"] == {"labeled": 1, "total": 2}

    records = store.read_all()
    assert len(records) == 1
    assert records[0].trace_id == trace_id
