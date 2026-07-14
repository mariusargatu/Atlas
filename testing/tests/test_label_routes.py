"""The HITL backend label route (SP8 Task 4, label collection half): GET the label item set (fixed
seed order, D30: no managed queue) plus the current progress, POST one adjudicator label appended
to the JSONL. Hermetic: a tiny FastAPI app in memory over a `tmp_path` item file and `LabelStore`,
no real graph, no cassettes needed here (the generator that PRODUCES the item file is tested
separately, `test_generate_label_set.py`).

SP8 Task 4 remainder: `post_label` also mirrors every stored label to a Phoenix annotation client
(D30, `atlas.adapters.phoenix_annotations`). Hermetic here too -- a recording stub stands in for a
live Phoenix client, never a real one; `atlas.label_routes`'s own default (`NullPhoenixAnnotationClient`)
is exercised by the existing tests above that never pass `phoenix_client` at all.
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

_ITEMS = [
    {
        "case_id": "fixture-1", "trace_id": "t1", "question": "Is my plan contract-free?",
        "answer": "Yes.", "retrieved_chunks": [{"doc_id": "d1", "chunk_id": "d1", "text": "contract-free", "score": 0.0}],
        "registry_facts": [{"fact_id": "f1", "value": "contract-free"}],
    },
    {
        "case_id": "fixture-2", "trace_id": "t2", "question": "Data cap?",
        "answer": "No cap.", "retrieved_chunks": [{"doc_id": "d2", "chunk_id": "d2", "text": "unlimited", "score": 0.0}],
        "registry_facts": [{"fact_id": "f2", "value": "unlimited"}],
    },
]


def _items_path(tmp_path):
    path = tmp_path / "items.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for item in _ITEMS:
            fh.write(json.dumps(item) + "\n")
    return path


def _app(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl", FrozenClock(datetime.fromisoformat("2026-06-15T12:00:00+00:00")))
    app = FastAPI()
    app.include_router(build_label_router(items_path=_items_path(tmp_path), store=store))
    return app, store


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_get_items_returns_the_full_item_set_in_fixed_seed_order(tmp_path):
    app, _store = _app(tmp_path)
    async with _client(app) as client:
        r = await client.get("/labels/items")
        assert r.status_code == 200
        body = r.json()
        assert [i["case_id"] for i in body["items"]] == ["fixture-1", "fixture-2"]
        assert body["progress"] == {"labeled": 0, "total": 2}


@pytest.mark.asyncio
async def test_get_items_includes_question_answer_chunks_and_registry_facts(tmp_path):
    app, _store = _app(tmp_path)
    async with _client(app) as client:
        r = await client.get("/labels/items")
        first = r.json()["items"][0]
        assert first["question"] == "Is my plan contract-free?"
        assert first["answer"] == "Yes."
        assert first["retrieved_chunks"][0]["text"] == "contract-free"
        assert first["registry_facts"][0]["value"] == "contract-free"


@pytest.mark.asyncio
async def test_post_label_appends_and_returns_updated_progress(tmp_path):
    app, store = _app(tmp_path)
    async with _client(app) as client:
        r = await client.post(
            "/labels", json={"trace_id": "t1", "verdict": "pass", "critique": "Fully grounded in the cited page."}
        )
        assert r.status_code == 200
        assert r.json()["progress"] == {"labeled": 1, "total": 2}

    records = store.read_all()
    assert len(records) == 1
    assert records[0].trace_id == "t1" and records[0].role == "adjudicator" and records[0].verdict == "pass"


@pytest.mark.asyncio
async def test_post_label_defaults_role_to_adjudicator(tmp_path):
    app, store = _app(tmp_path)
    async with _client(app) as client:
        await client.post("/labels", json={"trace_id": "t1", "verdict": "fail", "critique": "Unsupported claim."})
    assert store.read_all()[0].role == "adjudicator"


@pytest.mark.asyncio
async def test_post_label_progress_counts_distinct_trace_ids_not_raw_lines(tmp_path):
    app, _store = _app(tmp_path)
    async with _client(app) as client:
        await client.post("/labels", json={"trace_id": "t1", "verdict": "pass", "critique": "Grounded."})
        r = await client.post(
            "/labels", json={"trace_id": "t1", "verdict": "fail", "critique": "On review, ungrounded after all."}
        )
        assert r.json()["progress"] == {"labeled": 1, "total": 2}


@pytest.mark.asyncio
async def test_post_label_unknown_trace_id_is_404(tmp_path):
    app, _store = _app(tmp_path)
    async with _client(app) as client:
        r = await client.post("/labels", json={"trace_id": "nope", "verdict": "pass", "critique": "x"})
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_post_label_empty_critique_is_422(tmp_path):
    app, _store = _app(tmp_path)
    async with _client(app) as client:
        r = await client.post("/labels", json={"trace_id": "t1", "verdict": "pass", "critique": "   "})
        assert r.status_code == 422


@pytest.mark.asyncio
async def test_post_label_bad_verdict_is_422(tmp_path):
    app, _store = _app(tmp_path)
    async with _client(app) as client:
        r = await client.post("/labels", json={"trace_id": "t1", "verdict": "maybe", "critique": "x"})
        assert r.status_code == 422


@pytest.mark.asyncio
async def test_progress_reads_fresh_never_cached_across_two_gets(tmp_path):
    app, _store = _app(tmp_path)
    async with _client(app) as client:
        before = (await client.get("/labels/items")).json()["progress"]
        await client.post("/labels", json={"trace_id": "t2", "verdict": "pass", "critique": "Matches the page."})
        after = (await client.get("/labels/items")).json()["progress"]
    assert before == {"labeled": 0, "total": 2}
    assert after == {"labeled": 1, "total": 2}


@pytest.mark.asyncio
async def test_get_items_on_a_missing_item_file_is_an_empty_set_not_an_error(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl", FrozenClock(datetime.fromisoformat("2026-06-15T12:00:00+00:00")))
    app = FastAPI()
    app.include_router(build_label_router(items_path=tmp_path / "does-not-exist.jsonl", store=store))
    async with _client(app) as client:
        r = await client.get("/labels/items")
        assert r.status_code == 200
        assert r.json() == {"items": [], "progress": {"labeled": 0, "total": 0}}


# ---- SP8 Task 4 remainder: mirroring a stored label to Phoenix's annotation API (D30) --------------


class _RecordingPhoenixClient:
    """A stub `PhoenixAnnotationClient`: records every call instead of doing any I/O."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def annotate(self, *, trace_id: str, label: str, score: float, explanation: str) -> None:
        self.calls.append(
            {"trace_id": trace_id, "label": label, "score": score, "explanation": explanation}
        )


def _app_with_phoenix(tmp_path, phoenix_client):
    store = LabelStore(tmp_path / "labels.jsonl", FrozenClock(datetime.fromisoformat("2026-06-15T12:00:00+00:00")))
    app = FastAPI()
    app.include_router(
        build_label_router(items_path=_items_path(tmp_path), store=store, phoenix_client=phoenix_client)
    )
    return app, store


@pytest.mark.asyncio
async def test_post_label_mirrors_the_stored_label_to_the_phoenix_client(tmp_path):
    phoenix_client = _RecordingPhoenixClient()
    app, _store = _app_with_phoenix(tmp_path, phoenix_client)
    async with _client(app) as client:
        r = await client.post(
            "/labels", json={"trace_id": "t1", "verdict": "pass", "critique": "Fully grounded in the cited page."}
        )
        assert r.status_code == 200

    assert len(phoenix_client.calls) == 1
    call = phoenix_client.calls[0]
    assert call["trace_id"] == "t1"
    assert call["label"] == "pass"
    assert call["score"] == 1.0
    assert call["explanation"] == "Fully grounded in the cited page."


@pytest.mark.asyncio
async def test_post_label_mirrors_a_fail_verdict_scored_as_zero(tmp_path):
    phoenix_client = _RecordingPhoenixClient()
    app, _store = _app_with_phoenix(tmp_path, phoenix_client)
    async with _client(app) as client:
        await client.post("/labels", json={"trace_id": "t1", "verdict": "fail", "critique": "Unsupported claim."})

    assert phoenix_client.calls[0]["label"] == "fail"
    assert phoenix_client.calls[0]["score"] == 0.0


@pytest.mark.asyncio
async def test_post_label_never_mirrors_a_rejected_label(tmp_path):
    """A 422 (blank critique, bad verdict) or a 404 (unknown trace_id, covered separately by
    `test_post_label_unknown_trace_id_is_404`) both mean `LabelStore.append` never ran -- the
    mirror must never fire for a label that was never actually stored. This test drives the 422
    (blank critique) branch; the guarantee that the 404 branch never mirrors either rests on
    structure (the route raises before ever reaching the `append` call), not a separate assertion
    here."""
    phoenix_client = _RecordingPhoenixClient()
    app, _store = _app_with_phoenix(tmp_path, phoenix_client)
    async with _client(app) as client:
        r = await client.post("/labels", json={"trace_id": "t1", "verdict": "pass", "critique": "   "})
        assert r.status_code == 422

    assert phoenix_client.calls == []


@pytest.mark.asyncio
async def test_build_label_router_defaults_to_the_null_phoenix_client(tmp_path):
    """No `phoenix_client` argument at all (`server.py`'s own construction today) must still
    succeed -- the null client is the hermetic default, never a required parameter."""
    app, _store = _app(tmp_path)
    async with _client(app) as client:
        r = await client.post("/labels", json={"trace_id": "t1", "verdict": "pass", "critique": "Grounded."})
        assert r.status_code == 200
