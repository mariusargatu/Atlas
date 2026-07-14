"""The batch answer generation script (SP8 Task 4, label collection half): runs the real Atlas
graph over a label set drawn from SP7's seed cases and produces question+answer+retrieved_chunks
items a human labels on the HITL page.

Hermetic path only: every case here goes through the REPLAY gateway against cassettes seeded in
this test (the same `seed_cassette` + `InMemoryRetriever` + `serialize_tool_result` technique
`test_atlas_graph.py::test_answer_path_retrieves_then_grounded_answer_caught_at_render` already
proves out for a two step tool call flow), so this test needs no keys and no network. The REAL
~200 item generation (a live provider + a real retrieval index) is the operator's job, documented
separately, never exercised here.
"""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from determinism.canonical import serialize_tool_result
from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from replay.gateway import GatewayChatModel
from tracing import InMemoryTracer

from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.domain.actions import ActionsBackend
from atlas.domain.retrieval import RetrievalConfig
from atlas.orchestration.atlas_graph import build_atlas_graph

from labeling.generate_label_set import (
    FIXTURE_CASSETTE_DIR,
    FIXTURE_OUT,
    FIXTURE_SEED_CASES,
    build_generation_graph,
    generate_label_items,
    load_seed_cases,
    retrieved_chunks_from_messages,
    write_label_items,
)

_QUERY = "plan contract term cancel fee"


def _seed_two_step_cassette(cassette_dir, question: str, answer: str) -> None:
    """Seeds BOTH cassette entries a search_knowledge turn needs: the decision call (question ->
    a tool call) and the synthesis call (question + the real ToolMessage the retriever actually
    returns -> the final answer). Mirrors test_atlas_graph.py's own established technique exactly,
    so the SECOND cassette key matches what the real InMemoryRetriever produces, never a guess."""
    from replay.cassette_store import seed_cassette

    user = HumanMessage(question)
    toolcall = [{"name": "search_knowledge", "args": {"query": _QUERY}, "id": "k1"}]
    seed_cassette(cassette_dir, [user], {"content": "", "tool_calls": toolcall})

    chunks = InMemoryRetriever().search_chunks(_QUERY, config=RetrievalConfig())
    passages = serialize_tool_result(
        [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": c.score, "text": c.text} for c in chunks]
    )
    ai = AIMessage(content="", tool_calls=toolcall)
    tool_msg = ToolMessage(content=passages, tool_call_id="k1", name="search_knowledge")
    seed_cassette(cassette_dir, [user, ai, tool_msg], {"content": answer, "tool_calls": []})


def _graph(cassette_dir):
    tracer = InMemoryTracer()
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=cassette_dir, mode="replay")
    backend = ActionsBackend(IdFactory("ref"))
    graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer(), tracer=tracer)
    return graph, tracer


_CASES = [
    {
        "case_id": "fixture-case-1",
        "turns": [{"user": "What is the name of plan-fiber-500?"}],
        "expected_facts": [{"fact_id": "plan-fiber-500:name", "value": "Fiber 500"}],
    },
    {
        "case_id": "fixture-case-2",
        "turns": [{"user": "Is my plan contract-free?"}],
        "expected_facts": [{"fact_id": "plan-fiber-500:contract_months", "value": 0}],
    },
]


@pytest.mark.asyncio
async def test_generate_label_items_produces_well_formed_items(tmp_path):
    _seed_two_step_cassette(tmp_path, _CASES[0]["turns"][0]["user"], "Fiber 500.")
    _seed_two_step_cassette(tmp_path, _CASES[1]["turns"][0]["user"], "Your plan has no minimum term.")
    graph, _tracer = _graph(tmp_path)

    items = await generate_label_items(graph, _CASES)

    assert len(items) == 2
    for item, case in zip(items, _CASES):
        assert item["case_id"] == case["case_id"]
        assert item["question"] == case["turns"][0]["user"]
        assert item["answer"]
        assert item["trace_id"]
        assert item["retrieved_chunks"], "the search_knowledge tool ran and returned real chunks"
        for chunk in item["retrieved_chunks"]:
            assert chunk["doc_id"]
            assert "text" in chunk
        assert item["registry_facts"] == [
            {"fact_id": f["fact_id"], "value": str(f["value"])} for f in case["expected_facts"]
        ]


@pytest.mark.asyncio
async def test_generate_label_items_uses_fixed_seed_order_not_a_shuffled_queue(tmp_path):
    _seed_two_step_cassette(tmp_path, _CASES[0]["turns"][0]["user"], "Fiber 500.")
    _seed_two_step_cassette(tmp_path, _CASES[1]["turns"][0]["user"], "Your plan has no minimum term.")
    graph, _tracer = _graph(tmp_path)

    items = await generate_label_items(graph, _CASES)
    assert [i["case_id"] for i in items] == ["fixture-case-1", "fixture-case-2"]


@pytest.mark.asyncio
async def test_generate_label_items_skips_a_case_with_no_final_answer_never_fabricates_one(tmp_path):
    """A case whose turn never resolves to a final_response (e.g. a write proposal awaiting
    confirmation) is skipped, not padded with an invented answer -- the module's own "never
    fabricate a label item" rule."""
    from replay.cassette_store import seed_cassette

    question = "Switch me to the fast plan"
    seed_cassette(
        tmp_path, [HumanMessage(question)],
        {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]},
    )
    graph, _tracer = _graph(tmp_path)
    cases = [{"case_id": "fixture-write", "turns": [{"user": question}], "expected_facts": []}]

    items = await generate_label_items(graph, cases)
    assert items == []


def test_retrieved_chunks_from_messages_filters_to_knowledge_tool_messages_only():
    passages = json.dumps([{"doc_id": "d1", "chunk_id": "c1", "score": 1.0, "text": "hello"}])
    messages = [
        HumanMessage("q"),
        AIMessage(content="", tool_calls=[{"name": "get_bill", "args": {}, "id": "b1"}]),
        ToolMessage(content=json.dumps({"period": "2026-01"}), tool_call_id="b1", name="get_bill"),
        AIMessage(content="", tool_calls=[{"name": "search_knowledge", "args": {"query": "q"}, "id": "k1"}]),
        ToolMessage(content=passages, tool_call_id="k1", name="search_knowledge"),
    ]
    chunks = retrieved_chunks_from_messages(messages)
    assert chunks == [{"doc_id": "d1", "chunk_id": "c1", "score": 1.0, "text": "hello"}]


def test_retrieved_chunks_from_messages_dedupes_the_same_doc_chunk_pair():
    passages = json.dumps([{"doc_id": "d1", "chunk_id": "c1", "score": 1.0, "text": "hello"}])
    messages = [
        ToolMessage(content=passages, tool_call_id="k1", name="search_knowledge"),
        ToolMessage(content=passages, tool_call_id="k2", name="search_knowledge"),
    ]
    chunks = retrieved_chunks_from_messages(messages)
    assert len(chunks) == 1


def test_load_seed_cases_reads_in_file_order_a_fixed_seed_order_never_shuffled(tmp_path):
    path = tmp_path / "seed_cases.jsonl"
    path.write_text(
        '{"case_id": "b"}\n{"case_id": "a"}\n{"case_id": "c"}\n', encoding="utf-8"
    )
    cases = load_seed_cases(path)
    assert [c["case_id"] for c in cases] == ["b", "a", "c"]


def test_load_seed_cases_respects_a_limit_taking_the_first_n_in_file_order(tmp_path):
    path = tmp_path / "seed_cases.jsonl"
    path.write_text('{"case_id": "b"}\n{"case_id": "a"}\n{"case_id": "c"}\n', encoding="utf-8")
    cases = load_seed_cases(path, limit=2)
    assert [c["case_id"] for c in cases] == ["b", "a"]


def test_load_seed_cases_skips_blank_lines(tmp_path):
    path = tmp_path / "seed_cases.jsonl"
    path.write_text('{"case_id": "b"}\n\n{"case_id": "a"}\n', encoding="utf-8")
    cases = load_seed_cases(path)
    assert [c["case_id"] for c in cases] == ["b", "a"]


def test_write_label_items_round_trips_plain_json_never_canonical_float_tags(tmp_path):
    """`score` must come back out as a plain float a Pydantic `score: float` field can parse, not
    the `"F:0.0"` tag `determinism.canonical.canonical_json` would apply."""
    items = [{"case_id": "x", "retrieved_chunks": [{"doc_id": "d1", "score": 0.0}]}]
    out = tmp_path / "items.jsonl"
    write_label_items(items, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["retrieved_chunks"][0]["score"] == 0.0
    assert isinstance(loaded["retrieved_chunks"][0]["score"], float)


def test_write_label_items_is_byte_reproducible(tmp_path):
    items = [{"case_id": "x", "answer": "hi"}, {"case_id": "y", "answer": "bye"}]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_label_items(items, a)
    write_label_items(items, b)
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.asyncio
async def test_the_default_hermetic_invocation_reproduces_the_committed_fixture_byte_for_byte(tmp_path):
    """The exact recipe `task label:generate` runs (`FIXTURE_SEED_CASES` against
    `FIXTURE_CASSETTE_DIR`, `source="fixture"`) must reproduce the committed
    `label_items.fixture.jsonl` byte for byte -- the honesty check behind this module's own claim
    that a bare hermetic run reproduces the fixture, not just a hand wave in a docstring."""
    cases = load_seed_cases(FIXTURE_SEED_CASES)
    graph, _tracer = build_generation_graph("replay", FIXTURE_CASSETTE_DIR)
    items = await generate_label_items(graph, cases, source="fixture")

    out = tmp_path / "reproduced.jsonl"
    write_label_items(items, out)

    assert out.read_bytes() == FIXTURE_OUT.read_bytes()
