"""The agentic RAG variant (SP9 task 1), hermetic: a narrow subgraph beside atlas_graph.py, never a
branch inside it (D6). Exercises the graph end to end against `InMemoryRetriever` +
`GatewayChatModel` in replay mode (keyless, no network): the happy path (graded relevant on the
first pass, faithful on the first generate), the one bound rewrite (CRAG grading fails, the retry
fires exactly once, a tamper that forces perpetual low grades still stops at the budget), and the
faithfulness gate (an unfaithful first answer triggers exactly one regenerate, and a still
unfaithful second answer ships with a disclosure, never a silent pass and never a third attempt).

`grade_documents`'s entity_id overlap arithmetic is cross checked against
`testing.harness.quality.agent_metrics.citation_precision_recall` directly (same fixture, same
formula) since `agentic_rag.py` itself cannot import that harness module (the product/harness
import lint boundary, test_import_lint.py, is one way: harness may import backend, never the
reverse) -- the closest this boundary allows to a literal shared implementation.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from replay.gateway import GatewayChatModel

from quality.agent_metrics import citation_precision_recall

from atlas.adapters.cassette_reranker import CassetteReranker
from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.domain.budget import Budget, check_budget
from atlas.domain.retrieval import RetrievalConfig
from atlas.orchestration.agentic_rag import (
    _AGENTIC_BUDGET,
    _RETRIEVAL_TOOLS,
    _entity_overlap_precision_recall,
    _is_faithful,
    _rewrite_query,
    build_agentic_rag_graph,
    build_generate_prompt,
)
from atlas.ports.knowledge import Chunk

# A tiny fixture corpus carrying real entity_ids (unlike domain.corpus.CORPUS's toy 3 docs, which
# never populate entity_ids at all), so CRAG grading has something real to overlap against.
_ROUTER_CHUNK = Chunk(
    chunk_id="router-1", parent_id="router-1", doc_id="router-1",
    text="If your router light blinks orange, restart it by holding the power button for ten seconds.",
    entity_ids=("device_router",),
)
_FEE_CHUNK = Chunk(
    chunk_id="fee-1", parent_id="fee-1", doc_id="fee-1",
    text="The early termination fee is forty pounds if you cancel before your contract ends.",
    entity_ids=("fee_early_termination",),
)
_UNLINKED_CHUNK = Chunk(
    chunk_id="hours-1", parent_id="hours-1", doc_id="hours-1",
    text="Our support line is open from nine to five on weekdays.",
    entity_ids=(),
)


class _FakeRetriever:
    """Query string keyed canned responses, decoupled from `InMemoryRetriever`'s own keyword
    overlap heuristic so a rewrite test controls exactly what the rewritten query returns without
    having to hand predict overlap counts on a multi-word rewrite."""

    def __init__(self, table: dict[str, list[Chunk]]) -> None:
        self._table = table

    def search_chunks(self, query: str, k: int, config: RetrievalConfig) -> list[Chunk]:
        return self._table.get(query, [])[:k]


def _graph(cassette_dir, *, retriever=None, reranker=None):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=cassette_dir, mode="replay")
    return build_agentic_rag_graph(gw, retriever=retriever or InMemoryRetriever([_ROUTER_CHUNK]), reranker=reranker)


# ---------------------------------------------------------------------------------------------
# identical signature to naive: query in, chunks + answer out
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_query_in_chunks_and_answer_out(tmp_path, seed_cassette):
    query = "router light is orange"
    prompt = build_generate_prompt(query, [_ROUTER_CHUNK], corrective=False)
    answer = "Restart it by holding the power button for ten seconds."
    seed_cassette(tmp_path, [HumanMessage(prompt)], {"content": answer, "tool_calls": []})
    graph = _graph(tmp_path, retriever=InMemoryRetriever([_ROUTER_CHUNK]))
    out = await graph.ainvoke({"query": query})
    assert out["chunks"] == [_ROUTER_CHUNK]
    assert out["answer"] == answer
    assert out["tools_called"] == (
        "route_query", "retrieve", "rerank", "grade_documents", "generate", "check_faithfulness",
    )
    assert out["faithful"] is True


@pytest.mark.asyncio
async def test_no_query_entity_ids_supplied_is_graded_ok_vacuously_never_forces_a_rewrite(tmp_path, seed_cassette):
    """A live query with no golden entity linking (query_entity_ids defaults to ()) has nothing to
    grade against; CRAG grading must not force an unresolvable rewrite loop on every ordinary call."""
    query = "router light is orange"
    prompt = build_generate_prompt(query, [_ROUTER_CHUNK], corrective=False)
    seed_cassette(tmp_path, [HumanMessage(prompt)], {"content": "Restart it.", "tool_calls": []})
    graph = _graph(tmp_path, retriever=InMemoryRetriever([_ROUTER_CHUNK]))
    out = await graph.ainvoke({"query": query})
    assert "rewrite_query" not in out["tools_called"]
    assert out["graded_ok"] is True


# ---------------------------------------------------------------------------------------------
# grade_documents: CRAG style entity_id overlap, cross checked against quality/agent_metrics
# ---------------------------------------------------------------------------------------------


def test_grade_documents_overlap_matches_citation_precision_recall_on_a_shared_fixture():
    corpus = [_ROUTER_CHUNK, _FEE_CHUNK, _UNLINKED_CHUNK]
    cited = frozenset(eid for c in corpus for eid in c.entity_ids)
    expected = frozenset({"fee_early_termination"})
    metrics = citation_precision_recall(list(cited), expected)
    own = _entity_overlap_precision_recall(cited, expected)
    assert (metrics.precision, metrics.recall) == own  # the two formulas never silently drift apart
    assert metrics.recall == 1.0  # fee_early_termination IS among the retrieved chunks' entity_ids
    assert metrics.precision == pytest.approx(1 / 2)  # one of the two carried entity_ids matches


# ---------------------------------------------------------------------------------------------
# the one bound rewrite: fires at most once, tamper forces perpetual low grades, stops at budget
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rewrite_fires_once_and_recovers_when_the_retry_grades_relevant(tmp_path, seed_cassette):
    """The first pass retrieves an entity-unlinked chunk (grade fails); the one bound rewrite fires,
    the SAME retrieve/rerank helpers run again against the rewritten query, and this time the entity
    linked chunk comes back (grade passes). Never a third retrieve."""
    query = "what is the early termination fee"
    rewritten = _rewrite_query(query)
    assert rewritten != query  # the deterministic transform actually changed the string
    retriever = _FakeRetriever({query: [_UNLINKED_CHUNK], rewritten: [_FEE_CHUNK]})
    prompt = build_generate_prompt(rewritten, [_FEE_CHUNK], corrective=False)
    answer = "The early termination fee is forty pounds."
    seed_cassette(tmp_path, [HumanMessage(prompt)], {"content": answer, "tool_calls": []})
    graph = _graph(tmp_path, retriever=retriever)
    out = await graph.ainvoke({"query": query, "query_entity_ids": ("fee_early_termination",)})
    assert out["tools_called"].count("retrieve") == 2
    assert out["tools_called"].count("rewrite_query") == 1
    assert out["tools_called"].count("generate") == 1  # no infinite loop: generate ran exactly once
    assert out["graded_ok"] is True  # the retry's own grade genuinely passed
    assert out["answer"] == answer


@pytest.mark.asyncio
async def test_tamper_forcing_perpetual_low_grades_still_stops_at_the_budget(tmp_path, seed_cassette):
    """The tamper: BOTH the original and the rewritten query return the same entity-unlinked chunk,
    so grade_documents fails on every single pass, forever. The one bound rewrite (Budget, not a
    new counter) must still cap the loop at exactly one retry and hand off to generate, never spin."""
    query = "what is the early termination fee"
    rewritten = _rewrite_query(query)
    retriever = _FakeRetriever({query: [_UNLINKED_CHUNK], rewritten: [_UNLINKED_CHUNK]})
    prompt = build_generate_prompt(rewritten, [_UNLINKED_CHUNK], corrective=False)
    # grounded in _UNLINKED_CHUNK's own text (faithful on the first pass): this test is about the
    # retrieval/rewrite budget, not the faithfulness gate, so the generate call must not ALSO
    # trigger an (unrelated) regenerate and consume a second, unseeded cassette entry.
    answer = "Our support line is open from nine to five on weekdays."
    seed_cassette(tmp_path, [HumanMessage(prompt)], {"content": answer, "tool_calls": []})
    graph = _graph(tmp_path, retriever=retriever)
    out = await graph.ainvoke({"query": query, "query_entity_ids": ("fee_early_termination",)})
    assert out["tools_called"].count("retrieve") == 2  # initial + exactly one bound retry, never more
    assert out["tools_called"].count("rewrite_query") == 1
    assert out["graded_ok"] is False  # the tamper's own honesty: the grade genuinely never passed
    assert out["tools_called"].count("generate") == 1  # generate ran once, budget exhausted rather than stuck
    assert out["answer"] == answer


def test_agentic_budget_reuses_the_shared_check_budget_shape_not_a_new_counter():
    """The exact mechanism `atlas.domain.budget.check_budget` already is, over a narrower Budget
    instance, never a bespoke counter: two retrieve rounds is the boundary, a third is refused."""
    assert isinstance(_AGENTIC_BUDGET, Budget)
    ok_after_one_retry = check_budget(("retrieve", "retrieve"), _AGENTIC_BUDGET, retrieval_tools=_RETRIEVAL_TOOLS)
    assert ok_after_one_retry.ok
    refused_on_a_third = check_budget(
        ("retrieve", "retrieve", "retrieve"), _AGENTIC_BUDGET, retrieval_tools=_RETRIEVAL_TOOLS
    )
    assert not refused_on_a_third.ok


def test_the_agentic_budget_has_no_hand_counted_node_total_to_keep_in_sync():
    """Both limits police the same quantity, retrieve rounds, because `route_after_grade` passes
    `check_budget` only the retrieval subsequence. The `max_tool_calls=12` this replaced was a
    human recount of the longest node path, due to go stale the first time a node was added, and
    structurally incapable of binding (the reachable sequences are 5 and 9 nodes)."""
    assert _AGENTIC_BUDGET.max_tool_calls == _AGENTIC_BUDGET.max_retrieval_rounds
    # a long non retrieval node history must not consume the budget: only retrieves count
    long_history = ("route_query", "rerank", "grade_documents", "generate", "check_faithfulness") * 4
    assert check_budget(
        tuple(t for t in long_history + ("retrieve",) if t in _RETRIEVAL_TOOLS),
        _AGENTIC_BUDGET,
        retrieval_tools=_RETRIEVAL_TOOLS,
    ).ok


def test_the_deployed_retrieval_widths_are_declared_once_in_the_domain():
    """`K_FUSED`/`K_FINAL` live in `atlas.domain.retrieval` and every consumer imports them. Three
    modules used to redeclare the pair under comments asserting they matched, with nothing checking
    it, and `matrix.embedders` had already drifted (calling k=5 "the production DEPLOYED_K")."""
    from atlas.domain.retrieval import K_FINAL, K_FUSED
    from atlas.mcp_servers.knowledge_server import DEPLOYED_K
    from atlas.orchestration import agentic_rag, graph_rag

    assert DEPLOYED_K is K_FINAL
    assert (agentic_rag.K_FUSED, agentic_rag.K_FINAL) == (K_FUSED, K_FINAL)
    assert (graph_rag.K_FUSED, graph_rag.K_FINAL) == (K_FUSED, K_FINAL)
    for module in (agentic_rag, graph_rag):
        assert not hasattr(module, "_K_FUSED") and not hasattr(module, "_K_FINAL")


# ---------------------------------------------------------------------------------------------
# the faithfulness gate: reference based, deterministic, regenerate once then disclose
# ---------------------------------------------------------------------------------------------


def test_is_faithful_reference_based_deterministic_no_judge():
    grounded_answer = "Restart it by holding the power button for ten seconds."
    assert _is_faithful(grounded_answer, [_ROUTER_CHUNK]) is True
    hallucinated_answer = "Unplug it and mail it to our regional warehouse for a full refund today."
    assert _is_faithful(hallucinated_answer, [_ROUTER_CHUNK]) is False


@pytest.mark.asyncio
async def test_faithfulness_gate_triggers_one_regenerate_then_discloses(tmp_path, seed_cassette):
    query = "router light is orange"
    first_prompt = build_generate_prompt(query, [_ROUTER_CHUNK], corrective=False)
    second_prompt = build_generate_prompt(query, [_ROUTER_CHUNK], corrective=True)
    unfaithful = "Unplug it and mail it to our regional warehouse for a full refund today."
    seed_cassette(tmp_path, [HumanMessage(first_prompt)], {"content": unfaithful, "tool_calls": []})
    seed_cassette(tmp_path, [HumanMessage(second_prompt)], {"content": unfaithful, "tool_calls": []})
    graph = _graph(tmp_path, retriever=InMemoryRetriever([_ROUTER_CHUNK]))
    out = await graph.ainvoke({"query": query})
    assert out["tools_called"].count("generate") == 2  # exactly one regenerate, never a third attempt
    assert out["tools_called"].count("check_faithfulness") == 2
    assert out["faithful"] is False
    assert out["answer"].startswith(unfaithful)
    assert out["answer"] != unfaithful  # the disclosure suffix was appended, never a silent ship
    assert out["disclosed"] is True


@pytest.mark.asyncio
async def test_faithfulness_gate_passes_clean_on_a_faithful_first_answer_no_regenerate(tmp_path, seed_cassette):
    query = "router light is orange"
    prompt = build_generate_prompt(query, [_ROUTER_CHUNK], corrective=False)
    faithful_answer = "Restart it by holding the power button for ten seconds."
    seed_cassette(tmp_path, [HumanMessage(prompt)], {"content": faithful_answer, "tool_calls": []})
    graph = _graph(tmp_path, retriever=InMemoryRetriever([_ROUTER_CHUNK]))
    out = await graph.ainvoke({"query": query})
    assert out["tools_called"].count("generate") == 1
    assert out["answer"] == faithful_answer  # unmodified: no disclosure suffix on a faithful answer
    assert out.get("disclosed") is not True


# ---------------------------------------------------------------------------------------------
# rerank: the previously unwired Reranker port, first wiring, reordering is honoured
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rerank_node_uses_the_reranker_port_reordering_takes_effect(tmp_path, seed_cassette):
    reranker = CassetteReranker({"router light is orange": {"router-1": 1.0, "fee-1": 5.0}})
    retriever = InMemoryRetriever([_ROUTER_CHUNK, _FEE_CHUNK])
    query = "router light is orange"
    reordered = [_FEE_CHUNK, _ROUTER_CHUNK]  # fee-1 outscores router-1 per the cassette table above
    prompt = build_generate_prompt(query, reordered, corrective=False)
    # grounded in _FEE_CHUNK's own text: this test is about rerank ordering, not the faithfulness
    # gate, so the answer must not ALSO trigger an (unrelated) regenerate.
    answer = "The early termination fee is forty pounds."
    seed_cassette(tmp_path, [HumanMessage(prompt)], {"content": answer, "tool_calls": []})
    graph = _graph(tmp_path, retriever=retriever, reranker=reranker)
    out = await graph.ainvoke({"query": query})
    assert [c.chunk_id for c in out["chunks"]] == ["fee-1", "router-1"]
    assert out["tools_called"].count("generate") == 1
