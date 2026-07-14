"""`matrix.variants` (SP9 final review, finding I1): the naive vs agentic vs graph comparison stage,
hermetic. `orchestration/agentic_rag.py` and `orchestration/graph_rag.py` were real, tested
subgraphs with no caller anywhere in `testing/harness/matrix/`; this stage is that caller. Two
scenarios, each isolating ONE variant's own real mechanism against the SAME shared
retriever/reranker/graph/gateway fixtures, so a difference between rows is genuinely the mechanism,
never a different corpus or model:

  - graph traversal recovers an entity-linked chunk naive's fixed top-k truncation buries behind
    noise (agentic's rewrite does not help here either: the retriever is query-invariant).
  - the agentic bound rewrite recovers a chunk a query-keyed retriever only returns for the
    rewritten query (graph's traversal does not help here either: the graph is empty).

Together they prove the three variants are genuinely measured, not just invoked identically.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from replay.gateway import GatewayChatModel

from atlas.adapters.cassette_reranker import CassetteReranker
from atlas.adapters.inmemory_graph import InMemoryGraph
from atlas.orchestration.agentic_rag import _rewrite_query, build_generate_prompt
from atlas.ports.knowledge import Chunk
from atlas.ports.knowledge_graph import Edge, Node

from matrix.cases import MatrixCase
from matrix.variants import (
    AGENTIC,
    GRAPH,
    NAIVE,
    VARIANT_IDS,
    VariantsConfig,
    run_variant_comparison,
    variant_comparison_rows,
)


# ---------------------------------------------------------------------------------------------
# scenario A: graph traversal recovers a chunk naive's fixed truncation buries behind noise
# ---------------------------------------------------------------------------------------------

_QUERY_A = "does my Fiber 100 plan have any extra charges"
# agentic's one bound rewrite fires here too (the retriever is query-invariant, so the retry still
# sees the noise chunks, but `state["query"]` stays rewritten for the eventual budget-exhausted
# generate call -- its own prompt is keyed on THIS text, not the original).
_REWRITTEN_A = _rewrite_query(_QUERY_A)

_U1 = Chunk(chunk_id="u1", doc_id="u1", text="Our support line is open from nine to five on weekdays.")
_U2 = Chunk(chunk_id="u2", doc_id="u2", text="You can pay your bill online or by mail.")
_U3 = Chunk(chunk_id="u3", doc_id="u3", text="Router firmware updates happen automatically overnight.")
_PLAN = Chunk(
    chunk_id="plan-1", doc_id="plan-1", text="The Fiber 100 plan is our mid tier fiber offering.",
    entity_ids=("plan-fiber-100",),
)
_FEE_A = Chunk(
    chunk_id="fee-1", doc_id="fee-1", text="A regional surcharge applies to hardware you keep on premises.",
    entity_ids=("fee-equipment-rental",),
)

_GRAPH_A = InMemoryGraph(
    [
        Node("plan-fiber-100", "plan", "Fiber 100"),
        Node("region-north", "region", "North Region"),
        Node("fee-equipment-rental", "fee", "Equipment Rental Fee"),
    ],
    [
        Edge("plan-fiber-100", "available_in", "region-north"),
        Edge("region-north", "overrides_fee", "fee-equipment-rental"),
    ],
)

_CASE_A = MatrixCase(
    case_id="case-graph-benefit",
    query=_QUERY_A,
    relevant_doc_ids=frozenset({"fee-1"}),
    expected_facts=({"fact_id": "fee-equipment-rental:amount", "value": "10.00"},),
    query_entity_ids=frozenset({"fee-equipment-rental"}),
)

_NAIVE_ANSWER_A = "Our support line is open from nine to five on weekdays."  # verbatim: faithful, wrong fee
_GRAPH_ANSWER_A = "Yes, the equipment rental fee is 10.00."


class _FixedPoolRetriever:
    """A canned wide candidate pool, ignoring the query entirely (mirrors
    `test_graph_rag_variant.py`'s own `_AllChunksRetriever`): all three variants share this ONE
    retriever, so the only reason their chunks ever differ is each variant's own mechanism."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def search_chunks(self, query: str, k: int, config) -> list[Chunk]:
        return list(self._chunks)[:k]


@pytest.fixture
def _cassette_dir_a(tmp_path, seed_cassette):
    naive_prompt = build_generate_prompt(_QUERY_A, [_U1, _U2, _U3], corrective=False)
    # agentic's own bound rewrite fires (the retry still sees the noise chunks, budget spent), so its
    # eventual generate call is keyed on the REWRITTEN query text, a distinct prompt from naive's.
    agentic_prompt = build_generate_prompt(_REWRITTEN_A, [_U1, _U2, _U3], corrective=False)
    graph_prompt = build_generate_prompt(_QUERY_A, [_PLAN, _FEE_A], corrective=False)
    seed_cassette(tmp_path, [HumanMessage(naive_prompt)], {"content": _NAIVE_ANSWER_A, "tool_calls": []})
    seed_cassette(tmp_path, [HumanMessage(agentic_prompt)], {"content": _NAIVE_ANSWER_A, "tool_calls": []})
    seed_cassette(tmp_path, [HumanMessage(graph_prompt)], {"content": _GRAPH_ANSWER_A, "tool_calls": []})
    return tmp_path


@pytest.mark.asyncio
async def test_graph_traversal_recovers_a_chunk_naive_and_agentic_both_bury_behind_noise(_cassette_dir_a):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=_cassette_dir_a, mode="replay")
    retriever = _FixedPoolRetriever([_U1, _U2, _U3, _PLAN, _FEE_A])
    result = await run_variant_comparison(
        [_CASE_A], retriever=retriever, reranker=CassetteReranker({}), graph=_GRAPH_A, gateway=gw, k=3,
    )
    naive_case = result[NAIVE].per_case["case-graph-benefit"]
    agentic_case = result[AGENTIC].per_case["case-graph-benefit"]
    graph_case = result[GRAPH].per_case["case-graph-benefit"]

    # naive's fixed top-3 truncation never reaches the two entity-linked chunks (buried at rank 4-5)
    assert naive_case.doc_ids == ("u1", "u2", "u3")
    assert naive_case.correctness == 0.0
    assert naive_case.recall_at_k == 0.0

    # agentic's rewrite does not help here: the retriever is query-invariant, so the retry sees the
    # same buried candidates and, budget spent, hands off to generate with the same noise chunks.
    assert agentic_case.doc_ids == ("u1", "u2", "u3")
    assert agentic_case.correctness == 0.0

    # graph's traversal (Fiber 100 -> region-north -> fee-equipment-rental) joins the candidate pool
    # down to exactly the two entity-linked chunks BEFORE truncation, so both survive.
    assert graph_case.doc_ids == ("plan-1", "fee-1")
    assert graph_case.correctness == 1.0
    assert graph_case.recall_at_k == 1.0


# ---------------------------------------------------------------------------------------------
# scenario B: the agentic bound rewrite recovers a chunk a query-keyed retriever only returns for
# the REWRITTEN query; naive and graph both stay on the original (wrong) chunk.
# ---------------------------------------------------------------------------------------------

_QUERY_B = "what is the early termination fee"
_REWRITTEN_B = _rewrite_query(_QUERY_B)

_UNLINKED_B = Chunk(
    chunk_id="hours-1", doc_id="hours-1", text="Our support line is open from nine to five on weekdays.",
    entity_ids=(),
)
_FEE_B = Chunk(
    chunk_id="fee-1", doc_id="fee-1",
    text="The early termination fee is forty pounds if you cancel before your contract ends.",
    entity_ids=("fee-early-termination",),
)

_CASE_B = MatrixCase(
    case_id="case-agentic-benefit",
    query=_QUERY_B,
    relevant_doc_ids=frozenset({"fee-1"}),
    expected_facts=({"fact_id": "fee-early-termination:amount", "value": "forty pounds"},),
    query_entity_ids=frozenset({"fee-early-termination"}),
)

_NAIVE_ANSWER_B = "Our support line is open from nine to five on weekdays."
_AGENTIC_ANSWER_B = "The early termination fee is forty pounds."


class _QueryKeyedRetriever:
    """Query string keyed canned responses (mirrors `test_agentic_rag.py`'s own `_FakeRetriever`):
    only a caller that actually rewrites the query text ever reaches the fee chunk."""

    def __init__(self, table: dict[str, list[Chunk]]) -> None:
        self._table = table

    def search_chunks(self, query: str, k: int, config) -> list[Chunk]:
        return self._table.get(query, [])[:k]


@pytest.fixture
def _cassette_dir_b(tmp_path, seed_cassette):
    naive_prompt = build_generate_prompt(_QUERY_B, [_UNLINKED_B], corrective=False)
    agentic_prompt = build_generate_prompt(_REWRITTEN_B, [_FEE_B], corrective=False)
    seed_cassette(tmp_path, [HumanMessage(naive_prompt)], {"content": _NAIVE_ANSWER_B, "tool_calls": []})
    seed_cassette(tmp_path, [HumanMessage(agentic_prompt)], {"content": _AGENTIC_ANSWER_B, "tool_calls": []})
    return tmp_path


@pytest.mark.asyncio
async def test_agentic_rewrite_recovers_a_chunk_naive_and_graph_both_miss(_cassette_dir_b):
    assert _REWRITTEN_B != _QUERY_B  # the deterministic transform actually changed the string
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=_cassette_dir_b, mode="replay")
    retriever = _QueryKeyedRetriever({_QUERY_B: [_UNLINKED_B], _REWRITTEN_B: [_FEE_B]})
    result = await run_variant_comparison(
        [_CASE_B], retriever=retriever, reranker=CassetteReranker({}), graph=InMemoryGraph((), ()),
        gateway=gw, k=3,
    )
    naive_case = result[NAIVE].per_case["case-agentic-benefit"]
    agentic_case = result[AGENTIC].per_case["case-agentic-benefit"]
    graph_case = result[GRAPH].per_case["case-agentic-benefit"]

    assert naive_case.doc_ids == ("hours-1",)
    assert naive_case.correctness == 0.0

    # no entity resolves against an empty graph, so collect_chunks falls back to the un-joined pool
    # -- the SAME single (wrong) chunk naive got, since neither variant ever rewrites the query.
    assert graph_case.doc_ids == ("hours-1",)
    assert graph_case.correctness == 0.0

    assert agentic_case.doc_ids == ("fee-1",)
    assert agentic_case.correctness == 1.0
    assert agentic_case.recall_at_k == 1.0


# ---------------------------------------------------------------------------------------------
# shape, determinism, and the manifest row helper
# ---------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_variant_comparison_returns_exactly_the_three_named_variant_ids(_cassette_dir_a):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=_cassette_dir_a, mode="replay")
    retriever = _FixedPoolRetriever([_U1, _U2, _U3, _PLAN, _FEE_A])
    result = await run_variant_comparison(
        [_CASE_A], retriever=retriever, reranker=CassetteReranker({}), graph=_GRAPH_A, gateway=gw,
    )
    assert set(result) == set(VARIANT_IDS) == {"naive", "agentic", "graph"}


@pytest.mark.asyncio
async def test_two_runs_over_the_same_inputs_produce_identical_rows(_cassette_dir_a):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=_cassette_dir_a, mode="replay")
    retriever = _FixedPoolRetriever([_U1, _U2, _U3, _PLAN, _FEE_A])
    result_1 = await run_variant_comparison(
        [_CASE_A], retriever=retriever, reranker=CassetteReranker({}), graph=_GRAPH_A, gateway=gw,
    )
    result_2 = await run_variant_comparison(
        [_CASE_A], retriever=retriever, reranker=CassetteReranker({}), graph=_GRAPH_A, gateway=gw,
    )
    assert variant_comparison_rows(result_1) == variant_comparison_rows(result_2)


def test_variant_comparison_rows_is_sorted_by_variant_id_for_determinism():
    from matrix.variants import VariantResult

    results = {
        NAIVE: VariantResult(NAIVE, {}),
        AGENTIC: VariantResult(AGENTIC, {}),
        GRAPH: VariantResult(GRAPH, {}),
    }
    rows = variant_comparison_rows(results)
    assert [r["variant_id"] for r in rows] == ["agentic", "graph", "naive"]


def test_variants_config_bundles_every_fixture_run_variant_comparison_needs():
    config = VariantsConfig(
        retriever=_FixedPoolRetriever([]), reranker=CassetteReranker({}), graph=InMemoryGraph((), ()),
        gateway=GatewayChatModel(model_id="claude-test", cassette_dir="/tmp", mode="replay"),
    )
    assert config.k == 3
