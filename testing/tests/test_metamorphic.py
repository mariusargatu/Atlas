"""Metamorphic testing: assert RELATIONSHIPS between the outputs of related inputs, the way you test
non-deterministic evals with classical tools. Two forms here:

1. Pure metamorphic properties on the retriever, driven by Hypothesis (property-based): retrieval is
   invariant to query casing and surrounding whitespace, and adding a document that shares no token
   with the query never changes what comes back. Deterministic, generated over many inputs.

2. Paraphrase invariance through the REAL agent graph, behind the replay cassette: across a frozen
   family of paraphrasings of the cold-open question, the render guard's verdict is invariant — every
   false "no-contract" claim is caught for a customer who has a term, and every one RENDERS for a
   customer who does not. The oracle, not the wording, decides. The model is frozen into cassettes, so
   a non-deterministic system is tested by a deterministic fixture (article 9).
"""
from __future__ import annotations

import pytest
from hypothesis import given, strategies as st
from langchain_core.messages import HumanMessage

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from replay.gateway import GatewayChatModel

from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph
from atlas.ports.knowledge import Chunk
from evals.datasets.metamorphic_golden import PARAPHRASE_FAMILY
from evals.datasets.retrieval_golden import RETRIEVAL_CORPUS, RETRIEVAL_GOLDEN

_QUERIES = [case.query for case in RETRIEVAL_GOLDEN]


def _ids(chunks) -> list[str]:
    return [c.doc_id for c in chunks]


# --- 1. pure metamorphic properties on the retriever (Hypothesis) ---

@given(query=st.sampled_from(_QUERIES))
def test_retrieval_is_invariant_to_query_casing(query):
    retriever = InMemoryRetriever(RETRIEVAL_CORPUS)
    assert _ids(retriever.search(query)) == _ids(retriever.search(query.upper()))


@given(query=st.sampled_from(_QUERIES), pad=st.text(alphabet=" \t\n", max_size=6))
def test_retrieval_is_invariant_to_surrounding_whitespace(query, pad):
    retriever = InMemoryRetriever(RETRIEVAL_CORPUS)
    assert _ids(retriever.search(query)) == _ids(retriever.search(f"{pad}{query}{pad}"))


@given(query=st.sampled_from(_QUERIES), token=st.text(alphabet="qxz", min_size=3, max_size=6))
def test_adding_a_nonoverlapping_doc_never_changes_retrieval(query, token):
    # a distractor whose single token is a qxz-string shares no word with any English query,
    # so it can never be retrieved and can never displace a real hit
    distractor = Chunk(f"distractor-{token}", token)
    base = InMemoryRetriever(RETRIEVAL_CORPUS)
    plus = InMemoryRetriever(RETRIEVAL_CORPUS + [distractor])
    assert _ids(base.search(query)) == _ids(plus.search(query))


# --- 2. paraphrase invariance through the real atlas_graph, behind the replay cassette ---

def _graph(cassette_dir):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=cassette_dir, mode="replay")
    return build_atlas_graph(gw, IdFactory("idem"), ActionsBackend(IdFactory("ref")), new_checkpointer())


@pytest.mark.asyncio
async def test_the_render_guard_catches_the_false_claim_across_every_paraphrasing(tmp_path, seed_cassette):
    """The invariant: however the cold-open question and its false answer are worded, the render guard
    holds it for the legacy customer. One seed golden case, a whole family of derived cases, one
    relation checked deterministically."""
    graph = _graph(tmp_path)
    for i, (question, answer) in enumerate(PARAPHRASE_FAMILY):
        user = HumanMessage(question)
        seed_cassette(tmp_path, [user], {"content": answer, "tool_calls": []})
        out = await graph.ainvoke(
            {"messages": [user], "session": {"customer_id": "cust_legacy_term"}},
            {"configurable": {"thread_id": f"mm-legacy-{i}"}},
        )
        assert out["final_response"].startswith("[safe handoff]"), f"paraphrase not caught: {question!r}"


@pytest.mark.asyncio
async def test_the_same_paraphrases_render_for_the_current_customer(tmp_path, seed_cassette):
    """The dual relation that proves the ORACLE decides, not the words: the identical family renders
    for a customer whose plan really is term-free. Same inputs, opposite verdicts, one oracle."""
    graph = _graph(tmp_path)
    for i, (question, answer) in enumerate(PARAPHRASE_FAMILY):
        user = HumanMessage(question)
        seed_cassette(tmp_path, [user], {"content": answer, "tool_calls": []})
        out = await graph.ainvoke(
            {"messages": [user], "session": {"customer_id": "cust_current"}},
            {"configurable": {"thread_id": f"mm-current-{i}"}},
        )
        assert out["final_response"] == answer, f"should render for the current customer: {question!r}"
