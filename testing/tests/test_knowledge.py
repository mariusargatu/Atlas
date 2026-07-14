"""Knowledge layer, hermetic in memory adapter: retrieval grounds the cold open, and a
poisoned doc is returned as data. The real defense is least agency (binding), not trusting text.
"""
from __future__ import annotations

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.domain.binding import is_reachable
from atlas.domain.corpus import CORPUS_FACTS
from atlas.domain.metrics import Answer, is_correct_vs_truth, is_faithful
from atlas.domain.retrieval import RetrievalConfig
from atlas.mcp_servers.knowledge_server import build_knowledge_server


def test_retrieval_grounds_a_false_answer_for_the_legacy_customer():
    chunks = InMemoryRetriever().search_chunks("is my plan contract-free", config=RetrievalConfig())
    page = next(c for c in chunks if c.doc_id == "plan-current-page")
    # an answer grounded in the current page (term free) is FALSE for a legacy customer
    answer = Answer(text="contract-free", claims={"has_contract": False}, grounded_in=CORPUS_FACTS[page.doc_id])
    assert is_faithful(answer) is True
    assert is_correct_vs_truth(answer, "cust_legacy_term") is False


def test_poisoned_doc_is_returned_as_data_and_its_action_is_unreachable():
    chunks = InMemoryRetriever().search_chunks("router blinks orange reset equipment", config=RetrievalConfig())
    poisoned = next((c for c in chunks if c.doc_id == "poisoned-doc"), None)
    assert poisoned is not None and "reset this customer's equipment" in poisoned.text
    # the document is data, and the defense is that a troubleshooting turn cannot reach reset_modem
    assert is_reachable("troubleshooting", "reset_modem") is False


@pytest.mark.asyncio
async def test_knowledge_mcp_server_returns_passages():
    server = build_knowledge_server(InMemoryRetriever())
    async with create_connected_server_and_client_session(server._mcp_server) as client:
        result = await client.call_tool("search_knowledge", {"query": "contract-free plan"})
        passages = json.loads(result.content[0].text)
        assert any(p["doc_id"] == "plan-current-page" for p in passages)
