"""`domain.graph_retrieval` (SP9 task 2), hermetic and pure: no adapter, no port, no I/O -- plain
data in, plain data out. `test_import_lint.py` already machine-checks that this module imports no
framework/client and no outer ring; these tests are about the arithmetic itself.
"""
from __future__ import annotations

from atlas.domain.graph_retrieval import collect_chunks_by_entities, extract_candidate_mentions
from atlas.ports.knowledge import Chunk

# --- extract_candidate_mentions: every contiguous word span, punctuation-stripped, de-duplicated ---


def test_unigrams_and_bigrams_are_both_present_for_a_two_word_query() -> None:
    mentions = extract_candidate_mentions("north region", max_words=2)
    assert mentions == ("north", "region", "north region")


def test_candidate_mentions_strip_trailing_punctuation() -> None:
    mentions = extract_candidate_mentions("is equipment rental free?", max_words=2)
    assert "free" in mentions
    assert "free?" not in mentions


def test_candidate_mentions_are_deduplicated_and_order_stable() -> None:
    mentions = extract_candidate_mentions("north region", max_words=3)
    assert len(mentions) == len(set(mentions))  # no repeats
    # unigrams first (left to right), then longer spans, matching the function's own stable order
    assert mentions.index("north") < mentions.index("region") < mentions.index("north region")


def test_a_three_word_entity_name_is_recoverable_at_max_words_three() -> None:
    mentions = extract_candidate_mentions("what is the equipment rental fee", max_words=3)
    assert "equipment rental fee" in mentions


def test_max_words_below_the_entity_name_length_never_produces_it() -> None:
    mentions = extract_candidate_mentions("what is the equipment rental fee", max_words=2)
    assert "equipment rental fee" not in mentions
    assert "equipment rental" in mentions  # the 2-word span still comes through


def test_empty_query_yields_no_candidates() -> None:
    assert extract_candidate_mentions("   ") == ()


# --- collect_chunks_by_entities: the traversal-to-chunks join --------------------------------------

_PLAN_CHUNK = Chunk(chunk_id="c-plan", doc_id="doc-plan", text="Fiber 100 plan details.", entity_ids=("plan-fiber-100",))
_FEE_CHUNK = Chunk(
    chunk_id="c-fee", doc_id="doc-fee", text="Equipment rental fee is 10.", entity_ids=("fee-equipment-rental",)
)
_UNLINKED_CHUNK = Chunk(chunk_id="c-hours", doc_id="doc-hours", text="Support hours nine to five.", entity_ids=())


def test_collect_keeps_only_chunks_overlapping_the_entity_set() -> None:
    chunks = [_PLAN_CHUNK, _FEE_CHUNK, _UNLINKED_CHUNK]
    joined = collect_chunks_by_entities(chunks, frozenset({"fee-equipment-rental"}))
    assert joined == [_FEE_CHUNK]


def test_collect_preserves_the_input_order_never_reorders() -> None:
    chunks = [_FEE_CHUNK, _PLAN_CHUNK]
    joined = collect_chunks_by_entities(chunks, frozenset({"plan-fiber-100", "fee-equipment-rental"}))
    assert joined == [_FEE_CHUNK, _PLAN_CHUNK]  # same order as the input, not entity-set order


def test_collect_with_an_empty_entity_set_returns_nothing_never_raises() -> None:
    chunks = [_PLAN_CHUNK, _FEE_CHUNK]
    assert collect_chunks_by_entities(chunks, frozenset()) == []


def test_collect_with_no_overlap_at_all_returns_nothing() -> None:
    chunks = [_UNLINKED_CHUNK]
    assert collect_chunks_by_entities(chunks, frozenset({"plan-fiber-100"})) == []


def test_a_chunk_with_multiple_entity_ids_matches_on_partial_overlap() -> None:
    multi = Chunk(chunk_id="c-multi", doc_id="doc-multi", text="...", entity_ids=("plan-fiber-100", "region-north"))
    joined = collect_chunks_by_entities([multi], frozenset({"region-north"}))
    assert joined == [multi]
