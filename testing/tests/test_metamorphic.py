"""Metamorphic testing: assert relationships between the outputs of related inputs, the way you
test not deterministic evals with classical tools. Two forms here:

1. Pure metamorphic properties on the retriever, driven by Hypothesis (property based): retrieval is
   invariant to query casing and surrounding whitespace, and adding a document that shares no token
   with the query never changes what comes back. Deterministic, generated over many inputs.
   Unrelated to the registry seeded families below (a separate, generic labelled corpus,
   `evals.datasets.retrieval_golden`, also used by `test_ir_metrics.py`/`test_reranking.py`).

2. Registry seeded families (SP8 task 6, D32), rebuilt from scratch against `corpus/registry/
   core.yaml`'s `conflict-daniel-contract`, superseding the pre rewrite `evals/datasets/
   metamorphic_golden.py`'s toy corpus paraphrase family: however a customer asks "is my plan
   contract free" (a natural paraphrase, a typo, or pure surface noise), the retrieval stack keeps
   surfacing Daniel's own contract chunk, and the one correctly grounded answer expresses the same
   registry fact regardless of wording. Three deterministic, judge free invariants
   (`metamorphic.report`), checked over the real `InMemoryRetriever` against a small, real content
   stub corpus (`metamorphic.families.STUB_CORPUS`), never a hand frozen "as if retrieved" list.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from corpus_tools.registry import load_registry
from evals.datasets.retrieval_golden import RETRIEVAL_CORPUS, RETRIEVAL_GOLDEN

from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.domain.retrieval import RetrievalConfig
from atlas.ports.knowledge import Chunk

from metamorphic.families import (
    ALL_FAMILIES,
    DRIFTED_ANSWER,
    LOSING_DOC_ID,
    PARAPHRASE_FAMILY,
    QUERY_PERTURBATION_FAMILY,
    STUB_CORPUS,
    STUB_K,
    TYPO_NOISE_FAMILY,
    WINNING_DOC_ID,
    WINNING_FACT_ID,
    WINNING_FACT_VALUE,
)
from metamorphic.report import (
    id_based_retrieval_agreement,
    rank_overlap_floor_holds,
    registry_answer_equivalence_holds,
    run_all_families,
    run_family,
)

_QUERIES = [case.query for case in RETRIEVAL_GOLDEN]
_CONFIG = RetrievalConfig()

CORE_REGISTRY = Path("corpus/registry/core.yaml")


def _ids(chunks) -> list[str]:
    return [c.doc_id for c in chunks]


# --- 1. pure metamorphic properties on the retriever (Hypothesis) ---

@given(query=st.sampled_from(_QUERIES))
def test_retrieval_is_invariant_to_query_casing(query):
    retriever = InMemoryRetriever(RETRIEVAL_CORPUS)
    assert _ids(retriever.search_chunks(query, config=_CONFIG)) == _ids(retriever.search_chunks(query.upper(), config=_CONFIG))


@given(query=st.sampled_from(_QUERIES), pad=st.text(alphabet=" \t\n", max_size=6))
def test_retrieval_is_invariant_to_surrounding_whitespace(query, pad):
    retriever = InMemoryRetriever(RETRIEVAL_CORPUS)
    assert _ids(retriever.search_chunks(query, config=_CONFIG)) == _ids(retriever.search_chunks(f"{pad}{query}{pad}", config=_CONFIG))


@given(query=st.sampled_from(_QUERIES), token=st.text(alphabet="qxz", min_size=3, max_size=6))
def test_adding_a_nonoverlapping_doc_never_changes_retrieval(query, token):
    # a distractor whose single token is a qxz-string shares no word with any English query,
    # so it can never be retrieved and can never displace a real hit
    distractor = Chunk(chunk_id=f"distractor-{token}", doc_id=f"distractor-{token}", text=token)
    base = InMemoryRetriever(RETRIEVAL_CORPUS)
    plus = InMemoryRetriever(RETRIEVAL_CORPUS + [distractor])
    assert _ids(base.search_chunks(query, config=_CONFIG)) == _ids(plus.search_chunks(query, config=_CONFIG))


# --- 2. registry seeded metamorphic families, rebuilt against conflict-daniel-contract (D32) ---


def test_family_ground_truth_matches_the_registry_contradiction():
    """The frozen `WINNING_FACT_ID`/`WINNING_FACT_VALUE` in `families.py` are a committed snapshot,
    never recomputed at import time (the module's own docstring rule); this is the machine check
    that the snapshot has not silently drifted from the real registry, the same
    `conflict-daniel-contract` `judge.provisional.manufactured_cases` derives its own ground truth
    from."""
    registry = load_registry([CORE_REGISTRY])
    contradiction = next(c for c in registry.contradictions if c.id == "conflict-daniel-contract")
    entity_id, _, field = contradiction.winning_fact.partition(":")
    assert entity_id == "contract_term-daniel-2025"
    assert field == "contract_months"
    assert registry.entity(entity_id).fields[field] == WINNING_FACT_VALUE
    assert WINNING_FACT_ID == contradiction.winning_fact


def test_family_ids_are_unique_and_kinds_are_named():
    assert len({family.family_id for family in ALL_FAMILIES}) == len(ALL_FAMILIES)
    assert {family.kind for family in ALL_FAMILIES} == {"paraphrase", "typo_noise", "query_perturbation"}


def test_all_families_hold_every_invariant_over_the_stub_retriever():
    """The positive case: replaying every frozen family against the real (stub) retriever, all
    three D32 invariants hold for all three families."""
    report = run_all_families()
    assert report.all_hold
    assert len(report.results) == 3
    for result in report.results:
        assert result.id_agreement
        assert result.rank_overlap_holds
        assert result.answer_equivalence


def test_paraphrase_family_retrieves_the_winning_chunk_for_every_wording():
    result = run_family(PARAPHRASE_FAMILY)
    assert result.id_agreement
    for retrieved in result.retrieved_by_member:
        assert WINNING_DOC_ID in retrieved
        if LOSING_DOC_ID in retrieved:
            assert retrieved.index(WINNING_DOC_ID) <= retrieved.index(LOSING_DOC_ID)


def test_typo_noise_family_still_clears_its_own_looser_floor():
    result = run_family(TYPO_NOISE_FAMILY)
    assert result.id_agreement  # the ground truth chunk survives every typo
    assert result.rank_overlap_holds
    # a real typo does cost the other retrieved chunk
    assert result.min_rank_overlap == pytest.approx(0.5)


def test_query_perturbation_family_holds_exact_agreement_not_merely_a_floor():
    """Pure surface noise (casing, whitespace, punctuation) changes no meaningful token, so this
    family's floor is the strongest of the three: exact retrieval agreement, not merely >= floor."""
    result = run_family(QUERY_PERTURBATION_FAMILY)
    assert QUERY_PERTURBATION_FAMILY.rank_overlap_floor == 1.0
    assert result.min_rank_overlap == 1.0
    assert len({result.retrieved_by_member[0]} | set(result.retrieved_by_member)) == 1  # every member identical


# --- 3. the invariants have teeth: each one can genuinely fail, proven directly ---


def test_id_based_retrieval_agreement_fails_when_a_member_misses_the_ground_truth():
    retrieved_by_member = (("a", "b"), ("a", WINNING_DOC_ID), ("c", "d"))  # the third member misses it
    assert not id_based_retrieval_agreement(retrieved_by_member, WINNING_DOC_ID, 2)
    assert id_based_retrieval_agreement((("a", WINNING_DOC_ID), (WINNING_DOC_ID, "b")), WINNING_DOC_ID, 2)


def test_rank_overlap_floor_fails_when_members_diverge_too_much():
    scrambled = (("a", "b"), ("c", "d"), ("e", "f"))  # nothing shared between any pair
    assert not rank_overlap_floor_holds(scrambled, 2, floor=0.5)
    identical = (("a", "b"), ("a", "b"), ("a", "b"))
    assert rank_overlap_floor_holds(identical, 2, floor=1.0)


def test_registry_answer_equivalence_fails_on_a_drifted_answer():
    """The DRIFTED_ANSWER fixture is grounded in the LOSING chunk's own claim instead of the
    winning one -- a plausible answer a model that read the wrong document would give. It never
    mentions "12", so the equivalence check must catch it, proving the check is not vacuously true."""
    assert not registry_answer_equivalence_holds(PARAPHRASE_FAMILY.expected_facts, DRIFTED_ANSWER)
    assert registry_answer_equivalence_holds(PARAPHRASE_FAMILY.expected_facts, PARAPHRASE_FAMILY.answer)


def test_a_family_with_the_drifted_answer_fails_overall():
    drifted_family = replace(PARAPHRASE_FAMILY, answer=DRIFTED_ANSWER)
    result = run_family(drifted_family)
    assert result.id_agreement  # retrieval is untouched, only the answer drifted
    assert not result.answer_equivalence
    assert not result.holds


def test_a_broken_corpus_missing_the_winning_chunk_fails_id_agreement():
    """Drop the winning chunk from the stub corpus entirely (as if ingestion silently lost the
    doc): the retriever can no longer surface it for any wording, and the report says so."""
    broken_corpus = [c for c in STUB_CORPUS if c.doc_id != WINNING_DOC_ID]
    result = run_family(PARAPHRASE_FAMILY, corpus=broken_corpus, k=STUB_K)
    assert not result.id_agreement
    assert not result.holds
