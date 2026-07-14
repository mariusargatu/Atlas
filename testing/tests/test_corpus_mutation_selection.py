"""corpus_mutation.selection, hermetic (SP8 task 7): the mutation selection logic (WHICH registry
fact this lane mutates, deterministically), a throwaway mutated registry copy that never touches the
committed corpus/registry/*.yaml files, and which rendered documents the mutated fact actually
touches. No render, no re index, no live call anywhere in this file: those belong to the live/burst
operator lane (corpus_mutation/__main__.py), never to this hermetic test.
"""
from __future__ import annotations

import pytest
from corpus_tools.registry import load_registry
from corpus_tools.render import Placement, RenderedDoc, render_corpus

from corpus_mutation.selection import (
    DEFAULT_CONTRADICTION_ID,
    DEFAULT_REGISTRY_PATHS,
    TEMPLATES,
    FactMutation,
    MutationSelectionError,
    affected_doc_ids,
    mutate_registry,
    select_mutation,
)


def _reg():
    return load_registry(list(DEFAULT_REGISTRY_PATHS))


# ---- select_mutation: WHICH fact, deterministic, never random -----------------------------------


def test_select_mutation_targets_conflict_daniel_contracts_winning_fact():
    mutation = select_mutation(_reg())
    assert mutation.contradiction_id == "conflict-daniel-contract"
    assert mutation.fact_ref == "contract_term-daniel-2025:contract_months"
    assert mutation.old_value == 12
    assert mutation.new_value == 24
    assert mutation.question == "is my plan contract free"


def test_select_mutation_is_deterministic_across_repeated_calls():
    reg = _reg()
    assert select_mutation(reg) == select_mutation(reg)


def test_select_mutation_new_value_always_differs_from_old_value():
    mutation = select_mutation(_reg())
    assert mutation.new_value != mutation.old_value


def test_select_mutation_rejects_an_unknown_contradiction_id():
    with pytest.raises(MutationSelectionError):
        select_mutation(_reg(), contradiction_id="no-such-contradiction")


def test_select_mutation_defaults_to_loading_the_committed_registry_when_none_is_given():
    mutation = select_mutation()
    assert mutation.fact_ref == "contract_term-daniel-2025:contract_months"


def test_default_contradiction_id_matches_the_constant_select_mutation_actually_uses():
    # Pins the module's own default parameter to the exported constant, so a future edit that
    # changes one but not the other fails loudly here rather than drifting silently.
    assert select_mutation(_reg()).contradiction_id == DEFAULT_CONTRADICTION_ID


# ---- mutate_registry: a THROWAWAY copy, exactly one field changed, never mutates the input ------


def test_mutate_registry_changes_only_the_targeted_field():
    reg = _reg()
    mutation = select_mutation(reg)
    mutated = mutate_registry(reg, mutation)
    entity_id, _, field = mutation.fact_ref.partition(":")
    assert mutated.entity(entity_id).fields[field] == mutation.new_value


def test_mutate_registry_never_mutates_the_input_registry():
    reg = _reg()
    mutation = select_mutation(reg)
    entity_id, _, field = mutation.fact_ref.partition(":")
    original_value = reg.entity(entity_id).fields[field]

    mutate_registry(reg, mutation)

    assert reg.entity(entity_id).fields[field] == original_value


def test_mutate_registry_leaves_every_other_entity_byte_identical():
    reg = _reg()
    mutation = select_mutation(reg)
    mutated = mutate_registry(reg, mutation)
    entity_id, _, _ = mutation.fact_ref.partition(":")
    for original, changed in zip(reg.entities, mutated.entities, strict=True):
        if original.id == entity_id:
            continue
        assert original == changed


def test_mutate_registry_leaves_edges_and_contradictions_untouched():
    reg = _reg()
    mutation = select_mutation(reg)
    mutated = mutate_registry(reg, mutation)
    assert mutated.edges == reg.edges
    assert mutated.contradictions == reg.contradictions


def test_mutate_registry_returns_a_new_registry_object_not_the_same_instance():
    reg = _reg()
    mutation = select_mutation(reg)
    mutated = mutate_registry(reg, mutation)
    entity_id = mutation.fact_ref.partition(":")[0]
    assert mutated is not reg
    assert mutated.entity(entity_id) is not reg.entity(entity_id)


# ---- affected_doc_ids: only the documents the mutated fact actually touches ----------------------


def _doc(doc_id: str, *fact_refs: str) -> RenderedDoc:
    placements = tuple(Placement(fact_ref=ref, value="x", span=(0, 1)) for ref in fact_refs)
    return RenderedDoc(doc_id=doc_id, doc_type="contract_terms", text="irrelevant", placements=placements)


def test_affected_doc_ids_selects_only_docs_citing_the_mutated_fact():
    docs = (
        _doc("doc-a", "contract_term-daniel-2025:contract_months"),
        _doc("doc-b", "plan-fiber-100:contract_months"),
        _doc("doc-c", "contract_term-daniel-2025:contract_months", "contract_term-daniel-2025:vintage_year"),
    )
    affected = affected_doc_ids(docs, "contract_term-daniel-2025:contract_months")
    assert affected == frozenset({"doc-a", "doc-c"})


def test_affected_doc_ids_is_empty_when_nothing_cites_the_fact():
    docs = (_doc("doc-a", "plan-fiber-100:contract_months"),)
    assert affected_doc_ids(docs, "contract_term-daniel-2025:contract_months") == frozenset()


def test_affected_doc_ids_over_a_real_render_matches_exactly_the_one_contract_terms_doc():
    # The one place this file calls the REAL rendering library (corpus_tools.render.render_corpus,
    # never reimplemented): proves affected_doc_ids' selection against real render output, not just
    # hand built RenderedDoc fixtures. Still fully hermetic: render_corpus is pure template
    # rendering, no network, no live index, no re-indexing.
    reg = _reg()
    mutation = select_mutation(reg)
    mutated_reg = mutate_registry(reg, mutation)
    docs = render_corpus(mutated_reg, TEMPLATES, seed=1)
    affected = affected_doc_ids(docs, mutation.fact_ref)
    assert affected == frozenset({"doc-contract_terms-contract_term-daniel-2025"})


def test_fact_mutation_is_a_dataclass_instance():
    mutation = select_mutation(_reg())
    assert isinstance(mutation, FactMutation)


def test_fact_mutation_equality_is_by_value_not_identity():
    reg = _reg()
    assert select_mutation(reg) == select_mutation(reg)
    assert select_mutation(reg) is not select_mutation(reg)
