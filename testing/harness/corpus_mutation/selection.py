"""WHICH registry fact this lane mutates (deterministic, never random, never order dependent), a
THROWAWAY mutated copy of the registry, and WHICH rendered documents that one changed fact actually
touches. See the package docstring for the full picture and the Task 6 (metamorphic) contrast.

`select_mutation`/`mutate_registry`/`affected_doc_ids` are pure: no filesystem write, no network, no
wall clock. `render_corpus` (called by `affected_doc_ids`' own callers, never here) is the one real
library call this module's downstream users make; this module itself never renders anything.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from corpus_tools.registry import Registry, load_registry
from corpus_tools.render import RenderedDoc

# Single sourced from corpus_tools.build's own path constants (never redeclared as a second,
# silently divergent copy): the same registry files and template directory the committed
# corpus-0.1.1 render itself is built from.
from corpus_tools.build import CORE, GENERATED, TEMPLATES

DEFAULT_REGISTRY_PATHS: tuple[Path, ...] = (CORE, GENERATED)

# conflict-daniel-contract: the SAME contradiction Task 6's metamorphic lane and Task 3's
# manufactured failures already seed from (D32's own "no new corruption logic invented" preference
# for reusing one well understood registry fixture across sub projects, one registry fact doing
# triple duty). Its winning_fact, contract_term-daniel-2025:contract_months, is an int field, which
# is what lets select_mutation derive a guaranteed-different new_value below without inventing any
# corruption logic of its own either.
DEFAULT_CONTRADICTION_ID = "conflict-daniel-contract"

__all__ = [
    "CORE",
    "DEFAULT_CONTRADICTION_ID",
    "DEFAULT_REGISTRY_PATHS",
    "GENERATED",
    "TEMPLATES",
    "FactMutation",
    "MutationSelectionError",
    "affected_doc_ids",
    "mutate_registry",
    "select_mutation",
]


class MutationSelectionError(ValueError):
    """The registry does not carry the contradiction, or the fact, this lane expects to mutate."""


@dataclass(frozen=True)
class FactMutation:
    """ONE registry fact, chosen deterministically, with the value this lane mutates it to.
    `question` is the SAME natural wording used to probe the agent before and after the mutation
    (unlike Task 6's metamorphic families, this lane never varies the wording: the wording stays
    fixed and the TRUTH changes, the exact inverse of Task 6's own invariant)."""

    contradiction_id: str
    fact_ref: str
    old_value: object
    new_value: object
    question: str


def select_mutation(
    reg: Registry | None = None, *, contradiction_id: str = DEFAULT_CONTRADICTION_ID
) -> FactMutation:
    """Deterministically choose ONE registry fact to mutate: the winning fact of `contradiction_id`.
    Never random, never order dependent: `reg.contradictions` may list contradictions in any file
    order, but this function looks the ONE requested id up by name rather than picking "the first"
    or "a random" contradiction, so calling it twice on the same registry, or on the same registry
    loaded from files in a different order, returns byte identical output.

    `reg` defaults to the committed registry (`corpus/registry/core.yaml` +
    `generated_variants.yaml`, `DEFAULT_REGISTRY_PATHS`); hermetic tests may pass a fixture registry
    instead, mirroring `judge.provisional.manufactured_cases`' own default-loading convenience.
    """
    if reg is None:
        reg = load_registry(list(DEFAULT_REGISTRY_PATHS))

    try:
        contradiction = next(c for c in reg.contradictions if c.id == contradiction_id)
    except StopIteration:
        raise MutationSelectionError(f"registry has no contradiction {contradiction_id!r}") from None

    entity_id, _, field = contradiction.winning_fact.partition(":")
    old_value = reg.entity(entity_id).fields[field]
    if not isinstance(old_value, int) or isinstance(old_value, bool):
        raise MutationSelectionError(
            f"{contradiction.winning_fact} is {old_value!r} ({type(old_value).__name__}); "
            "select_mutation only knows how to derive a guaranteed-different value for an int fact"
        )
    # +12 (another plausible whole-year contract length) is deterministic and, for every contract
    # length actually on record in this registry (0, 12, 24), never collides with the old value.
    new_value = old_value + 12

    return FactMutation(
        contradiction_id=contradiction.id,
        fact_ref=contradiction.winning_fact,
        old_value=old_value,
        new_value=new_value,
        question=contradiction.question_hint or contradiction.id,
    )


def mutate_registry(reg: Registry, mutation: FactMutation) -> Registry:
    """A THROWAWAY copy of `reg` with exactly ONE entity field changed to `mutation.new_value`.
    Pure and immutable (`dataclasses.replace` only, never an in place field assignment): `reg` and
    every `Entity` it holds are left byte identical, so a caller's own `reg` stays safe to compare
    the mutated render against afterward. Never writes to `corpus/registry/*.yaml`; the mutation
    lives only in the returned, in-memory `Registry` object."""
    entity_id, _, field = mutation.fact_ref.partition(":")
    target = reg.entity(entity_id)
    mutated_entity = replace(target, fields={**target.fields, field: mutation.new_value})
    mutated_entities = tuple(mutated_entity if e.id == entity_id else e for e in reg.entities)
    return replace(reg, entities=mutated_entities)


def affected_doc_ids(docs: Sequence[RenderedDoc], fact_ref: str) -> frozenset[str]:
    """Which rendered docs actually express `fact_ref`: the "only the affected documents" half of
    this lane. `corpus_tools.render.render_corpus` still renders every document in the mutated
    registry (the library's own contract; this lane never reimplements it or asks it for a partial
    render), but only the documents whose OWN placements cite the mutated fact need to be written
    into the ephemeral corpus_version and re indexed."""
    return frozenset(doc.doc_id for doc in docs if any(p.fact_ref == fact_ref for p in doc.placements))
