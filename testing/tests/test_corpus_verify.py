"""The verifier: no rendered claim may drift from the registry, and both conflict sides must exist."""

from __future__ import annotations

from dataclasses import replace

import pytest
from corpus_tools import registry, render, verify
from .fixtures.corpus_expectations import CORE, GENERATED, TEMPLATES


@pytest.fixture(scope="module")
def reg() -> registry.Registry:
    return registry.load_registry([CORE, GENERATED])


@pytest.fixture(scope="module")
def docs(reg: registry.Registry) -> tuple[render.RenderedDoc, ...]:
    return render.render_corpus(reg, TEMPLATES, seed=7)


def test_committed_pipeline_verifies_clean(docs, reg) -> None:
    assert verify.verify_corpus(docs, reg) == ()


def test_a_drifted_price_is_caught(docs, reg) -> None:
    tampered = list(docs)
    victim = tampered[0]
    placement = victim.placements[0]
    drifted = replace(placement, value=placement.value + "9")
    tampered[0] = replace(victim, placements=(drifted, *victim.placements[1:]))
    violations = verify.verify_corpus(tuple(tampered), reg)
    assert any(placement.fact_ref in v for v in violations)


def test_a_leaked_hidden_entity_is_caught(docs, reg) -> None:
    leaked = replace(docs[0], text=docs[0].text + "\nTry our new Quantum 5G plan!")
    violations = verify.verify_corpus((leaked, *docs[1:]), reg)
    assert any("plan-quantum-5g" in v for v in violations)


def test_both_conflict_sides_must_be_rendered(docs, reg) -> None:
    losing_side = "plan-fiber-100:contract_months"
    thinned = tuple(d for d in docs if losing_side not in {p.fact_ref for p in d.placements})
    violations = verify.verify_corpus(thinned, reg)
    assert any("conflict-daniel-contract" in v for v in violations)


def test_a_placement_value_missing_from_text_is_caught(docs, reg) -> None:
    # A placement can agree with the registry (value == expected) while the template prose
    # never actually spells it out in the rendered text. That is the final fallback check in
    # _placement_violations, distinct from a drifted value; scrub the value out of the text
    # while leaving the placement's recorded value untouched to isolate it.
    victim = docs[0]
    placement = victim.placements[0]
    fact_ref, value = placement.fact_ref, placement.value
    assert value, "expected a non empty placement value to scrub"
    scrubbed = replace(victim, text=victim.text.replace(value, "REDACTED"))
    violations = verify.verify_corpus((scrubbed, *docs[1:]), reg)
    assert any(
        fact_ref in v and "not literally present in text" in v for v in violations
    )
