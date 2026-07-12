"""The renderer: template first, seeded variation, provenance for every placed fact."""

from __future__ import annotations

import json

import pytest
from corpus_tools import expand, registry, render
from .fixtures.corpus_expectations import CORE, GENERATED, TEMPLATES
from .fixtures.corpus_expectations import COMMITTED_CORPUS_DIR


@pytest.fixture(scope="module")
def docs() -> tuple[render.RenderedDoc, ...]:
    reg = registry.load_registry([CORE, GENERATED])
    return render.render_corpus(reg, TEMPLATES, seed=7)


def test_rendering_is_deterministic() -> None:
    reg = registry.load_registry([CORE, GENERATED])
    first = render.render_corpus(reg, TEMPLATES, seed=7)
    second = render.render_corpus(reg, TEMPLATES, seed=7)
    assert first == second


def test_never_rendered_entities_stay_hidden(docs: tuple[render.RenderedDoc, ...]) -> None:
    joined = "\n".join(d.text for d in docs)
    assert "quantum" not in joined.lower()
    assert "teleport" not in joined.lower()


def test_every_document_has_placements(docs: tuple[render.RenderedDoc, ...]) -> None:
    for doc in docs:
        assert doc.placements, f"{doc.doc_id} rendered with no fact placements"


def test_placements_are_true_against_the_registry(docs: tuple[render.RenderedDoc, ...]) -> None:
    reg = registry.load_registry([CORE, GENERATED])
    for doc in docs:
        for placement in doc.placements:
            entity_id, _, fact_field = placement.fact_ref.partition(":")
            assert str(reg.entity(entity_id).fields[fact_field]) == placement.value


def test_plan_family_docs_are_near_duplicates(docs: tuple[render.RenderedDoc, ...]) -> None:
    family = [d for d in docs if d.doc_id.startswith("doc-plan_page-plan-fiber-500")]
    assert len(family) >= 3
    prices = {
        next(
            (p.value for p in d.placements if p.fact_ref == f"{d.doc_id.removeprefix('doc-plan_page-')}:monthly_price"),
            None,
        )
        for d in family
    }
    assert len(prices) == len(family)


def test_doc_type_mix_is_roughly_the_target(docs: tuple[render.RenderedDoc, ...]) -> None:
    total = len(docs)
    plan_pages = sum(1 for d in docs if d.doc_type == "plan_page")
    assert total >= 40
    assert 0.2 <= plan_pages / total <= 0.5


def test_all_rendered_docs_are_pairwise_distinct(docs: tuple[render.RenderedDoc, ...]) -> None:
    by_text: dict[str, list[str]] = {}
    for doc in docs:
        by_text.setdefault(doc.text, []).append(doc.doc_id)
    collisions = [ids for ids in by_text.values() if len(ids) > 1]
    assert not collisions, f"byte identical docs under different doc_ids: {collisions}"


def test_promo_pages_place_waiver_claims(docs: tuple[render.RenderedDoc, ...]) -> None:
    promo_doc = next(d for d in docs if d.doc_id == "doc-promo_page-promotion-fiber500-launch-north")
    assert any(
        p.fact_ref == "promotion-fiber500-launch-north:equipment_rental_waived" and p.value == "true"
        for p in promo_doc.placements
    )


def test_fee_schedules_place_override_amounts(docs: tuple[render.RenderedDoc, ...]) -> None:
    fee_doc = next(d for d in docs if d.doc_id == "doc-fee_schedule-region-north")
    assert any(
        p.fact_ref == "region-north:equipment_rental_override_amount" and p.value == "5.00" for p in fee_doc.placements
    )


def test_promotion_without_applies_to_edge_raises_registry_error() -> None:
    # A promotion with no applies_to edge is a registry authoring error (the compiler's
    # integrity_report also flags it), not a renderer crash: _promo_context's next() over the
    # applies_to edges must not surface as a bare StopIteration.
    reg = registry.Registry(
        entities=(
            registry.Entity(
                id="promotion-orphan", kind="promotion", render=True, fields={"name": "Orphan Promo"}
            ),
        ),
        edges=(),
        contradictions=(),
    )
    with pytest.raises(registry.RegistryError, match="promotion-orphan"):
        render.render_corpus(reg, TEMPLATES, seed=7)


def test_template_variants_are_discovered_from_the_template_directory() -> None:
    """`_VARIANTS` used to be a hand written doc_type -> variants table that had to be edited in
    lockstep with `corpus/templates/`. A variant added to the directory but not the table was never
    picked; one added to the table but not the directory crashed the render with a KeyError. The
    directory listing IS the mapping, since `<doc_type>_<variant>.txt` is already the naming
    contract every template follows."""
    discovered = render._discover_variants(TEMPLATES)
    on_disk = sorted(p.stem for p in TEMPLATES.glob("*.txt"))
    flattened = sorted(f"{doc_type}_{v}" for doc_type, vs in discovered.items() for v in vs)
    assert flattened == on_disk
    assert not hasattr(render, "_VARIANTS")


def test_variant_selection_is_stable_for_a_seed_and_template_set() -> None:
    """Determinism: the candidate list is sorted before `rng.choice` sees it, so the same seed over
    the same templates always picks the same variant, whatever order the filesystem lists them in."""
    reg = registry.load_registry([CORE, GENERATED])
    first = render.render_corpus(reg, TEMPLATES, seed=expand.DEFAULT_SEED)
    second = render.render_corpus(reg, TEMPLATES, seed=expand.DEFAULT_SEED)
    assert [d.text for d in first] == [d.text for d in second]


def test_rendering_the_committed_registry_reproduces_the_committed_corpus_exactly() -> None:
    """The staleness gate for `corpus/rendered/`: re-rendering the committed registry at the seed
    the build used must reproduce every committed document byte for byte, and every provenance span
    exactly. Without this, a renderer change that alters output silently leaves the committed corpus
    (and the index built from it) describing a render nobody can reproduce."""
    docs = render.render_corpus(registry.load_registry([CORE, GENERATED]), TEMPLATES, seed=expand.DEFAULT_SEED)
    committed = COMMITTED_CORPUS_DIR
    assert {d.doc_id for d in docs} == {p.stem for p in (committed / "docs").glob("*.txt")}
    for doc in docs:
        assert (committed / "docs" / f"{doc.doc_id}.txt").read_text() == doc.text
        sidecar = json.loads((committed / "provenance" / f"{doc.doc_id}.json").read_text())
        # `clause` is added by build.py after rendering, so compare only what the renderer produces
        assert [(p["fact_ref"], p["value"], tuple(p["span"])) for p in sidecar["placements"]] == [
            (p.fact_ref, p.value, p.span) for p in doc.placements
        ]
