"""The variant expander: deterministic near duplicate distractor families."""

from __future__ import annotations

from corpus_tools import expand, registry
from .fixtures.corpus_expectations import CORE, GENERATED


def test_expansion_is_deterministic() -> None:
    reg = registry.load_registry([CORE])
    assert expand.expand_variants(reg, seed=1) == expand.expand_variants(reg, seed=1)
    assert expand.expand_variants(reg, seed=1) != expand.expand_variants(reg, seed=2)


def test_committed_variants_match_regeneration() -> None:
    reg = registry.load_registry([CORE])
    assert GENERATED.read_text() == expand.expand_variants(reg, seed=expand.DEFAULT_SEED)


def test_variants_form_near_duplicate_families() -> None:
    combined = registry.load_registry([CORE, GENERATED])
    fiber_family = [e for e in combined.entities if e.fields.get("variant_of") == "plan-fiber-500"]
    assert len(fiber_family) >= 2
    for variant in fiber_family:
        base = combined.entity("plan-fiber-500")
        assert variant.fields["download_mbps"] == base.fields["download_mbps"]
        assert variant.fields["monthly_price"] != base.fields["monthly_price"]


def test_combined_registry_still_validates() -> None:
    combined = registry.load_registry([CORE, GENERATED])
    assert len(combined.entities) > len(registry.load_registry([CORE]).entities)


def test_never_rendered_entities_get_no_variants() -> None:
    combined = registry.load_registry([CORE, GENERATED])
    assert not [e for e in combined.entities if e.fields.get("variant_of") == "plan-quantum-5g"]


def test_variant_prices_are_unique_within_a_family() -> None:
    combined = registry.load_registry([CORE, GENERATED])
    families: dict[str, list] = {}
    for plan in combined.by_kind("plan"):
        base_id = plan.fields.get("variant_of")
        if plan.render and base_id:
            families.setdefault(base_id, []).append(plan)
    assert families, "expected at least one variant family in the committed registry"
    for base_id, variants in families.items():
        if len(variants) < 2:
            continue
        base = combined.entity(base_id)
        prices = [base.fields["monthly_price"], *(v.fields["monthly_price"] for v in variants)]
        assert len(set(prices)) == len(prices), f"{base_id} family has duplicate monthly_price values: {prices}"
