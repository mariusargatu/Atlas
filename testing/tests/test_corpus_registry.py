"""Registry loading and validation: the root artifact must be internally consistent."""

from __future__ import annotations

from pathlib import Path

import pytest
from corpus_tools import registry
from .fixtures.corpus_expectations import CORE


@pytest.fixture(scope="module")
def reg() -> registry.Registry:
    return registry.load_registry([CORE])


def test_kinds_and_relations_are_the_hld_sets() -> None:
    assert registry.ENTITY_KINDS == ("plan", "region", "fee", "device", "contract_term", "promotion", "policy")
    assert registry.EDGE_RELATIONS == ("available_in", "applies_to", "overrides_fee", "compatible_with", "supersedes")


def test_core_registry_loads_with_narrative_entities(reg: registry.Registry) -> None:
    assert reg.entity("plan-fiber-500").fields["monthly_price"] == "39.99"
    assert reg.entity("plan-fiber-100-legacy").fields["contract_months"] == 12


def test_grounded_not_true_contradiction_is_typed(reg: registry.Registry) -> None:
    c = next(c for c in reg.contradictions if c.id == "conflict-daniel-contract")
    assert c.conflict_type == "temporal"
    assert c.winning_fact == "contract_term-daniel-2025:contract_months"
    assert c.losing_fact == "plan-fiber-100:contract_months"
    assert "supersedes" in c.resolution_rule


def test_never_rendered_pool_exists(reg: registry.Registry) -> None:
    hidden = [e for e in reg.entities if not e.render]
    assert {e.id for e in hidden} >= {"plan-quantum-5g", "fee-teleport-setup"}


def test_edges_reference_existing_entities(reg: registry.Registry) -> None:
    ids = {e.id for e in reg.entities}
    for edge in reg.edges:
        assert edge.src in ids and edge.dst in ids


def test_unknown_kind_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("entities:\n  - id: x-1\n    kind: satellite\n    fields: {}\n")
    with pytest.raises(registry.RegistryError, match="x-1"):
        registry.load_registry([bad])


def test_dangling_edge_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "entities:\n  - id: plan-a\n    kind: plan\n    fields: {}\n"
        "edges:\n  - relation: available_in\n    src: plan-a\n    dst: region-nowhere\n"
    )
    with pytest.raises(registry.RegistryError, match="region-nowhere"):
        registry.load_registry([bad])


def test_contradiction_facts_must_dereference(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "entities:\n  - id: plan-a\n    kind: plan\n    fields: {price: '10'}\n"
        "contradictions:\n  - id: c-1\n    conflict_type: temporal\n    hops: 1\n"
    )
    with pytest.raises(registry.RegistryError, match="c-1"):
        registry.load_registry([bad])
