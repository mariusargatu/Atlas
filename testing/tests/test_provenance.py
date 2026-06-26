"""Both golden record types normalise onto one provenance vocabulary, without either changing shape."""
from __future__ import annotations

import pytest

from evals.evalkit.provenance import Provenance, provenance_of


def _dataset_case(**overrides) -> dict:
    base = {"case_id": "gen-1", "origin": "synthetic", "intent": "troubleshooting", "turns": [{"user": "hi"}]}
    base.update(overrides)
    return base


def _golden_case():
    from evals.datasets.seed import GOLDEN

    return GOLDEN[0]


def test_dataset_case_normalises():
    p = provenance_of(_dataset_case())
    assert p == Provenance(id="gen-1", origin="synthetic", source="generated", tier="gold", risk="troubleshooting")


def test_promoted_dataset_case_is_silver():
    assert provenance_of(_dataset_case(origin="promoted")).tier == "silver"


def test_authored_dataset_case_is_gold():
    assert provenance_of(_dataset_case(origin="authored")).tier == "gold"


def test_golden_case_normalises_onto_the_same_vocabulary():
    p = provenance_of(_golden_case())
    assert p.origin in ("synthetic", "authored", "promoted")
    assert p.tier in ("gold", "silver")
    assert p.id == _golden_case().id


def test_unknown_origin_raises():
    with pytest.raises(ValueError, match="unknown origin"):
        provenance_of(_dataset_case(origin="made-up"))
