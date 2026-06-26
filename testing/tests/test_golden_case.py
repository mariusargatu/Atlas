"""The typed golden record: validation, enrichment, and the two projections.

``GoldenCase`` is the REQUIRED typed format: invalid construction must raise, the committed seed must
be valid on import, ``enrich`` must harden a loose draft, and the case must project cleanly to the
runner (``to_eval_case``).
"""
from __future__ import annotations

import pytest

from evals.datasets.seed import GOLDEN
from evals.evalkit.golden_case import GoldenCase, GoldenDraft, enrich

_VALID = dict(
    id="x", turns=("hi",), customer_id="cust_current", expected="ok",
    category="action", risk="r", consequence="high", oracle="o",
    source="hand_written", author_role="engineer", tier="gold",
)


def test_valid_case_constructs():
    case = GoldenCase(**_VALID)
    assert case.id == "x" and case.tier == "gold"


@pytest.mark.parametrize("field", ["id", "customer_id", "expected", "risk", "oracle"])
def test_required_string_empty_raises(field):
    with pytest.raises(ValueError, match=f"{field} is required"):
        GoldenCase(**{**_VALID, field: "  "})


def test_empty_turns_raises():
    with pytest.raises(ValueError, match="at least one turn"):
        GoldenCase(**{**_VALID, "turns": ()})


def test_unknown_customer_raises():
    with pytest.raises(ValueError, match="not a seeded account"):
        GoldenCase(**{**_VALID, "customer_id": "cust_nope"})


@pytest.mark.parametrize("field", ["category", "consequence", "source", "author_role", "tier"])
def test_out_of_range_enum_raises(field):
    with pytest.raises(ValueError, match=f"{field}="):
        GoldenCase(**{**_VALID, field: "bogus"})


def test_enrich_hardens_a_draft():
    draft = GoldenDraft(id="d", turns=("hi",), customer_id="cust_current", expected="ok", notes="n")
    case = enrich(
        draft, category="policy_question", risk="r", consequence="low", oracle="o",
        source="hand_written", author_role="sme", tier="gold",
    )
    assert isinstance(case, GoldenCase)
    assert case.notes == "n"                       # draft notes carried through
    assert case.expected == "ok"


def test_enrich_missing_required_meta_raises():
    draft = GoldenDraft(id="d", turns=("hi",), customer_id="cust_current", expected="ok")
    with pytest.raises(TypeError):                  # missing required enrichment kwargs
        enrich(draft, category="action")


def test_to_eval_case_projects_runner_fields():
    ec = GoldenCase(**_VALID).to_eval_case()
    assert ec.id == "x" and ec.customer_id == "cust_current"
    assert ec.risk == "r" and ec.expected == "ok"


# ---- the committed seed itself ----

def test_seed_is_ten_valid_gold_cases():
    assert len(GOLDEN) == 10
    assert all(isinstance(c, GoldenCase) for c in GOLDEN)       # validated on import
    assert all(c.tier == "gold" for c in GOLDEN)
    assert len({c.id for c in GOLDEN}) == 10                    # ids unique


def test_seed_stratified_toward_consequence():
    high = [c for c in GOLDEN if c.consequence == "high"]
    assert len(high) >= len(GOLDEN) / 2                         # weight on the costly surface


def test_seed_covers_all_intent_families():
    assert {c.category for c in GOLDEN} == {"policy_question", "account_read", "action"}


def test_seed_carries_the_cold_open_trap():
    trap = next(c for c in GOLDEN if c.id == "cap-legacy-trap")
    assert trap.customer_id == "cust_legacy_term"
    assert trap.consequence == "high"
