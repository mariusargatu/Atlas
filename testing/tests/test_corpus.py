"""The current-plan document is generated from the catalog, not hand typed, so it can never
describe a plan differently than the catalog itself does.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from atlas.domain.catalog import Plan
from atlas.domain.corpus import CORPUS, CORPUS_FACTS, render_current_plan_chunk


def test_render_current_plan_chunk_states_no_fee_when_catalog_has_none():
    plan = Plan(
        id="plan_test", name="Test Plan", monthly_price=Decimal("10.00"),
        has_term=False, early_termination_fee=Decimal("0.00"), data_cap_gb=None,
    )
    text = render_current_plan_chunk(plan)
    assert "no early-termination fee" in text
    assert "unlimited" in text.lower()


def test_render_current_plan_chunk_states_the_real_fee_when_the_catalog_has_one():
    plan = Plan(
        id="plan_test", name="Test Plan", monthly_price=Decimal("10.00"),
        has_term=False, early_termination_fee=Decimal("15.00"), data_cap_gb=None,
    )
    text = render_current_plan_chunk(plan)
    assert "15.00" in text


def test_render_current_plan_chunk_states_the_real_data_cap():
    plan = Plan(
        id="plan_test", name="Test Plan", monthly_price=Decimal("10.00"),
        has_term=False, early_termination_fee=Decimal("0.00"), data_cap_gb=250,
    )
    text = render_current_plan_chunk(plan)
    assert "250" in text


def test_render_current_plan_chunk_refuses_a_plan_with_a_term():
    plan = Plan(
        id="plan_test", name="Test Plan", monthly_price=Decimal("10.00"),
        has_term=True, early_termination_fee=Decimal("50.00"), data_cap_gb=None,
    )
    with pytest.raises(ValueError, match="has a term"):
        render_current_plan_chunk(plan)


def test_the_real_corpus_chunk_is_generated_from_the_real_catalog():
    chunk = next(c for c in CORPUS if c.doc_id == "plan-current-page")
    assert "no early-termination fee" in chunk.text
    assert "unlimited" in chunk.text.lower()


def test_corpus_facts_are_derived_not_separately_hand_kept():
    assert CORPUS_FACTS["plan-current-page"]["has_contract"] is False
    assert CORPUS_FACTS["plan-current-page"]["has_data_cap"] is False
