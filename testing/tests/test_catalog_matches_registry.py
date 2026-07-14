"""One product universe: the account/catalog domain must tell the SAME story as the RAG corpus.

`backend/atlas/domain/catalog.py` (the account + oracle subsystem) and `corpus/registry/core.yaml`
(the RAG corpus subsystem) are two hand-authored sources that describe the same broadband provider.
They cannot single-source each other at runtime: the catalog is pure domain, the registry lives in
the harness, and the import lint forbids `backend -> harness`. So instead this test, which sits on
the harness side and may import BOTH, pins them together: the catalog's current/legacy plans must
match the registry's `plan-fiber-100` / `plan-fiber-100-legacy` facts, the legacy early-termination
fee must match `fee-early-termination`, and the cold-open customer (Daniel) must be the one the
registry's `contract_term-daniel-2025` names, on the legacy plan.

Before this test the two subsystems had drifted into different plan names entirely: the catalog sold
a `Fast (current)` / `Value (legacy)` universe that existed in no rendered document, while every
help page a customer could retrieve talked about `Fiber 100`. A reviewer reading the hermetic tests
saw one product; a customer retrieving a document saw another. This gate makes that drift a red
build, not a thing you notice months later.
"""
from __future__ import annotations

from decimal import Decimal

from atlas.domain import accounts, catalog
from corpus_tools.registry import load_registry

from .fixtures.corpus_expectations import CORE

_registry = load_registry([CORE])


def _plan(entity_id: str):
    return _registry.entity(entity_id).fields


def test_current_plan_matches_registry_fiber_100() -> None:
    current = catalog.CATALOG["plan_current_fast"]
    reg = _plan("plan-fiber-100")
    assert current.name == reg["name"] == "Fiber 100"
    assert current.monthly_price == Decimal(reg["monthly_price"])
    # registry contract_months == 0 <=> the catalog plan is term free
    assert current.has_term is (int(reg["contract_months"]) > 0) is False
    assert reg["page_status"] == "current"


def test_legacy_plan_matches_registry_fiber_100_legacy() -> None:
    legacy = catalog.CATALOG["plan_legacy_value"]
    reg = _plan("plan-fiber-100-legacy")
    assert legacy.name == reg["name"] == "Fiber 100 Legacy"
    assert legacy.monthly_price == Decimal(reg["monthly_price"])
    assert legacy.has_term is (int(reg["contract_months"]) > 0) is True
    assert reg["page_status"] == "superseded"
    # `check_eligibility` refuses a newly-taken plan whose name contains "legacy"; the registry's
    # own name carries it, so the two agree on which plan is discontinued.
    assert "legacy" in legacy.name.lower()
    assert catalog.check_eligibility("plan_legacy_value") is False
    assert catalog.check_eligibility("plan_current_fast") is True


def test_legacy_early_termination_fee_matches_the_registry_fee_entity() -> None:
    legacy = catalog.CATALOG["plan_legacy_value"]
    fee = _plan("fee-early-termination")
    assert legacy.early_termination_fee == Decimal(fee["amount"]) == Decimal("150.00")


def test_the_cold_open_customer_is_the_one_the_registry_names() -> None:
    """Daniel is the cold open: the account store puts him on the legacy plan, and the registry's
    own `contract_term-daniel-2025` puts the same Daniel on the same legacy plan with a 12 month
    term. If the two ever named different customers, the hermetic cold open and the RAG corpus would
    be telling different stories about who gets the wrong answer."""
    daniel = accounts.get_account("cust_legacy_term")
    contract = _plan("contract_term-daniel-2025")
    assert daniel.name == contract["customer_name"] == "Daniel"
    assert daniel.plan_id == "plan_legacy_value"                    # the catalog's legacy plan
    assert contract["plan_id"] == "plan-fiber-100-legacy"          # the registry's legacy plan
    assert int(contract["contract_months"]) == 12
    # and the current happy-path customer is Sarah on the current plan, both sides
    sarah = accounts.get_account("cust_current")
    assert sarah.name == "Sarah" and sarah.plan_id == "plan_current_fast"


def test_daniels_bill_equals_his_plan_price_so_the_account_data_is_internally_honest() -> None:
    """A guard against re-pricing the catalog and forgetting the seed: Daniel's seeded bill must be
    what his plan actually costs, or the account store contradicts the catalog it draws from."""
    daniel = accounts.get_account("cust_legacy_term")
    assert daniel.bill.amount == catalog.compute_price(daniel.plan_id) == Decimal("24.99")
    sarah = accounts.get_account("cust_current")
    assert sarah.bill.amount == catalog.compute_price(sarah.plan_id) == Decimal("29.99")
