"""Registry integrity as a gate, over the loaded `Registry` itself.

These checks used to run as SQL against a throwaway SQLite database that `compile_registry`
materialized from the registry and `build_corpus` deleted with its temp dir. This file's own
`lookup_fact`/`queryable_with_plain_sql` tests were the ONLY callers of the query surface that
database existed to provide; `build.py` read `integrity_report` once and never touched it again.
The rules are unchanged and pinned identically below, one serialization round trip lighter.
"""

from __future__ import annotations

from corpus_tools import compile as reg_compile
from corpus_tools import registry
from .fixtures.corpus_expectations import CORE, GENERATED


def _committed() -> registry.Registry:
    return registry.load_registry([CORE, GENERATED])


def test_integrity_report_is_clean_for_the_committed_registry() -> None:
    assert reg_compile.integrity_report(_committed()) == ()


def test_integrity_report_reads_the_registry_directly_with_no_database() -> None:
    """The gate takes a `Registry`, not a path: nothing is written, nothing is serialized, and
    there is no compile step to keep in sync with the loader's own dataclasses."""
    import inspect

    (parameter,) = inspect.signature(reg_compile.integrity_report).parameters.values()
    assert parameter.annotation == "Registry"  # PEP 563: string annotations under __future__
    assert not hasattr(reg_compile, "compile_registry")
    assert not hasattr(reg_compile, "lookup_fact")


def test_the_registry_answers_the_cold_open_facts_without_a_lookup_layer() -> None:
    """`lookup_fact(db, "entity:field")` was a SQL wrapper over what the loaded registry already
    exposes as an attribute; this is the same two facts, read the direct way."""
    reg = _committed()
    assert reg.entity("contract_term-daniel-2025").fields["contract_months"] == 12
    assert reg.entity("plan-fiber-100").fields["contract_months"] == 0


def test_integrity_catches_a_planless_promotion() -> None:
    reg = registry.Registry(
        entities=(registry.Entity(id="promotion-x", kind="promotion", render=True, fields={"discount": "5"}),),
        edges=(),
        contradictions=(),
    )
    assert any("promotion-x" in violation for violation in reg_compile.integrity_report(reg))


def test_integrity_catches_a_rendered_plan_that_is_sold_nowhere() -> None:
    reg = registry.Registry(
        entities=(registry.Entity(id="plan-x", kind="plan", render=True, fields={"name": "X"}),),
        edges=(),
        contradictions=(),
    )
    assert any("plan-x" in violation and "available_in" in violation for violation in reg_compile.integrity_report(reg))


def test_an_unrendered_plan_sold_nowhere_is_not_a_violation() -> None:
    """The `render = 1` half of the original SQL: a plan that is never written into a document has
    no availability page to be inconsistent with."""
    reg = registry.Registry(
        entities=(registry.Entity(id="plan-hidden", kind="plan", render=False, fields={"name": "H"}),),
        edges=(),
        contradictions=(),
    )
    assert reg_compile.integrity_report(reg) == ()


def test_integrity_catches_a_contradiction_naming_a_fact_that_does_not_resolve() -> None:
    reg = registry.Registry(
        entities=(registry.Entity(id="plan-a", kind="plan", render=False, fields={"name": "A"}),),
        edges=(),
        contradictions=(
            registry.Contradiction(
                id="conflict-x", conflict_type="temporal", hops=1,
                winning_fact="plan-a:name", losing_fact="plan-ghost:name",
                resolution_rule="newer wins", question_hint="?",
            ),
        ),
    )
    report = reg_compile.integrity_report(reg)
    assert any("conflict-x" in v and "plan-ghost:name" in v for v in report)
    assert not any("plan-a:name" in v for v in report)  # the resolvable side must not be flagged


def test_integrity_catches_a_supersedes_edge_with_no_contradiction_record() -> None:
    reg = registry.Registry(
        entities=(
            registry.Entity(id="plan-new", kind="plan", render=False, fields={"name": "N"}),
            registry.Entity(id="plan-old", kind="plan", render=False, fields={"name": "O"}),
        ),
        edges=(registry.Edge(relation="supersedes", src="plan-new", dst="plan-old"),),
        contradictions=(),
    )
    report = reg_compile.integrity_report(reg)
    assert any("supersedes" in v and "plan-new" in v and "plan-old" in v for v in report)


def test_integrity_catches_a_mirrored_override_desync() -> None:
    # Pins the duplication conflict-promo-price-north introduced: overrides_fee.override_amount
    # must equal the src region's equipment_rental_override_amount field, or they can drift apart.
    reg = registry.Registry(
        entities=(
            registry.Entity(
                id="region-x",
                kind="region",
                render=True,
                fields={"name": "Region X", "equipment_rental_override_amount": "5.00"},
            ),
            registry.Entity(
                id="fee-equipment-rental",
                kind="fee",
                render=True,
                fields={"name": "Equipment Rental Fee", "amount": "10.00"},
            ),
        ),
        edges=(
            registry.Edge(
                relation="overrides_fee",
                src="region-x",
                dst="fee-equipment-rental",
                fields={"override_amount": "6.00", "reason": "desynced on purpose"},
            ),
        ),
        contradictions=(),
    )
    report = reg_compile.integrity_report(reg)
    assert any("region-x" in violation and "fee-equipment-rental" in violation for violation in report)
