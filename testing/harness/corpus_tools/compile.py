"""Registry integrity checks (HLD D4): the gate `corpus_tools.build` runs BEFORE rendering, so a
fact authoring error is named precisely instead of surfacing as a renderer crash.

Pure functions over the already loaded, frozen `Registry` dataclasses. This module used to
materialize the whole registry into a throwaway SQLite database first (a `_SCHEMA` of four tables,
four `executemany` inserts, and a `lookup_fact` helper with no caller anywhere) purely so these four
checks could be written as SQL. `build_corpus` created that database inside a `TemporaryDirectory`,
read it exactly once, and let it die with the temp dir; nothing hashed it into the manifest and
nothing ever queried it again. The checks below are the same four rules, expressed against the same
data the loader has already parsed, with no serialization round trip in between.
"""

from __future__ import annotations

from corpus_tools.registry import Registry


def _fact_resolves(reg: Registry, fact_ref: str) -> bool:
    """A `entity_id:field` reference resolves when that entity exists and carries that field."""
    entity_id, _, fact_field = fact_ref.partition(":")
    try:
        entity = reg.entity(entity_id)
    except KeyError:
        return False
    return fact_field in entity.fields


def _entities_missing_an_edge(reg: Registry, kind: str, relation: str, *, rendered_only: bool) -> tuple[str, ...]:
    """Ids of every `kind` entity that is never the source of a `relation` edge, sorted. The SQL
    this replaced applied `render = 1` to plans and not to promotions; `rendered_only` keeps that
    distinction explicit rather than implied by two nearly identical queries."""
    sources = {edge.src for edge in reg.edges if edge.relation == relation}
    return tuple(
        sorted(
            e.id
            for e in reg.by_kind(kind)
            if (e.render or not rendered_only) and e.id not in sources
        )
    )


def integrity_report(reg: Registry) -> tuple[str, ...]:
    """Every registry integrity violation, in a stable order. Empty means the registry is coherent.

    Four rules, unchanged from the SQL they replaced:
      1. a rendered plan must be sold somewhere (`available_in`)
      2. a promotion must apply to something (`applies_to`)
      3. both facts a contradiction names must actually resolve
      4. a `supersedes` edge must have a contradiction record covering it

    Plus the mirrored override check (added beyond the brief's literal schema): an `overrides_fee`
    edge may carry an `override_amount` duplicating the src region's own
    `equipment_rental_override_amount` field, and if both exist they must agree, or the two copies
    have silently drifted apart.
    """
    violations: list[str] = []

    violations += [
        f"plan {pid} has no available_in edge"
        for pid in _entities_missing_an_edge(reg, "plan", "available_in", rendered_only=True)
    ]
    violations += [
        f"promotion {pid} has no applies_to edge"
        for pid in _entities_missing_an_edge(reg, "promotion", "applies_to", rendered_only=False)
    ]

    # A regional variant must be sold in the region it is a variant OF. `available_in` existing at
    # all (rule 1) was the only availability check before, so a generated variant pairing a plan
    # with a region its base plan is not sold in passed cleanly: the committed corpus carried two.
    for variant in sorted(reg.by_kind("plan"), key=lambda e: e.id):
        base_id = variant.fields.get("variant_of")
        region = variant.fields.get("region")
        if not base_id or not region:
            continue
        base_regions = {e.dst for e in reg.edges if e.relation == "available_in" and e.src == base_id}
        if region not in base_regions:
            violations.append(
                f"plan variant {variant.id} sells {base_id} in {region}, which is not among its "
                f"base plan's available_in regions ({sorted(base_regions)})"
            )

    for contradiction in sorted(reg.contradictions, key=lambda c: c.id):
        for ref in (contradiction.winning_fact, contradiction.losing_fact):
            if not _fact_resolves(reg, ref):
                violations.append(f"contradiction {contradiction.id}: fact {ref} does not resolve")

    covered = {
        (c.winning_fact.partition(":")[0], c.losing_fact.partition(":")[0]) for c in reg.contradictions
    }
    for edge in sorted(
        (e for e in reg.edges if e.relation == "supersedes"), key=lambda e: (e.src, e.dst)
    ):
        if (edge.src, edge.dst) not in covered:
            violations.append(f"supersedes edge {edge.src} -> {edge.dst} has no contradiction record")

    for edge in sorted(
        (e for e in reg.edges if e.relation == "overrides_fee"), key=lambda e: (e.src, e.dst)
    ):
        if "override_amount" not in edge.fields:
            continue
        override_amount = str(edge.fields["override_amount"])
        try:
            mirror = reg.entity(edge.src).fields.get("equipment_rental_override_amount")
        except KeyError:
            mirror = None
        if mirror is not None and str(mirror) != override_amount:
            violations.append(
                f"overrides_fee edge {edge.src} -> {edge.dst}: override_amount {override_amount!r} "
                f"!= {edge.src}:equipment_rental_override_amount {str(mirror)!r}"
            )

    return tuple(violations)
