"""Generate near duplicate regional plan variants: the corpus's distractor mechanism (research 06)."""

from __future__ import annotations

import random

import yaml

from corpus_tools.registry import Registry

DEFAULT_SEED = 20260718
_PRICE_DELTAS = ("-3.00", "-1.00", "2.00", "4.00")


def _regions_selling(reg: Registry, plan_id: str) -> tuple[str, ...]:
    """The regions the registry AUTHORS this plan as available in, sorted.

    Derived from the plan's own `available_in` edges, never from the full region list. The version
    this replaced sampled over every region in the registry with `rng.random() >= 0.15`, ignoring
    availability entirely, and so invented plan/region pairs the registry says do not exist: the
    committed corpus shipped `plan-fiber-100-legacy--region-coast` and `plan-starter-50--region-north`
    as rendered plan pages, both selling a plan in a region it is not sold in.
    `compile.integrity_report` only ever checked that SOME `available_in` edge existed, so neither
    was caught.

    This also removes the hand tuned skip probability (and the "lowered from 0.34 to 0.15 to keep the
    variant count up" note that went with it): which variants exist is now a fact the registry
    authors, not a coin flip whose rate had to be retuned whenever the rng stream shifted.
    """
    return tuple(sorted(e.dst for e in reg.edges if e.relation == "available_in" and e.src == plan_id))


def expand_variants(reg: Registry, seed: int) -> str:
    # Regenerating after ANY core.yaml change reshuffles the price deltas by design (one sequential
    # rng stream consumed in entity/region order): the pinned regeneration equality test
    # (test_committed_variants_match_regeneration) is the staleness gate, not a change detector
    # to work around. Which variants EXIST is no longer part of that stream; only their prices are.
    rng = random.Random(seed)
    entities: list[dict] = []
    edges: list[dict] = []
    for plan in sorted(reg.by_kind("plan"), key=lambda e: e.id):
        if not plan.render or plan.fields.get("variant_of"):
            continue
        # Deltas are drawn without replacement across the whole family in one call, so prices are
        # guaranteed distinct within every family. At most 3 regions are ever authored for one plan
        # and there are 4 deltas, so `len(kept) <= len(_PRICE_DELTAS)` holds and `rng.sample` never
        # raises; assert it rather than leave it as a comment, since a fourth region added to one
        # plan in the registry would otherwise fail here with a bare ValueError from `sample`.
        kept = _regions_selling(reg, plan.id)
        if len(kept) > len(_PRICE_DELTAS):
            raise ValueError(
                f"plan {plan.id} is available in {len(kept)} regions but only {len(_PRICE_DELTAS)} "
                f"distinct price deltas exist; add another delta to _PRICE_DELTAS"
            )
        deltas = rng.sample(_PRICE_DELTAS, len(kept))
        for region_id, delta in zip(kept, deltas):
            price = f"{float(plan.fields['monthly_price']) + float(delta):.2f}"
            fields = {**plan.fields, "monthly_price": price, "variant_of": plan.id, "region": region_id}
            variant_id = f"{plan.id}--{region_id}"
            entities.append({"id": variant_id, "kind": "plan", "render": True, "fields": fields})
            edges.append({"relation": "available_in", "src": variant_id, "dst": region_id})
    return yaml.safe_dump({"entities": entities, "edges": edges}, sort_keys=True, width=100)
