"""Generate near duplicate regional plan variants: the corpus's distractor mechanism (research 06)."""

from __future__ import annotations

import random

import yaml

from corpus_tools.registry import Registry

DEFAULT_SEED = 20260718
_PRICE_DELTAS = ("-3.00", "-1.00", "2.00", "4.00")


def expand_variants(reg: Registry, seed: int) -> str:
    # Regenerating after ANY core.yaml change reshuffles all variants by design (one sequential
    # rng stream consumed in entity/region order): the pinned regeneration equality test
    # (test_committed_variants_match_regeneration) is the staleness gate, not a change detector
    # to work around.
    rng = random.Random(seed)
    entities: list[dict] = []
    edges: list[dict] = []
    regions = sorted(e.id for e in reg.by_kind("region"))
    for plan in sorted(reg.by_kind("plan"), key=lambda e: e.id):
        if not plan.render or plan.fields.get("variant_of"):
            continue
        # Decide the kept regions first, in region sort order, exactly as before. Then draw
        # deltas without replacement across the whole family in one call: sampling (not
        # choosing independently per region) guarantees distinct prices within every family.
        # At most 3 regions ever compete for 4 deltas, so len(kept_regions) <= len(_PRICE_DELTAS)
        # always holds and rng.sample never raises. Skip probability lowered from 0.34 to 0.15
        # (keep more regions) after switching to sampling without replacement shifted the rng
        # stream and dropped the committed corpus's total variant count below the document mix
        # target; this keeps enough near duplicate regional variants for the downstream renderer.
        kept_regions = [region_id for region_id in regions if rng.random() >= 0.15]
        deltas = rng.sample(_PRICE_DELTAS, len(kept_regions))
        for region_id, delta in zip(kept_regions, deltas):
            price = f"{float(plan.fields['monthly_price']) + float(delta):.2f}"
            fields = {**plan.fields, "monthly_price": price, "variant_of": plan.id, "region": region_id}
            variant_id = f"{plan.id}--{region_id}"
            entities.append({"id": variant_id, "kind": "plan", "render": True, "fields": fields})
            edges.append({"relation": "available_in", "src": variant_id, "dst": region_id})
    return yaml.safe_dump({"entities": entities, "edges": edges}, sort_keys=True, width=100)
