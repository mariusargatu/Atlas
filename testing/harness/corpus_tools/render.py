"""Template first renderer: seeded variation, per document rng, provenance for every placed fact.

Design note (registry field mismatch, per the task brief's own instruction to check actual field
names and handle absence explicitly): device entities carry `name`, `device_type`, and
`firmware_version` but no `model` or `ports` field, so `device_manual` renders `name` where a
`model` label would otherwise go and omits ports entirely. Policy entities split into two
disjoint field sets (`credit_per_hour`/`max_monthly_credit` for outage credits vs
`threshold_gb`/`throttle_mbps` for fair use); `_policy_context` branches on which set is present
rather than assuming a shared `usage_cap` field that does not exist in the registry.

Design note (document volume): troubleshooting docs are rendered per device per plan in the full
family (the compatible base plan plus every one of its regional variants), not one per device,
because a device's compatible base plan and its near duplicate regional variants share the same
hardware compatibility. This is the same distractor mechanism the plan variants use, extended to
support docs, and it is what gets the corpus to a plausible document type mix.

Design note (conflict evidence coverage): promo_page and fee_schedule exist so that both sides of
conflict-promo-price-north are rendered somewhere: the promotion's own equipment_rental_waived
claim (the losing fact) needs a document to live in just as much as the region's overrides_fee
chain (the winning fact) does. Both `_promo_context` and `_fee_schedule_context` place every field
the source entity carries via `_fields_dump` (sorted, deterministic) rather than hand enumerating
promotion/region-specific field names, because promotions and regions have different field sets
per instance (only the north promo has equipment_rental_waived; only region-north has
equipment_rental_override_amount) and a generic dump can't silently miss a field a future entity
adds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from corpus_tools.registry import Entity, Registry, RegistryError

# doc_type -> template variant suffixes available; the renderer picks one per document via the
# per document rng, so template choice does not depend on entity iteration order.
_VARIANTS: dict[str, tuple[str, ...]] = {
    "plan_page": ("a", "b"),
    "contract_terms": ("a",),
    "troubleshooting": ("a", "b"),
    "device_manual": ("a",),
    "policy": ("a",),
    "promo_page": ("a",),
    "fee_schedule": ("a",),
}


@dataclass(frozen=True)
class Placement:
    # value is always the raw, registry-consistent fact value (drift detection in verify.py
    # compares it directly against the registry field). span is the char range in the doc's
    # text that EXPRESSES the fact: for literal fills, the value's own occurrence; for prose
    # branches (contract_months=0 rendering as "No contract. Cancel any time.", where the digit
    # "0" never appears as a token), the whole clause. When span's slice does not literally
    # contain value, that is exactly the signal that a placement is a prose branch: build.py's
    # sidecar writer and verify.py both key off that same "value in slice" test rather than a
    # separate flag, so there is only one place (render.py, computing the span itself) that
    # needs to know which branch a fact took.
    fact_ref: str
    value: str
    span: tuple[int, int]


@dataclass(frozen=True)
class RenderedDoc:
    doc_id: str
    doc_type: str
    text: str
    placements: tuple[Placement, ...]


def _load_templates(templates_dir: Path) -> dict[str, str]:
    return {
        f"{doc_type}_{variant}": (templates_dir / f"{doc_type}_{variant}.txt").read_text()
        for doc_type, variants in _VARIANTS.items()
        for variant in variants
    }


def _contract_clause(entity_id: str, fields: dict) -> tuple[str, list[tuple[str, str, str]]]:
    # contract_months is an int in the registry YAML; the comparison MUST cast to str, or every
    # zero contract plan takes the wrong branch and the cold open (plan-fiber-100's current page
    # vs contract_term-daniel-2025, which supersedes it) silently breaks. The placement is
    # recorded in both branches: "No contract" is itself a claim about contract_months being 0.
    #
    # The third element of each placement triple is the anchor: the exact substring _render_doc
    # should locate via str.find to compute the placement's span. For the nonzero branch the
    # anchor is the raw value itself ("12" is a token inside "a 12 month contract."). For the
    # zero branch the anchor is the WHOLE clause, not "0": the digit never appears as its own
    # token anywhere in "No contract. Cancel any time.", so anchoring on the value would either
    # fail to find a span at all or, worse, silently match an unrelated "0" elsewhere in the
    # document (the digest's "0 inside 100" case). The renderer is the only place that knows
    # "contract_months=0" maps to this clause, so it is the only place that should decide the
    # anchor.
    contract_months = fields["contract_months"]
    value = str(contract_months)
    if value != "0":
        clause = f"This plan runs on a {contract_months} month contract."
        return clause, [(f"{entity_id}:contract_months", value, value)]
    clause = "No contract. Cancel any time."
    return clause, [(f"{entity_id}:contract_months", value, clause)]


def _region_clause(entity_id: str, fields: dict) -> tuple[str, list[tuple[str, str, str]]]:
    region = fields.get("region")
    if region is None:
        return "", []
    value = str(region)
    return f" in the {region} region", [(f"{entity_id}:region", value, value)]


def _plan_context(plan: Entity) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    fields = plan.fields
    contract_text, contract_placements = _contract_clause(plan.id, fields)
    region_text, region_placements = _region_clause(plan.id, fields)
    name = fields["name"]
    download_mbps = str(fields["download_mbps"])
    upload_mbps = str(fields["upload_mbps"])
    monthly_price = str(fields["monthly_price"])
    placements = [
        (f"{plan.id}:name", name, name),
        (f"{plan.id}:download_mbps", download_mbps, download_mbps),
        (f"{plan.id}:upload_mbps", upload_mbps, upload_mbps),
        (f"{plan.id}:monthly_price", monthly_price, monthly_price),
        *contract_placements,
        *region_placements,
    ]
    context = {
        "plan_name": fields["name"],
        "download_mbps": str(fields["download_mbps"]),
        "upload_mbps": str(fields["upload_mbps"]),
        "monthly_price": str(fields["monthly_price"]),
        "region_clause": region_text,
        "contract_clause": contract_text,
    }
    return context, placements


def _contract_term_context(term: Entity, reg: Registry) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    fields = term.fields
    fee = reg.entity("fee-early-termination")
    name = fields["name"]
    customer_name = fields["customer_name"]
    contract_months = str(fields["contract_months"])
    vintage_year = fields["vintage_year"]
    fee_amount = fee.fields["amount"]
    placements = [
        (f"{term.id}:name", name, name),
        (f"{term.id}:customer_name", customer_name, customer_name),
        (f"{term.id}:contract_months", contract_months, contract_months),
        (f"{term.id}:vintage_year", vintage_year, vintage_year),
        (f"{fee.id}:amount", fee_amount, fee_amount),
    ]
    context = {
        "name": fields["name"],
        "customer_name": fields["customer_name"],
        "contract_months": str(fields["contract_months"]),
        "vintage_year": fields["vintage_year"],
        "termination_fee_amount": fee.fields["amount"],
    }
    return context, placements


def _device_manual_context(device: Entity) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    fields = device.fields
    name = fields["name"]
    device_type = fields["device_type"]
    firmware_version = fields["firmware_version"]
    placements = [
        (f"{device.id}:name", name, name),
        (f"{device.id}:device_type", device_type, device_type),
        (f"{device.id}:firmware_version", firmware_version, firmware_version),
    ]
    context = {
        "name": fields["name"],
        "device_type": fields["device_type"],
        "firmware_version": fields["firmware_version"],
    }
    return context, placements


def _troubleshooting_context(device: Entity, plan: Entity) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    # Plan family members share the base plan's name field, so name/firmware alone made siblings
    # byte identical whenever the same a/b template got picked. monthly_price (and region, when
    # present) are the facts that actually differ per family member, so they carry the variation.
    fields = device.fields
    region_text, region_placements = _region_clause(plan.id, plan.fields)
    name = fields["name"]
    firmware_version = fields["firmware_version"]
    plan_name = plan.fields["name"]
    monthly_price = str(plan.fields["monthly_price"])
    placements = [
        (f"{device.id}:name", name, name),
        (f"{device.id}:firmware_version", firmware_version, firmware_version),
        (f"{plan.id}:name", plan_name, plan_name),
        (f"{plan.id}:monthly_price", monthly_price, monthly_price),
        *region_placements,
    ]
    context = {
        "name": fields["name"],
        "firmware_version": fields["firmware_version"],
        "plan_name": plan.fields["name"],
        "monthly_price": str(plan.fields["monthly_price"]),
        "region_clause": region_text,
    }
    return context, placements


def _policy_context(policy: Entity) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    fields = policy.fields
    name = fields["name"]
    placements = [(f"{policy.id}:name", name, name)]
    if "credit_per_hour" in fields:
        credit_per_hour = str(fields["credit_per_hour"])
        max_monthly_credit = str(fields["max_monthly_credit"])
        detail = (
            f"Outage credits accrue at {credit_per_hour} per hour of qualifying "
            f"outage, up to {max_monthly_credit} per month."
        )
        placements += [
            (f"{policy.id}:credit_per_hour", credit_per_hour, credit_per_hour),
            (f"{policy.id}:max_monthly_credit", max_monthly_credit, max_monthly_credit),
        ]
    else:
        threshold_gb = str(fields["threshold_gb"])
        throttle_mbps = str(fields["throttle_mbps"])
        detail = (
            f"Usage above {threshold_gb} GB in a billing cycle may be throttled to "
            f"{throttle_mbps} Mbps."
        )
        placements += [
            (f"{policy.id}:threshold_gb", threshold_gb, threshold_gb),
            (f"{policy.id}:throttle_mbps", throttle_mbps, throttle_mbps),
        ]
    context = {"name": name, "policy_detail": detail}
    return context, placements


def _fields_dump(entity_id: str, fields: dict) -> tuple[str, list[tuple[str, str, str]]]:
    # Every field on the entity, sorted for determinism, each recorded as its own placement. Used
    # where the field set genuinely varies per instance (promotions, regions) so no field is ever
    # silently missed. The anchor equals the value: each field renders on its own "field: value"
    # line, and lines/placements are built from the same sorted iteration, so the sequential
    # cursor in _render_doc walks fields_dump's own placements in the same order they appear in
    # the rendered text, correctly disambiguating any two fields that happen to share a value.
    lines = [f"{field}: {value}" for field, value in sorted(fields.items())]
    placements = [(f"{entity_id}:{field}", str(value), str(value)) for field, value in sorted(fields.items())]
    return "\n".join(lines), placements


def _promo_context(promo: Entity, reg: Registry) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    fields = promo.fields
    try:
        plan_id = next(e.dst for e in reg.edges if e.relation == "applies_to" and e.src == promo.id)
    except StopIteration:
        # Same condition compile.integrity_report calls "promotion has no applies_to edge"; a
        # registry authoring error must fail closed with that name, not a bare StopIteration out
        # of a generator expression.
        raise RegistryError(f"promotion {promo.id} has no applies_to edge") from None
    plan = reg.entity(plan_id)
    dump_text, dump_placements = _fields_dump(promo.id, fields)
    if "promo_monthly_price" in fields:
        detail = f"The promotional price is {fields['promo_monthly_price']} per month."
    elif "credit_amount" in fields:
        detail = f"This promotion provides a {fields['credit_amount']} credit."
    else:
        detail = "This promotion carries no price adjustment on record."
    if "equipment_rental_waived" in fields:
        waiver = f"Equipment rental waived: {fields['equipment_rental_waived']}."
    else:
        waiver = "No equipment rental waiver is claimed by this promotion."
    plan_name = plan.fields["name"]
    context = {
        "name": fields["name"],
        "plan_name": plan_name,
        "promo_detail": detail,
        "waiver_clause": waiver,
        "fields_dump": dump_text,
    }
    placements = [(f"{plan.id}:name", plan_name, plan_name), *dump_placements]
    return context, placements


def _fee_schedule_context(region: Entity, reg: Registry) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    fields = region.fields
    dump_text, dump_placements = _fields_dump(region.id, fields)
    if "equipment_rental_override_amount" in fields:
        override_line = f"Equipment rental override in this region: {fields['equipment_rental_override_amount']}."
    else:
        override_line = "No fee overrides apply in this region."
    # fee_lines and fee_placements are built from the same sorted-by-id fees list, in the same
    # order they will appear in the rendered fee_list text, for the same reason _fields_dump's
    # lines and placements share one sorted iteration: the sequential span cursor needs
    # placement order to match text order to disambiguate fees that share an amount.
    fees = sorted((e for e in reg.by_kind("fee") if e.render), key=lambda e: e.id)
    fee_lines = [f"{fee.fields['name']}: {fee.fields['amount']}" for fee in fees]
    fee_placements = [(f"{fee.id}:amount", str(fee.fields["amount"]), str(fee.fields["amount"])) for fee in fees]
    context = {
        "name": fields["name"],
        "override_line": override_line,
        "fields_dump": dump_text,
        "fee_list": "\n".join(fee_lines),
    }
    placements = [*dump_placements, *fee_placements]
    return context, placements


def _plan_family(reg: Registry, base_plan_id: str) -> tuple[Entity, ...]:
    base = reg.entity(base_plan_id)
    variants = [e for e in reg.by_kind("plan") if e.render and e.fields.get("variant_of") == base_plan_id]
    members = [base, *variants] if base.render else variants
    return tuple(sorted(members, key=lambda e: e.id))


def _locate_span(text: str, anchor: str, cursor: int, *, doc_id: str, fact_ref: str) -> tuple[int, int]:
    # Search forward from cursor first: placements are supplied in roughly the order their
    # anchors appear in the template, so this alone resolves same-valued neighbors correctly
    # (e.g. a symmetric plan's download_mbps and upload_mbps both being "100" finds the first
    # "100" for download, then advances past it before upload_mbps searches for the second).
    # Anchors that render earlier than the current cursor (a prose clause hoisted to the top of
    # a template variant while its context key is late in placement order) fall back to a full
    # document search rather than a whole text re-search per placement being the default: a
    # cursor-scoped find is still the primary lookup, this is only a retry when it comes up
    # empty, and it costs nothing extra for the common case where the primary search succeeds.
    idx = text.find(anchor, cursor)
    if idx == -1:
        idx = text.find(anchor)
    if idx == -1:
        raise ValueError(f"{doc_id}: could not locate rendered span for {fact_ref} (anchor {anchor!r})")
    return idx, idx + len(anchor)


def _render_doc(
    doc_id: str,
    doc_type: str,
    templates: dict[str, str],
    seed_key: str,
    context: dict[str, str],
    raw_placements: list[tuple[str, str, str]],
) -> RenderedDoc:
    rng = random.Random(seed_key)
    variant = rng.choice(_VARIANTS[doc_type])
    text = templates[f"{doc_type}_{variant}"].format(**context)
    placements: list[Placement] = []
    cursor = 0
    for fact_ref, value, anchor in raw_placements:
        span = _locate_span(text, anchor, cursor, doc_id=doc_id, fact_ref=fact_ref)
        placements.append(Placement(fact_ref=fact_ref, value=value, span=span))
        cursor = span[1]
    return RenderedDoc(doc_id=doc_id, doc_type=doc_type, text=text, placements=tuple(placements))


def render_corpus(reg: Registry, templates_dir: Path, seed: int) -> tuple[RenderedDoc, ...]:
    templates = _load_templates(templates_dir)
    docs: list[RenderedDoc] = []

    for plan in reg.by_kind("plan"):
        if not plan.render:
            continue
        context, placements = _plan_context(plan)
        docs.append(_render_doc(f"doc-plan_page-{plan.id}", "plan_page", templates, f"{seed}:{plan.id}", context, placements))

    for term in reg.by_kind("contract_term"):
        if not term.render:
            continue
        context, placements = _contract_term_context(term, reg)
        docs.append(
            _render_doc(f"doc-contract_terms-{term.id}", "contract_terms", templates, f"{seed}:{term.id}", context, placements)
        )

    for device in reg.by_kind("device"):
        if not device.render:
            continue
        manual_context, manual_placements = _device_manual_context(device)
        docs.append(
            _render_doc(
                f"doc-device_manual-{device.id}", "device_manual", templates, f"{seed}:{device.id}", manual_context, manual_placements
            )
        )

        compatible_base_ids = sorted(
            {e.dst for e in reg.edges if e.relation == "compatible_with" and e.src == device.id}
        )
        for base_id in compatible_base_ids:
            for plan in _plan_family(reg, base_id):
                entity_id = f"{device.id}--{plan.id}"
                context, placements = _troubleshooting_context(device, plan)
                docs.append(
                    _render_doc(f"doc-troubleshooting-{entity_id}", "troubleshooting", templates, f"{seed}:{entity_id}", context, placements)
                )

    for policy in reg.by_kind("policy"):
        if not policy.render:
            continue
        context, placements = _policy_context(policy)
        docs.append(_render_doc(f"doc-policy-{policy.id}", "policy", templates, f"{seed}:{policy.id}", context, placements))

    for promo in reg.by_kind("promotion"):
        if not promo.render:
            continue
        context, placements = _promo_context(promo, reg)
        docs.append(_render_doc(f"doc-promo_page-{promo.id}", "promo_page", templates, f"{seed}:{promo.id}", context, placements))

    for region in reg.by_kind("region"):
        if not region.render:
            continue
        context, placements = _fee_schedule_context(region, reg)
        docs.append(
            _render_doc(f"doc-fee_schedule-{region.id}", "fee_schedule", templates, f"{seed}:{region.id}", context, placements)
        )

    return tuple(sorted(docs, key=lambda d: d.doc_id))
