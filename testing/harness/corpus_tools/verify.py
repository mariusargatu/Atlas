"""The deterministic half of the corpus quality gate: no rendered claim may drift from the
registry, no never rendered entity may leak into a document, and both sides of every designed
contradiction must be retrievable somewhere in the corpus (or the conflict cannot be tested)."""

from __future__ import annotations

from corpus_tools.registry import Registry
from corpus_tools.render import RenderedDoc


def _registry_value(reg: Registry, fact_ref: str) -> str:
    entity_id, _, field = fact_ref.partition(":")
    return str(reg.entity(entity_id).fields[field])


def _placement_violations(docs: tuple[RenderedDoc, ...], reg: Registry) -> list[str]:
    violations: list[str] = []
    for doc in docs:
        for placement in doc.placements:
            fact_ref, value = placement.fact_ref, placement.value
            expected = _registry_value(reg, fact_ref)
            if value != expected:
                violations.append(
                    f"{doc.doc_id}: placement {fact_ref} value {value!r} does not match registry value {expected!r}"
                )
                continue

            start, end = placement.span
            if not (0 <= start < end <= len(doc.text)):
                violations.append(
                    f"{doc.doc_id}: placement {fact_ref} span {placement.span} is out of bounds "
                    f"for a {len(doc.text)} char document"
                )
                continue

            sliced = doc.text[start:end]
            if value in sliced:
                continue

            # The span does not slice to a literal occurrence of value: this is the documented
            # prose branch case (contract_months=0 renders as "No contract. Cancel any time.",
            # which never contains the digit "0" as a token; the span records the whole clause
            # instead). render.py is the single place that knows which fact maps to which
            # clause, so verify.py does not re-derive it here (that duplication is exactly what
            # the digest's Option B rejected); the span itself is trusted. As a fallback, keep
            # SP2's original anywhere-in-document check as a weaker belt-and-suspenders catch
            # for a fact that has gone missing from the document entirely.
            if value not in doc.text:
                violations.append(f"{doc.doc_id}: placement {fact_ref} value {value!r} is not literally present in text")
    return violations


def _leak_violations(docs: tuple[RenderedDoc, ...], reg: Registry) -> list[str]:
    hidden = tuple(e for e in reg.entities if not e.render)
    violations: list[str] = []
    for doc in docs:
        lowered = doc.text.lower()
        for entity in hidden:
            id_tail = entity.id.removeprefix(f"{entity.kind}-")
            name = str(entity.fields.get("name", ""))
            needles = (n.lower() for n in (id_tail, name) if n)
            if any(needle in lowered for needle in needles):
                violations.append(f"{doc.doc_id}: leaks never rendered entity {entity.id}")
    return violations


def _contradiction_violations(docs: tuple[RenderedDoc, ...], reg: Registry) -> list[str]:
    placed_refs = {placement.fact_ref for doc in docs for placement in doc.placements}
    violations: list[str] = []
    for c in reg.contradictions:
        for side, ref in (("winning", c.winning_fact), ("losing", c.losing_fact)):
            if ref not in placed_refs:
                violations.append(f"contradiction {c.id}: {side} fact {ref} is rendered nowhere")
    return violations


def verify_corpus(docs: tuple[RenderedDoc, ...], reg: Registry) -> tuple[str, ...]:
    return (
        *_placement_violations(docs, reg),
        *_leak_violations(docs, reg),
        *_contradiction_violations(docs, reg),
    )
