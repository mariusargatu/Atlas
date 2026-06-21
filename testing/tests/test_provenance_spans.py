"""Every placement carries a char span that slices to the text expressing the fact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from corpus_tools import registry, render

CORE = Path("corpus/registry/core.yaml")
GENERATED = Path("corpus/registry/generated_variants.yaml")
TEMPLATES = Path("corpus/templates")
COMMITTED = Path("corpus/rendered/corpus-0.1.1")


@pytest.fixture(scope="module")
def docs() -> tuple[render.RenderedDoc, ...]:
    reg = registry.load_registry([CORE, GENERATED])
    return render.render_corpus(reg, TEMPLATES, seed=7)


def test_every_placement_has_a_valid_span(docs) -> None:
    for doc in docs:
        for placement in doc.placements:
            start, end = placement.span
            assert 0 <= start < end <= len(doc.text), f"{doc.doc_id}: bad span for {placement.fact_ref}"


def test_prose_branch_spans_cover_the_contract_clause(docs) -> None:
    page = next(d for d in docs if d.doc_id == "doc-plan_page-plan-fiber-100")
    months = next(p for p in page.placements if p.fact_ref == "plan-fiber-100:contract_months")
    start, end = months.span
    assert page.text[start:end] == "No contract. Cancel any time."


def test_literal_spans_slice_to_their_value(docs) -> None:
    for doc in docs:
        for placement in doc.placements:
            start, end = placement.span
            sliced = doc.text[start:end]
            if placement.value in sliced:
                continue
            # The one documented prose exception (see test_prose_branch_spans_cover_the_contract_clause):
            # contract_months=0 renders as "No contract. Cancel any time.", which never contains the
            # digit "0" as a token, so its span records the whole clause instead of the bare value.
            # Anything outside that exact, named case is a real failure to locate the fact's span.
            assert placement.fact_ref.endswith(":contract_months") and placement.value == "0", (
                f"{doc.doc_id} {placement.fact_ref}: span slices to {sliced!r}, value {placement.value!r} absent"
            )


def test_committed_sidecars_carry_spans() -> None:
    for sidecar_path in sorted((COMMITTED / "provenance").glob("*.json")):
        sidecar = json.loads(sidecar_path.read_text())
        doc_text = (COMMITTED / "docs" / f"{sidecar['doc_id']}.txt").read_text()
        for entry in sidecar["placements"]:
            start, end = entry["span"]
            assert entry["value"] in doc_text[start:end] or doc_text[start:end] == entry["clause"], sidecar_path.name


def test_all_no_contract_spans_slice_to_the_exact_clause(docs) -> None:
    # A golden, non tautological check: the exact clause string is asserted here as a fixed
    # literal, independent of anything build.py or verify.py derived from the span itself (both
    # of those compare a span's slice against text taken from that SAME span, which cannot catch
    # a span that has drifted to point somewhere else entirely). This must hold across all 10
    # no contract plan pages (fiber-100, fiber-500, starter-50 and every one of their regional
    # variants), not just the one doc test_prose_branch_spans_cover_the_contract_clause pins, and
    # across both surfaces the corpus exposes provenance through: the live render and the
    # committed sidecars, since either one could drift independently of the other.
    exact_clause = "No contract. Cancel any time."
    checked_live = 0
    for doc in docs:
        for placement in doc.placements:
            if not (placement.fact_ref.endswith(":contract_months") and placement.value == "0"):
                continue
            start, end = placement.span
            assert doc.text[start:end] == exact_clause, f"{doc.doc_id} {placement.fact_ref}: live span drifted"
            checked_live += 1
    assert checked_live == 10, f"expected 10 no contract placements in the live render, found {checked_live}"

    checked_committed = 0
    for sidecar_path in sorted((COMMITTED / "provenance").glob("*.json")):
        sidecar = json.loads(sidecar_path.read_text())
        doc_text = (COMMITTED / "docs" / f"{sidecar['doc_id']}.txt").read_text()
        for entry in sidecar["placements"]:
            if not (entry["fact_ref"].endswith(":contract_months") and entry["value"] == "0"):
                continue
            start, end = entry["span"]
            assert doc_text[start:end] == exact_clause, f"{sidecar_path.name} {entry['fact_ref']}: committed span drifted"
            checked_committed += 1
    assert checked_committed == 10, f"expected 10 no contract placements in the committed corpus, found {checked_committed}"
