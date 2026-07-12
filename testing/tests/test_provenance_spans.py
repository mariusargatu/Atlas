"""Every placement carries a char span that slices to the text expressing the fact."""

from __future__ import annotations

import json

import pytest
from corpus_tools import expand, registry, render
from .fixtures.corpus_expectations import COMMITTED_CORPUS_DIR as COMMITTED, CORE, GENERATED, TEMPLATES


@pytest.fixture(scope="module")
def docs() -> tuple[render.RenderedDoc, ...]:
    reg = registry.load_registry([CORE, GENERATED])
    # `expand.DEFAULT_SEED` is the seed `corpus_tools.build` actually renders with, and therefore
    # the only seed whose output can be compared against the committed corpus. A different literal
    # here silently exercises a different set of template variants than the repo ships.
    return render.render_corpus(reg, TEMPLATES, seed=expand.DEFAULT_SEED)


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


# --- spans must land on the RIGHT occurrence, not merely on matching text -------------------------


def test_no_two_placements_in_a_doc_share_a_span(docs) -> None:
    """`value in text[span]` cannot catch a span pointing at the wrong instance of the same value,
    because "100" is in "100". Two distinct facts resolving to the identical span always can."""
    for doc in docs:
        spans = [(p.fact_ref, p.span) for p in doc.placements]
        duplicates = {span for _, span in spans if [s for _, s in spans].count(span) > 1}
        assert not duplicates, f"{doc.doc_id}: distinct facts share span(s) {duplicates}"


def test_symmetric_plan_download_and_upload_span_the_figures_the_prose_labels(docs) -> None:
    """The exact defect the whole document `str.find` walk produced. Fiber 100 is symmetric, so
    its name, its download figure and its upload figure all render the string "100". The old
    cursor walk gave `download_mbps` the "100" inside the plan NAME and gave `upload_mbps` the
    "100" the prose calls DOWNLOAD, leaving the real upload figure unspanned. Every `value in
    sliced` guard passed throughout."""
    page = next(d for d in docs if d.doc_id == "doc-plan_page-plan-fiber-100")
    by_ref = {p.fact_ref: p.span for p in page.placements}
    download, upload = by_ref["plan-fiber-100:download_mbps"], by_ref["plan-fiber-100:upload_mbps"]
    name = by_ref["plan-fiber-100:name"]

    assert download != upload != name and download != name
    # the words immediately following each span are what the prose calls that figure
    assert page.text[download[1] : download[1] + 15].strip().startswith("Mbps download")
    assert page.text[upload[1] : upload[1] + 15].strip().startswith("Mbps upload")
    # and neither may sit inside the plan name's own span
    for span in (download, upload):
        assert not (name[0] <= span[0] < name[1]), "a figure was attributed to the plan name"


def test_every_committed_sidecar_span_is_labelled_by_its_own_prose(docs) -> None:
    """Across the whole corpus: a numeric fact must never be spanned inside another fact's value.
    Checked by span containment, which is what `value in sliced` structurally cannot see."""
    for doc in docs:
        spans = sorted((p.span, p.fact_ref) for p in doc.placements)
        for (a_span, a_ref), (b_span, b_ref) in zip(spans, spans[1:]):
            assert a_span[1] <= b_span[0] or a_span == b_span, (
                f"{doc.doc_id}: {a_ref} {a_span} overlaps {b_ref} {b_span}"
            )


def test_the_committed_sidecars_match_a_fresh_render() -> None:
    """The staleness gate for `corpus/rendered/*/provenance/`: the committed spans must be the ones
    the current renderer produces. Without it, a span fix (or a span regression) can sit in the code
    while the committed sidecars keep describing the old behaviour."""
    reg = registry.load_registry([CORE, GENERATED])
    rendered = {d.doc_id: d for d in render.render_corpus(reg, TEMPLATES, seed=expand.DEFAULT_SEED)}
    for sidecar_path in sorted((COMMITTED / "provenance").glob("*.json")):
        sidecar = json.loads(sidecar_path.read_text())
        doc = rendered[sidecar["doc_id"]]
        assert [(p["fact_ref"], tuple(p["span"])) for p in sidecar["placements"]] == [
            (p.fact_ref, p.span) for p in doc.placements
        ], sidecar_path.name
