"""The heading aware chunker: graceful single chunk degradation on the committed corpus, real
splitting only on a synthetic oversized fixture, deterministic content addressed chunk ids.

`corpus/rendered/corpus-0.1.1` is small by design (max ~113 estimated tokens per doc): every
one of the committed docs is
smaller than the 300 token split threshold, so `chunk_document` degrades to exactly one chunk per
doc (`parent_id == doc_id`). The splitting branch for oversized sections never fires on that corpus, so
it is exercised here by a fabricated 1200+ token doc with three H2 sections instead.
"""
from __future__ import annotations

import json
import re

import pytest
from rag_tools import chunker
from .fixtures import corpus_expectations
from .fixtures.corpus_expectations import COMMITTED_CORPUS_DIR as COMMITTED, CORPUS_VERSION


_H1_RE = re.compile(r"^# (.+)$", re.MULTILINE)


def _first_h1(text: str) -> str:
    match = _H1_RE.search(text)
    assert match is not None, "fixture text must carry exactly one H1"
    return match.group(1).strip()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((COMMITTED / "manifest.json").read_text())


@pytest.fixture(scope="module")
def committed_docs(manifest) -> tuple[dict, ...]:
    """Every committed doc plus its sidecar and the doc_version read from the manifest, loaded once."""
    docs = []
    for doc_id, doc_version in manifest["docs"].items():
        text = (COMMITTED / "docs" / f"{doc_id}.txt").read_text()
        sidecar = json.loads((COMMITTED / "provenance" / f"{doc_id}.json").read_text())
        docs.append(
            {
                "doc_id": doc_id,
                "doc_type": sidecar["doc_type"],
                "text": text,
                "doc_version": doc_version,
                "placements": sidecar["placements"],
            }
        )
    return tuple(docs)


def _sentence(i: int) -> str:
    return f"This is test sentence number {i} for the oversized fixture."


def _make_oversized_fixture() -> dict:
    """A fabricated ~1300 estimated token doc, one H1 + three H2 sections, well over the 300
    token split threshold. Two marker placements sit at the very start and the very end of the
    body so their overlap with the resulting children is unambiguous regardless of exactly where
    the greedy sentence boundary splitter cuts."""
    start_marker = "Marker start token appears in this sentence."
    end_marker = "This sentence ends with the marker end token."

    section_1 = " ".join([start_marker, *(_sentence(i) for i in range(1, 15))])
    section_2 = " ".join(_sentence(i) for i in range(15, 55))
    section_3 = " ".join([*(_sentence(i) for i in range(55, 95)), end_marker])

    text = (
        "# Oversized Synthetic Doc\n\n"
        "## Overview\n\n" + section_1 + "\n\n"
        "## Details\n\n" + section_2 + "\n\n"
        "## Notes\n\n" + section_3 + "\n"
    )

    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    assert start_idx != -1 and end_idx != -1

    placements = [
        {"fact_ref": "entity-start:field", "span": [start_idx, start_idx + len(start_marker)], "value": start_marker},
        {"fact_ref": "entity-end:field", "span": [end_idx, end_idx + len(end_marker)], "value": end_marker},
    ]
    return {
        "doc_id": "doc-synthetic-oversized",
        "doc_type": "synthetic",
        "text": text,
        "doc_version": "synthetic-version-1",
        "placements": placements,
    }


# --- One chunk per committed doc (regression) ---------------------------------------------------


def test_every_committed_doc_yields_exactly_one_chunk(committed_docs) -> None:
    assert len(committed_docs) == corpus_expectations.DOC_COUNT
    for doc in committed_docs:
        chunks = chunker.chunk_document(
            doc_id=doc["doc_id"],
            doc_type=doc["doc_type"],
            text=doc["text"],
            doc_version=doc["doc_version"],
            corpus_version=CORPUS_VERSION,
            placements=doc["placements"],
        )
        assert len(chunks) == 1, f"{doc['doc_id']}: expected one chunk, got {len(chunks)}"
        (chunk,) = chunks
        assert chunk.parent_id == doc["doc_id"]
        assert chunk.heading_path == (_first_h1(doc["text"]),)
        assert chunk.doc_id == doc["doc_id"]
        assert chunk.doc_type == doc["doc_type"]
        assert chunk.corpus_version == CORPUS_VERSION
        assert chunk.chunker_version == chunker.CHUNKER_VERSION
        assert chunk.char_span == (0, len(doc["text"]))


def test_a_few_known_titles_spot_check(committed_docs) -> None:
    by_id = {doc["doc_id"]: doc for doc in committed_docs}
    expectations = {
        "doc-plan_page-plan-fiber-100": "Fiber 100",
        "doc-device_manual-device-modem-d3": "Modem D3 device manual",
        "doc-policy-policy-fair-use": "Fair Use Policy policy",
        "doc-troubleshooting-device-modem-d3--plan-fiber-100": "Troubleshooting: Modem D3",
    }
    for doc_id, expected_title in expectations.items():
        doc = by_id[doc_id]
        (chunk,) = chunker.chunk_document(
            doc_id=doc["doc_id"],
            doc_type=doc["doc_type"],
            text=doc["text"],
            doc_version=doc["doc_version"],
            corpus_version=CORPUS_VERSION,
            placements=doc["placements"],
        )
        assert chunk.heading_path == (expected_title,)


def test_fiber_100_h1_appears_after_its_h2_sections(committed_docs) -> None:
    # A quirk worth pinning explicitly: this doc's H1 line is NOT the first line (two H2 blocks
    # render before it), so title extraction must scan the whole doc, not just the first line.
    doc = next(d for d in committed_docs if d["doc_id"] == "doc-plan_page-plan-fiber-100")
    assert doc["text"].splitlines()[0].startswith("## Contract terms")
    (chunk,) = chunker.chunk_document(
        doc_id=doc["doc_id"],
        doc_type=doc["doc_type"],
        text=doc["text"],
        doc_version=doc["doc_version"],
        corpus_version=CORPUS_VERSION,
        placements=doc["placements"],
    )
    assert chunk.heading_path == ("Fiber 100",)


# --- entity_ids via span overlap -----------------------------------------------------------------


def test_entity_ids_carries_the_contract_clause_entity(committed_docs) -> None:
    doc = next(d for d in committed_docs if d["doc_id"] == "doc-plan_page-plan-fiber-100")
    clause_placement = next(p for p in doc["placements"] if p["fact_ref"] == "plan-fiber-100:contract_months")
    assert doc["text"][clause_placement["span"][0] : clause_placement["span"][1]] == "No contract. Cancel any time."

    # Isolate the clause placement (drop every other placement) to prove the overlap logic keys
    # off span, not off "value" text matching against the fiber-100:name / :download_mbps digits.
    (chunk,) = chunker.chunk_document(
        doc_id=doc["doc_id"],
        doc_type=doc["doc_type"],
        text=doc["text"],
        doc_version=doc["doc_version"],
        corpus_version=CORPUS_VERSION,
        placements=[clause_placement],
    )
    assert chunk.entity_ids == ("plan-fiber-100",)


def test_entity_ids_deduplicated_and_sorted(committed_docs) -> None:
    doc = next(d for d in committed_docs if d["doc_id"] == "doc-plan_page-plan-fiber-100")
    (chunk,) = chunker.chunk_document(
        doc_id=doc["doc_id"],
        doc_type=doc["doc_type"],
        text=doc["text"],
        doc_version=doc["doc_version"],
        corpus_version=CORPUS_VERSION,
        placements=doc["placements"],
    )
    # Every placement in this doc belongs to the same entity: five fact_refs, one entity_id.
    assert chunk.entity_ids == ("plan-fiber-100",)


def test_entity_ids_excludes_non_overlapping_placements_across_split_children() -> None:
    fixture = _make_oversized_fixture()
    chunks = chunker.chunk_document(
        doc_id=fixture["doc_id"],
        doc_type=fixture["doc_type"],
        text=fixture["text"],
        doc_version=fixture["doc_version"],
        corpus_version=CORPUS_VERSION,
        placements=fixture["placements"],
    )
    assert len(chunks) >= 4
    first, last = chunks[0], chunks[-1]
    assert "entity-start" in first.entity_ids
    assert "entity-end" not in first.entity_ids
    assert "entity-end" in last.entity_ids
    assert "entity-start" not in last.entity_ids


# --- synthetic oversized doc: the splitting branch ------------------------------------------------


def test_split_into_children_never_appends_an_empty_trailing_span() -> None:
    """Crafted regression (SP3 final review): when the last real sentence boundary lands exactly at
    the section's end (trailing whitespace after the final '.') AND that final sentence alone
    already exceeds the ~300 token child target, `_split_into_children`'s own appended
    `len(section_text)` split point duplicates the regex's last real match. Unguarded, that
    duplicate zero-token 'sentence' still trips the cut condition off the running total carried
    over from the real final sentence, landing `child_start` exactly on `len(section_text)` right
    before the function's unconditional trailing append -- which would then emit an empty
    `(len(section_text), len(section_text))` span. This never happens on any committed doc or on
    `_make_oversized_fixture()`'s text (neither ends mid-sentence with trailing whitespace exactly
    at the boundary), which is why it takes a crafted input to reach at all."""
    tail = "word " * 240 + "end. "  # one ~313 estimated token 'sentence', ending in trailing whitespace
    text = "Hi. " + tail
    spans = chunker._split_into_children(text, (0, len(text)))

    assert all(start < end for start, end in spans), f"an empty span slipped through: {spans}"
    assert spans[0] == (0, 4)  # "Hi. " cut off as its own child first
    assert spans[-1][1] == len(text)  # spans still fully cover the text, no content lost
    reconstructed = "".join(text[start:end] for start, end in spans)
    assert reconstructed == text


def test_oversized_doc_splits_into_children_sharing_one_parent() -> None:
    fixture = _make_oversized_fixture()
    chunks = chunker.chunk_document(
        doc_id=fixture["doc_id"],
        doc_type=fixture["doc_type"],
        text=fixture["text"],
        doc_version=fixture["doc_version"],
        corpus_version=CORPUS_VERSION,
        placements=fixture["placements"],
    )
    assert len(chunks) >= 4, f"expected 4+ children, got {len(chunks)}"

    parent_ids = {chunk.parent_id for chunk in chunks}
    assert parent_ids == {fixture["doc_id"]}

    # Spans are contiguous and non overlapping, in document order, and reconstruct the doc text.
    assert chunks[0].char_span[0] == 0
    assert chunks[-1].char_span[1] == len(fixture["text"])
    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier.char_span[1] == later.char_span[0]

    reconstructed = "".join(
        fixture["text"][chunk.char_span[0] : chunk.char_span[1]] for chunk in chunks
    )
    assert reconstructed == fixture["text"]

    # heading_path names the section (the doc's one H1) every child belongs to.
    for chunk in chunks:
        assert chunk.heading_path == ("Oversized Synthetic Doc",)

    # Children roughly target 300 estimated tokens; none should be wildly over target.
    for chunk in chunks:
        assert chunk.token_count <= 300 * 2


def test_oversized_doc_children_have_distinct_chunk_ids() -> None:
    fixture = _make_oversized_fixture()
    chunks = chunker.chunk_document(
        doc_id=fixture["doc_id"],
        doc_type=fixture["doc_type"],
        text=fixture["text"],
        doc_version=fixture["doc_version"],
        corpus_version=CORPUS_VERSION,
        placements=fixture["placements"],
    )
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "children share a doc_id/doc_version/corpus_version/chunker_version but differ by span; ids must differ"


# --- chunk_id determinism --------------------------------------------------------------------------


def _single_chunk(**overrides) -> chunker.ChunkRecord:
    defaults = {
        "doc_id": "doc-fixture",
        "doc_type": "plan_page",
        "text": "# Fixture Doc\n\nA short fixture body under the split threshold.",
        "doc_version": "version-a",
        "corpus_version": CORPUS_VERSION,
        "placements": [],
    }
    defaults.update(overrides)
    (chunk,) = chunker.chunk_document(**defaults)
    return chunk


def test_chunk_id_is_stable_across_repeated_calls() -> None:
    first = _single_chunk()
    second = _single_chunk()
    assert first.chunk_id == second.chunk_id


def test_chunk_id_flips_with_corpus_version() -> None:
    baseline = _single_chunk()
    changed = _single_chunk(corpus_version="corpus-9.9.9")
    assert baseline.chunk_id != changed.chunk_id


def test_chunk_id_flips_with_doc_id() -> None:
    baseline = _single_chunk()
    changed = _single_chunk(doc_id="doc-other-fixture")
    assert baseline.chunk_id != changed.chunk_id


def test_chunk_id_flips_with_doc_version() -> None:
    baseline = _single_chunk()
    changed = _single_chunk(doc_version="version-b")
    assert baseline.chunk_id != changed.chunk_id


def test_chunk_id_flips_with_chunker_version(monkeypatch) -> None:
    baseline = _single_chunk()
    monkeypatch.setattr(chunker, "CHUNKER_VERSION", "hchunk-9")
    changed = _single_chunk()
    assert baseline.chunk_id != changed.chunk_id
    assert changed.chunker_version == "hchunk-9"


def test_chunk_id_flips_with_span_via_split_children() -> None:
    # Same doc_id/doc_version/corpus_version/chunker_version for every child in one call; only
    # span differs, and every chunk_id differs (covered again here as the dedicated span case).
    fixture = _make_oversized_fixture()
    chunks = chunker.chunk_document(
        doc_id=fixture["doc_id"],
        doc_type=fixture["doc_type"],
        text=fixture["text"],
        doc_version=fixture["doc_version"],
        corpus_version=CORPUS_VERSION,
        placements=fixture["placements"],
    )
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_id_and_content_hash_are_16_hex_chars() -> None:
    chunk = _single_chunk()
    assert len(chunk.chunk_id) == 16
    int(chunk.chunk_id, 16)  # raises if not hex
    assert len(chunk.content_hash) == 16
    int(chunk.content_hash, 16)


# --- contextual_header ------------------------------------------------------------------------------


def test_contextual_header_format_single_element_path() -> None:
    assert chunker.contextual_header("Fiber 100", ("Fiber 100",)) == "Fiber 100 > Fiber 100"


def test_contextual_header_format_multi_element_path() -> None:
    header = chunker.contextual_header("Doc Title", ("Section A", "Subsection B"))
    assert header == "Doc Title > Section A > Subsection B"


def test_embed_text_prepends_the_contextual_header() -> None:
    chunk = _single_chunk()
    expected_header = chunker.contextual_header(chunk.doc_title, chunk.heading_path)
    assert chunk.embed_text == f"{expected_header}\n{chunk.text}"
    assert chunk.text not in expected_header.splitlines()  # header itself excludes chunk text


# --- chunker_hash -------------------------------------------------------------------------------


def test_chunker_hash_is_a_stable_16_hex_string() -> None:
    first = chunker.chunker_hash()
    second = chunker.chunker_hash()
    assert first == second
    assert len(first) == 16
    int(first, 16)


def test_content_hash_matches_chunk_text() -> None:
    import hashlib

    chunk = _single_chunk()
    expected = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:16]
    assert chunk.content_hash == expected
