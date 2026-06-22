"""The versioned failure taxonomy (SP8 Task 5, D34): contracts/dataset/taxonomy.yaml, loaded and
validated the same fail closed way corpus_tools.registry.load_registry validates the fact registry
(test_corpus_registry.py is this file's own structural precedent).

`failure_class` (contracts/dataset/schema.json) stays a free string, never a JSON Schema enum: this
module, not the schema, is what checks a value against the taxonomy's known codes, per the digest's
own design question 6. `test_contract_dataset.py` already proves the schema accepts any string or
null for failure_class; this file proves the APPLICATION layer rejects an unknown one.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from dataset_tools import taxonomy


@pytest.fixture(scope="module")
def tax() -> taxonomy.Taxonomy:
    return taxonomy.load_taxonomy()


# ---- the committed taxonomy.yaml -------------------------------------------------------------------


def test_committed_taxonomy_loads(tax: taxonomy.Taxonomy) -> None:
    assert tax.version == "0.1.0"
    assert len(tax.codes) > 0


def test_committed_taxonomy_has_no_duplicate_ids(tax: taxonomy.Taxonomy) -> None:
    ids = [c.id for c in tax.codes]
    assert len(set(ids)) == len(ids)


def test_committed_taxonomy_starter_codes_are_present(tax: taxonomy.Taxonomy) -> None:
    expected = {
        "ungrounded_claim", "missed_refusal", "false_refusal", "wrong_tool",
        "stale_fact", "conflict_misresolution", "hallucinated_entity",
    }
    assert expected <= tax.code_ids


def test_committed_taxonomy_every_code_has_description_and_example(tax: taxonomy.Taxonomy) -> None:
    for code in tax.codes:
        assert code.description.strip()
        assert code.example.strip()


# ---- failure_class validation, both directions -----------------------------------------------------


def test_known_code_passes(tax: taxonomy.Taxonomy) -> None:
    tax.validate_failure_class("ungrounded_claim")  # no raise


def test_none_always_passes(tax: taxonomy.Taxonomy) -> None:
    tax.validate_failure_class(None)  # a case with no failure_class at all: not this module's concern


def test_unknown_code_is_rejected(tax: taxonomy.Taxonomy) -> None:
    with pytest.raises(taxonomy.TaxonomyError, match="not_a_real_code"):
        tax.validate_failure_class("not_a_real_code")


def test_is_known_matches_validate_failure_class(tax: taxonomy.Taxonomy) -> None:
    assert tax.is_known("wrong_tool") is True
    assert tax.is_known("not_a_real_code") is False


# ---- loader failure modes, fail closed --------------------------------------------------------------


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "taxonomy.yaml"
    path.write_text(text)
    return path


def test_duplicate_code_id_is_rejected(tmp_path: Path) -> None:
    bad = _write(
        tmp_path,
        "taxonomy_version: '0.1.0'\n"
        "codes:\n"
        "  - id: dup_code\n    description: 'a'\n    example: 'a'\n"
        "  - id: dup_code\n    description: 'b'\n    example: 'b'\n",
    )
    with pytest.raises(taxonomy.TaxonomyError, match="dup_code"):
        taxonomy.load_taxonomy(bad)


@pytest.mark.parametrize("bad_version", ["1.0", "1", "1.0.0-beta", "abc", "1.0.0.0", ""])
def test_malformed_semver_is_rejected(tmp_path: Path, bad_version: str) -> None:
    bad = _write(
        tmp_path,
        f"taxonomy_version: '{bad_version}'\n"
        "codes:\n  - id: c1\n    description: 'a'\n    example: 'a'\n",
    )
    with pytest.raises(taxonomy.TaxonomyError, match="semver"):
        taxonomy.load_taxonomy(bad)


def test_missing_taxonomy_version_is_rejected(tmp_path: Path) -> None:
    bad = _write(tmp_path, "codes:\n  - id: c1\n    description: 'a'\n    example: 'a'\n")
    with pytest.raises(taxonomy.TaxonomyError, match="semver"):
        taxonomy.load_taxonomy(bad)


def test_zero_codes_is_rejected(tmp_path: Path) -> None:
    bad = _write(tmp_path, "taxonomy_version: '0.1.0'\ncodes: []\n")
    with pytest.raises(taxonomy.TaxonomyError, match="zero codes"):
        taxonomy.load_taxonomy(bad)


def test_code_missing_description_is_rejected(tmp_path: Path) -> None:
    bad = _write(tmp_path, "taxonomy_version: '0.1.0'\ncodes:\n  - id: c1\n    example: 'a'\n")
    with pytest.raises(taxonomy.TaxonomyError, match="description"):
        taxonomy.load_taxonomy(bad)


def test_code_missing_example_is_rejected(tmp_path: Path) -> None:
    bad = _write(tmp_path, "taxonomy_version: '0.1.0'\ncodes:\n  - id: c1\n    description: 'a'\n")
    with pytest.raises(taxonomy.TaxonomyError, match="example"):
        taxonomy.load_taxonomy(bad)


def test_code_missing_id_is_rejected(tmp_path: Path) -> None:
    bad = _write(tmp_path, "taxonomy_version: '0.1.0'\ncodes:\n  - description: 'a'\n    example: 'a'\n")
    with pytest.raises(taxonomy.TaxonomyError, match="id"):
        taxonomy.load_taxonomy(bad)
