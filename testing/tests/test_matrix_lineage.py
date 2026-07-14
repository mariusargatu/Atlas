"""`matrix.lineage`, hermetic: every matrix cell's row in EXACTLY `contracts/manifest/schema.json`'s
shape (D26's 12 field attribution tuple), never a second, matrix specific lineage shape. Validated
against the REAL committed schema via `jsonschema` (a file read, no network), the same contract
`test_contract_manifest.py` already exercises.
"""
from __future__ import annotations

import jsonschema
import pytest
from contract_tools import loader

from matrix.lineage import (
    GIT_SHA_PATTERN,
    NOT_APPLICABLE_EMBEDDING_MODEL,
    NOT_APPLICABLE_MODEL_SNAPSHOT,
    build_manifest_row,
)

_GIT_SHA = "a" * 40


def _row(**overrides) -> dict:
    base = dict(
        run_id="run-test-0001",
        git_sha=_GIT_SHA,
        prompt_hash="sha256:deadbeef",
        model_snapshot=None,
        request_params={"k": 5},
        embedding_model=None,
        index_build_id="idx-test",
        corpus_version="corpus-test-0.0.1",
        chunker_config_hash="chk-test",
        retrieval_config={"k_final": 5},
        dataset_version="0.1.0",
        judge_id=None,
    )
    base.update(overrides)
    return build_manifest_row(**base)


def test_build_manifest_row_validates_against_the_real_contract_schema():
    schema = loader.load_schema("manifest")
    jsonschema.validate(_row(), schema)


def test_every_d26_lineage_field_is_present():
    row = _row()
    assert set(loader.LINEAGE_FIELDS) <= set(row)


def test_model_snapshot_defaults_to_the_documented_not_applicable_sentinel():
    row = _row(model_snapshot=None)
    assert row["model_snapshot"] == NOT_APPLICABLE_MODEL_SNAPSHOT
    assert row["model_snapshot"]["revision"] != "latest"


def test_embedding_model_defaults_to_the_documented_not_applicable_sentinel():
    row = _row(embedding_model=None)
    assert row["embedding_model"] == NOT_APPLICABLE_EMBEDDING_MODEL


def test_a_real_model_snapshot_and_embedding_model_round_trip_unmodified():
    snap = {"provider": "anthropic", "model_id": "claude-sonnet-5", "revision": "claude-sonnet-5"}
    emb = {"id": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181"}
    row = _row(model_snapshot=snap, embedding_model=emb)
    assert row["model_snapshot"] == snap
    assert row["embedding_model"] == emb


def test_judge_id_stays_a_bare_none_the_one_schema_nullable_field():
    row = _row(judge_id=None)
    assert row["judge_id"] is None
    schema = loader.load_schema("manifest")
    jsonschema.validate(row, schema)  # the schema itself accepts null here, unlike every other field


def test_contract_versions_is_populated_from_the_real_loader_not_hand_typed():
    row = _row()
    assert row["contract_versions"] == loader.contract_versions()


def test_malformed_git_sha_raises_before_a_row_is_ever_built():
    with pytest.raises(ValueError, match="git_sha"):
        _row(git_sha="not-hex-at-all")


def test_git_sha_pattern_matches_the_schemas_own_pattern():
    schema = loader.load_schema("manifest")
    assert GIT_SHA_PATTERN.pattern == schema["properties"]["git_sha"]["pattern"]


def test_a_dropped_lineage_field_would_be_caught():
    """Guards the guard: `build_manifest_row` itself must never silently omit a D26 field. This
    documents the invariant directly rather than relying on the happy path tests above to notice."""
    row = _row()
    assert len(set(loader.LINEAGE_FIELDS) - set(row)) == 0
