"""Manifest contract: the 12 field lineage tuple is required and model aliases are banned."""

from __future__ import annotations

import jsonschema
import pytest
from contract_tools import loader


@pytest.fixture(scope="module")
def schema() -> dict:
    return loader.load_schema("manifest")


@pytest.fixture(scope="module")
def example() -> dict:
    return loader.load_examples("manifest")["benchmark_run"]


def test_example_validates(schema: dict, example: dict) -> None:
    jsonschema.validate(example, schema)


def test_required_is_exactly_lineage_plus_contract_versions(schema: dict) -> None:
    assert set(schema["required"]) == set(loader.LINEAGE_FIELDS) | {"contract_versions"}


def test_model_revision_latest_is_banned(schema: dict, example: dict) -> None:
    bad = {**example, "model_snapshot": {**example["model_snapshot"], "revision": "latest"}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_embedding_model_revision_latest_is_banned(schema: dict, example: dict) -> None:
    bad = {**example, "embedding_model": {**example["embedding_model"], "revision": "latest"}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_judge_id_is_required_but_nullable(schema: dict, example: dict) -> None:
    jsonschema.validate({**example, "judge_id": None}, schema)
    missing = {k: v for k, v in example.items() if k != "judge_id"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing, schema)


def test_contract_versions_covers_all_four_families(schema: dict) -> None:
    assert set(schema["properties"]["contract_versions"]["required"]) == set(loader.FAMILIES)
