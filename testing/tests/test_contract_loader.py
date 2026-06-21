"""The contract loader: every family loads, is valid Draft 2020-12, and declares a semver version."""

from __future__ import annotations

import re

import jsonschema
import pytest
from contract_tools import loader

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def test_families_are_the_four_from_the_hld() -> None:
    assert loader.FAMILIES == ("trace", "dataset", "manifest", "sse")


@pytest.mark.parametrize("family", ["trace", "dataset", "manifest", "sse"])
def test_schema_loads_and_is_valid_draft_2020_12(family: str) -> None:
    schema = loader.load_schema(family)
    jsonschema.validators.validator_for(schema).check_schema(schema)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith(f"/{family}/{schema['x-contract-version']}")


@pytest.mark.parametrize("family", ["trace", "dataset", "manifest", "sse"])
def test_schema_declares_semver_contract_version(family: str) -> None:
    assert SEMVER.match(loader.load_schema(family)["x-contract-version"])


def test_contract_versions_returns_the_full_tuple() -> None:
    versions = loader.contract_versions()
    assert set(versions) == set(loader.FAMILIES)
    assert all(SEMVER.match(v) for v in versions.values())


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown contract family"):
        loader.load_schema("prompts")
