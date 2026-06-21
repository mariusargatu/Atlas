"""Trace contract: golden examples validate and the reserved production spine fields exist."""

from __future__ import annotations

import jsonschema
import pytest
from contract_tools import loader

DEGRADATION_MODES = {"none", "retry", "provider_fallback", "drop_rerank", "lexical_only", "refusal"}


@pytest.fixture(scope="module")
def schema() -> dict:
    return loader.load_schema("trace")


@pytest.fixture(scope="module")
def examples() -> dict[str, dict]:
    return loader.load_examples("trace")


def test_both_golden_examples_exist(examples: dict) -> None:
    assert set(examples) == {"chat_turn", "degraded_turn"}


@pytest.mark.parametrize("name", ["chat_turn", "degraded_turn"])
def test_example_validates(schema: dict, examples: dict, name: str) -> None:
    jsonschema.validate(examples[name], schema)


def test_every_reserved_attribute_is_declared(schema: dict) -> None:
    declared = set(schema["properties"]["attributes"]["properties"])
    missing = set(loader.RESERVED_TRACE_ATTRIBUTES) - declared
    assert not missing, f"reserved attributes missing from trace schema: {sorted(missing)}"


def test_degradation_ladder_is_exactly_five_rungs_plus_none(schema: dict) -> None:
    enum = schema["properties"]["attributes"]["properties"]["atlas.degradation.mode"]["enum"]
    assert set(enum) == DEGRADATION_MODES and len(enum) == 6


def test_contract_tuple_and_synthetic_flag_are_required(schema: dict) -> None:
    required = set(schema["properties"]["attributes"]["required"])
    assert "atlas.contract.trace_version" in required
    assert "atlas.privacy.synthetic" in required


def test_user_feedback_event_requires_a_label(schema: dict, examples: dict) -> None:
    bad = dict(examples["chat_turn"])
    bad["events"] = [{"name": "user_feedback", "attributes": {"comment": "no label"}}]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
