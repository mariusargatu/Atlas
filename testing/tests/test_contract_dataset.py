"""Dataset contract: one case schema with a turns array; single turn is just length 1."""

from __future__ import annotations

import jsonschema
import pytest
from contract_tools import loader


@pytest.fixture(scope="module")
def schema() -> dict:
    return loader.load_schema("dataset")


@pytest.fixture(scope="module")
def examples() -> dict[str, dict]:
    return loader.load_examples("dataset")


@pytest.mark.parametrize("name", ["single_turn_case", "multi_turn_case"])
def test_example_validates(schema: dict, examples: dict, name: str) -> None:
    jsonschema.validate(examples[name], schema)


def test_single_turn_case_is_a_length_one_turns_array(examples: dict) -> None:
    assert len(examples["single_turn_case"]["turns"]) == 1
    assert len(examples["multi_turn_case"]["turns"]) > 1


def test_empty_turns_is_rejected(schema: dict, examples: dict) -> None:
    bad = {**examples["single_turn_case"], "turns": []}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_split_is_a_closed_enum(schema: dict, examples: dict) -> None:
    bad = {**examples["single_turn_case"], "split": "production"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_unknown_top_level_fields_are_rejected(schema: dict, examples: dict) -> None:
    bad = {**examples["single_turn_case"], "notes": "free text"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_required_core_matches_the_hld(schema: dict) -> None:
    assert set(schema["required"]) == {"case_id", "split", "origin", "intent", "answerable", "turns"}


def test_stray_field_inside_a_turn_is_rejected(schema: dict, examples: dict) -> None:
    bad = {**examples["single_turn_case"]}
    bad["turns"] = [{"user": "Is my plan contract free?", "mood": "curious"}]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)
