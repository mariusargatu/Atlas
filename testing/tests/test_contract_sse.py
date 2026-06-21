"""SSE contract: every event in the golden sequences validates; message_end is the terminal event."""

from __future__ import annotations

import jsonschema
import pytest
from contract_tools import loader

EVENT_TYPES = {"message_start", "token", "citation", "degradation", "error", "message_end"}


@pytest.fixture(scope="module")
def schema() -> dict:
    return loader.load_schema("sse")


@pytest.fixture(scope="module")
def sequences() -> dict[str, list]:
    return loader.load_examples("sse")


def test_defs_cover_the_event_vocabulary(schema: dict) -> None:
    assert set(schema["$defs"]) == EVENT_TYPES


@pytest.mark.parametrize("name", ["stream_ok", "stream_error_midstream"])
def test_every_event_in_sequence_validates(schema: dict, sequences: dict, name: str) -> None:
    for event in sequences[name]:
        jsonschema.validate(event, schema)


@pytest.mark.parametrize("name", ["stream_ok", "stream_error_midstream"])
def test_sequences_end_with_the_terminal_event(sequences: dict, name: str) -> None:
    assert sequences[name][0]["event"] == "message_start"
    assert sequences[name][-1]["event"] == "message_end"


def test_error_event_requires_recoverable_flag(schema: dict) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"event": "error", "code": "provider_529", "message": "overloaded"}, schema)


def test_error_midstream_sequence_finishes_with_reason_error(sequences: dict) -> None:
    assert sequences["stream_error_midstream"][-1]["finish_reason"] == "error"


def test_token_event_rejects_an_unknown_extra_field(schema: dict) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"event": "token", "text": "hi", "unexpected": True}, schema)


def test_message_end_rejects_an_out_of_vocabulary_finish_reason(schema: dict) -> None:
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"event": "message_end", "finish_reason": "not-a-real-reason"}, schema)
