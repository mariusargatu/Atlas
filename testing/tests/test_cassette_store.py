"""Cassette store behaviour: both adapters, including the cases mutation testing exposed.

The in memory store must KEEP every cassette (an earlier 'replace the dict' bug would have served
the wrong recording the moment a second cassette was saved); the file store must name the file when
a recording is corrupt, because 'one bad file among hundreds' is the painful debugging case.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from cassette import Cassette, build_request
from cassette_store import FileCassetteStore, InMemoryCassetteStore


def _cassette(text: str, content: str) -> Cassette:
    msgs = [HumanMessage(text)]
    return Cassette(model_id="m", request=build_request("m", msgs), response={"content": content, "tool_calls": []})


def test_inmemory_store_keeps_multiple_distinct_cassettes():
    store = InMemoryCassetteStore()
    a, b = _cassette("q-a", "ans-a"), _cassette("q-b", "ans-b")
    store.save(a)
    store.save(b)  # must NOT clobber the first
    assert store.load(a.key).response["content"] == "ans-a"
    assert store.load(b.key).response["content"] == "ans-b"


def test_inmemory_miss_returns_none():
    assert InMemoryCassetteStore().load("nope") is None


def test_file_store_round_trips_through_disk(tmp_path):
    store = FileCassetteStore(tmp_path)
    c = _cassette("q", "ans")
    store.save(c)
    loaded = store.load(c.key)
    assert loaded is not None and loaded.response["content"] == "ans"
    assert loaded.key == c.key  # content addressed: round trip preserves the key


def test_file_store_miss_returns_none(tmp_path):
    assert FileCassetteStore(tmp_path).load("absent") is None


def test_corrupt_cassette_error_names_the_file(tmp_path):
    (tmp_path / "deadbeef.json").write_text("{ not valid json")
    with pytest.raises(ValueError, match="deadbeef.json"):
        FileCassetteStore(tmp_path).load("deadbeef")
