"""The append only label JSONL writer (SP8 Task 4, label collection half, pulled early).

`LabelStore` is the local path standing in for the S3 label prefix D30 names as the eventual
system of record (real S3/R2 sync is late binding, never a prerequisite, per the plan's Global
Constraints). Every write goes through `determinism.canonical.canonical_json`, the SAME
canonicalization the cassette key and run digest already use, so two writers fed the identical
sequence of calls under the identical frozen clock produce byte identical files -- the "byte
reproducible" requirement the plan's Global Constraints binds every label artifact to.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from determinism.sources import FrozenClock

from atlas.adapters.label_store import LabelStore


def _clock(instant: str = "2026-06-15T12:00:00+00:00") -> FrozenClock:
    return FrozenClock(datetime.fromisoformat(instant))


def test_append_writes_one_json_line_with_created_at_from_the_injected_clock(tmp_path):
    path = tmp_path / "labels.jsonl"
    store = LabelStore(path, _clock())
    record = store.append(trace_id="t1", role="adjudicator", verdict="pass", critique="Fully grounded in the cited page.")

    assert record.created_at == "2026-06-15T12:00:00+00:00"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_append_is_append_only_never_rewrites_earlier_lines(tmp_path):
    path = tmp_path / "labels.jsonl"
    store = LabelStore(path, _clock())
    store.append(trace_id="t1", role="adjudicator", verdict="pass", critique="Grounded, matches the cited page.")
    store.append(trace_id="t2", role="adjudicator", verdict="fail", critique="Unsupported claim about the fee.")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"trace_id":"t1"' in lines[0]
    assert '"trace_id":"t2"' in lines[1]


def test_append_is_byte_reproducible_under_the_frozen_clock(tmp_path):
    path_a = tmp_path / "a" / "labels.jsonl"
    store_a = LabelStore(path_a, _clock())
    store_a.append(trace_id="t1", role="adjudicator", verdict="pass", critique="Grounded, matches the cited page.")
    store_a.append(trace_id="t2", role="adjudicator", verdict="fail", critique="Unsupported claim about the fee.")

    path_b = tmp_path / "b" / "labels.jsonl"
    store_b = LabelStore(path_b, _clock())
    store_b.append(trace_id="t1", role="adjudicator", verdict="pass", critique="Grounded, matches the cited page.")
    store_b.append(trace_id="t2", role="adjudicator", verdict="fail", critique="Unsupported claim about the fee.")

    assert path_a.read_bytes() == path_b.read_bytes()


def test_append_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "deep" / "labels.jsonl"
    store = LabelStore(path, _clock())
    store.append(trace_id="t1", role="adjudicator", verdict="pass", critique="Grounded, matches the cited page.")
    assert path.is_file()


def test_role_field_accepts_adjudicator_and_end_user(tmp_path):
    path = tmp_path / "labels.jsonl"
    store = LabelStore(path, _clock())
    a = store.append(trace_id="t1", role="adjudicator", verdict="pass", critique="Grounded, matches the cited page.")
    b = store.append(trace_id="t2", role="end_user", verdict="fail", critique="This did not answer my question.")
    assert a.role == "adjudicator"
    assert b.role == "end_user"


def test_unknown_role_is_rejected(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl", _clock())
    with pytest.raises(ValueError, match="role"):
        store.append(trace_id="t1", role="supervisor", verdict="pass", critique="not a real role")


def test_unknown_verdict_is_rejected(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl", _clock())
    with pytest.raises(ValueError, match="verdict"):
        store.append(trace_id="t1", role="adjudicator", verdict="maybe", critique="not pass or fail")


def test_empty_critique_is_rejected_a_one_sentence_critique_is_required(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl", _clock())
    with pytest.raises(ValueError, match="critique"):
        store.append(trace_id="t1", role="adjudicator", verdict="pass", critique="   ")


def test_missing_trace_id_is_rejected(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl", _clock())
    with pytest.raises(ValueError, match="trace_id"):
        store.append(trace_id="", role="adjudicator", verdict="pass", critique="fine")


def test_read_all_round_trips_every_appended_record(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl", _clock())
    store.append(trace_id="t1", role="adjudicator", verdict="pass", critique="Grounded, matches the cited page.")
    store.append(trace_id="t2", role="end_user", verdict="fail", critique="Wrong answer for my plan.")

    records = store.read_all()
    assert [r.trace_id for r in records] == ["t1", "t2"]
    assert [r.role for r in records] == ["adjudicator", "end_user"]


def test_read_all_on_a_missing_file_is_an_empty_list(tmp_path):
    store = LabelStore(tmp_path / "nope" / "labels.jsonl", _clock())
    assert store.read_all() == []


def test_labeled_trace_ids_filters_by_role(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl", _clock())
    store.append(trace_id="t1", role="adjudicator", verdict="pass", critique="Grounded, matches the cited page.")
    store.append(trace_id="t2", role="end_user", verdict="fail", critique="Wrong answer for my plan.")

    assert store.labeled_trace_ids(role="adjudicator") == {"t1"}
    assert store.labeled_trace_ids(role="end_user") == {"t2"}


def test_labeled_trace_ids_dedupes_a_trace_labeled_more_than_once(tmp_path):
    store = LabelStore(tmp_path / "labels.jsonl", _clock())
    store.append(trace_id="t1", role="adjudicator", verdict="pass", critique="First pass.")
    store.append(trace_id="t1", role="adjudicator", verdict="fail", critique="On review, actually ungrounded.")
    assert store.labeled_trace_ids(role="adjudicator") == {"t1"}
    # append only: BOTH lines survive on disk, even though the trace counts once toward progress
    assert len(store.read_all()) == 2
