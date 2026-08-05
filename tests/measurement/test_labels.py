from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from atlas.contracts import QuestionName
from evals.labels import LABEL_SCHEMA_VERSION, Label, append_label, read_labels

FIRST = Label(schema=LABEL_SCHEMA_VERSION, question=QuestionName("q0042"),
              rubric_version="1.1.0", verdict="fail", reason="price disagrees with the record",
              run_id="3f9c1a2b7d04", judge_verdict="pass", rater="human")


def test_labels_round_trip_through_the_file_on_disk(tmp_path):
    # Through the file, not through a dictionary: the failure this guards is the format.
    path = tmp_path / "labels.jsonl"
    append_label(path, FIRST)
    assert read_labels(path) == (FIRST,)
    with pytest.raises(FrozenInstanceError):
        read_labels(path)[0].verdict = "pass"


def test_appending_never_rewrites_an_existing_line(tmp_path):
    # Human labels cannot be regenerated at any price, so a writer that rebuilds the
    # file rather than appending can lose all of them.
    path = tmp_path / "labels.jsonl"
    append_label(path, FIRST)
    before = path.read_text()
    append_label(path, replace(FIRST, question=QuestionName("q0043")))
    assert path.read_text().startswith(before) and len(read_labels(path)) == 2


def test_an_unknown_schema_version_is_refused(tmp_path):
    # The bad line is a complete, valid label with exactly one field changed, so the
    # rejection can only come from the schema comparison and not from missing fields.
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(asdict(replace(FIRST, schema="9.9.9"))) + "\n")
    with pytest.raises(ValueError, match="9.9.9"):
        read_labels(path)


def test_reading_a_store_that_does_not_exist_yet_gives_an_empty_history(tmp_path):
    # The first annotation session reads before it writes, so raising here would make
    # the very first label impossible to add.
    assert read_labels(tmp_path / "labels.jsonl") == ()
