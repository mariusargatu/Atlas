"""The golden CSV loader: it ingests the intermediary CSV into loose drafts
and fails loud on a bad one.

The loader is the forgiving inbound door: a flat, SME editable CSV becomes ``GoldenDraft`` records
that ``enrich`` later hardens into typed ``GoldenCase``. These tests pin both halves: the real CSV
loads into well formed drafts, and every malformed shape (unknown account, empty turns, duplicate
id, missing column, missing file) raises rather than silently ingesting nothing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from evals.evalkit.golden_loader import load_golden_drafts

_GOLDEN = Path(__file__).resolve().parents[1] / "harness/evals/datasets/atlas_golden.csv"


def _write(tmp_path, text: str) -> Path:
    path = tmp_path / "set.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_real_csv_loads_well_formed():
    drafts = load_golden_drafts(_GOLDEN)
    assert len(drafts) == 10
    assert len({d.id for d in drafts}) == len(drafts)           # ids unique
    assert all(d.customer_id and d.turns and d.expected for d in drafts)


def test_multi_turn_cell_splits_on_separator():
    by_id = {d.id: d for d in load_golden_drafts(_GOLDEN)}
    assert by_id["plan-change-confirmed"].turns == ("switch me to the fast plan", "CONFIRM")
    assert by_id["cap-legacy-trap"].turns == ("is there a cap on my data?",)


def test_identical_utterance_distinguished_by_session():
    by_id = {d.id: d for d in load_golden_drafts(_GOLDEN)}
    trap, happy = by_id["cap-legacy-trap"], by_id["cap-current-happy"]
    assert trap.turns == happy.turns                            # same words
    assert trap.customer_id != happy.customer_id                # different session = different test


def test_unknown_customer_id_fails_loud(tmp_path):
    path = _write(tmp_path, "id,customer_id,turns,expected\nx,cust_nope,hi,ok\n")
    with pytest.raises(ValueError, match="not a seeded account"):
        load_golden_drafts(path)


def test_empty_turns_fails(tmp_path):
    path = _write(tmp_path, "id,customer_id,turns,expected\nx,cust_current,||,ok\n")
    with pytest.raises(ValueError, match="no turns"):
        load_golden_drafts(path)


def test_duplicate_id_fails(tmp_path):
    path = _write(
        tmp_path,
        "id,customer_id,turns,expected\nx,cust_current,hi,ok\nx,cust_current,bye,ok\n",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        load_golden_drafts(path)


def test_missing_id_fails(tmp_path):
    path = _write(tmp_path, "id,customer_id,turns,expected\n  ,cust_current,hi,ok\n")
    with pytest.raises(ValueError, match="missing id"):
        load_golden_drafts(path)


def test_missing_required_column_fails(tmp_path):
    path = _write(tmp_path, "id,customer_id,turns\nx,cust_current,hi\n")
    with pytest.raises(ValueError, match="missing required column"):
        load_golden_drafts(path)


def test_missing_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_golden_drafts(tmp_path / "nope.csv")
