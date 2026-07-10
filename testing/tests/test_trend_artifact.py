"""Hermetic consumer test over the COMMITTED trend file: structure, provenance, ordering.

Reads `evalkit/artifacts/trend.jsonl` as any downstream consumer would: parse every line, check
the fields `EvalReport.as_dict()` promises are present, and confirm dates only ever move forward.
It never asserts freshness (see the note on `test_dates_parse_iso_and_are_non_decreasing`), and it
never regenerates rows: the file is a static fixture here, appended to only by `task eval`.
"""
from __future__ import annotations

from datetime import date

from evals.evalkit.report import TREND_PATH, read_trend_rows


def _rows() -> list[dict]:
    # consume the harness's own reader and public path, not a re-implemented parser
    return read_trend_rows(TREND_PATH)


def test_trend_file_exists_with_real_rows():
    assert TREND_PATH.exists()
    rows = _rows()
    assert len(rows) >= 1


def test_every_line_parses_as_one_json_object():
    # read_trend_rows parses every non-blank line; each parsed row must be a JSON object
    for row in _rows():
        assert isinstance(row, dict)


def test_every_row_carries_the_fields_as_dict_promises():
    for row in _rows():
        assert set(row) >= {"date", "provenance", "overall", "cases"}
        assert set(row["provenance"]) >= {"lane", "model_id"}
        assert set(row["overall"]) >= {"passes", "trials", "rate", "ci95"}


def test_every_row_is_stamped_with_a_known_lane_and_a_model_id():
    for row in _rows():
        assert row["provenance"]["lane"] in {"replay", "live"}
        assert row["provenance"]["model_id"]


def test_overall_rate_sits_inside_its_own_confidence_interval():
    for row in _rows():
        lo, hi = row["overall"]["ci95"]
        rate = row["overall"]["rate"]
        assert 0.0 <= lo <= rate <= hi <= 1.0


def test_case_rows_carry_their_fields_and_rate_sits_inside_its_interval():
    for row in _rows():
        assert row["cases"], "a trend row with zero cases means the suite ran nothing"
        for case in row["cases"]:
            assert {"id", "name", "risk", "passes", "k", "rate", "ci95"} <= set(case)
            lo, hi = case["ci95"]
            assert 0.0 <= lo <= case["rate"] <= hi <= 1.0


def test_dates_parse_iso_and_are_non_decreasing():
    # No freshness assertion here, on purpose. The hermetic suite is byte stable and CI runs it
    # twice and diffs the output, so a wall-clock staleness check on this committed file would
    # turn the same commit red after enough days pass. Staleness belongs to a scheduled workflow
    # reading this file over time, never to a test that must pass identically on every replay.
    dates = [date.fromisoformat(row["date"]) for row in _rows()]
    assert dates == sorted(dates)
