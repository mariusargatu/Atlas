"""The judge-artifact staleness check, tested as a pure function (the wall clock is DATA here).

The freshness logic used to live untested in the staleness GitHub workflow's bash. Extracting it to
`evals.staleness` makes it suite-testable: `today` is passed in, so a fixed date pins every case, and
the four behaviours the workflow relied on (fresh passes, stale fails, empty fails distinctly,
malformed ignored) are asserted here instead of only ever observed in CI.
"""
from __future__ import annotations

from datetime import date

from evals.staleness import check_staleness, main

_TODAY = date(2026, 7, 17)


def test_a_fresh_artifact_passes():
    verdict = check_staleness(["2026-07-10.md", "2026-06-01.md", "latest.md"], _TODAY, 45)
    assert verdict.ok
    assert verdict.newest == "2026-07-10.md" and verdict.age_days == 7  # the newest dated file, its age


def test_a_stale_artifact_fails():
    verdict = check_staleness(["2026-05-01.md", "latest.md"], _TODAY, 45)
    assert not verdict.ok and "STALE" in verdict.message and verdict.age_days > 45


def test_no_dated_artifact_fails_with_a_distinct_message():
    # an empty/pointer-only directory is a DIFFERENT failure from a stale one, and says so
    verdict = check_staleness(["latest.md", "README.md"], _TODAY, 45)
    assert not verdict.ok and "no dated calibration artifact" in verdict.message


def test_malformed_filenames_are_ignored_and_the_real_newest_wins():
    # a non-date name and a shape-but-not-a-real-date (month 13) are both skipped, not errors
    verdict = check_staleness(["notes.md", "2026-13-40.md", "2026-07-14.md"], _TODAY, 45)
    assert verdict.ok and verdict.newest == "2026-07-14.md"


def test_exactly_at_the_threshold_is_not_stale():
    # strict '>': an artifact exactly max_age_days old still passes
    verdict = check_staleness(["2026-06-02.md"], _TODAY, 45)  # 45 days old
    assert verdict.age_days == 45 and verdict.ok


def test_cli_over_a_real_directory_returns_zero_when_fresh(tmp_path, capsys):
    (tmp_path / "2026-07-10.md").write_text("x")
    (tmp_path / "latest.md").write_text("x")
    rc = main([str(tmp_path), "2026-07-17", "45"])
    assert rc == 0 and "OK" in capsys.readouterr().out


def test_cli_on_a_missing_directory_returns_one_with_the_distinct_message(tmp_path, capsys):
    rc = main([str(tmp_path / "nope"), "2026-07-17", "45"])
    assert rc == 1 and "no dated calibration artifact" in capsys.readouterr().out
