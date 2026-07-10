"""The shared write and report epilogue: writes each artifact and echoes where it landed."""
from __future__ import annotations

from evals.artifacts import write_artifacts


def test_write_artifacts_writes_every_file(tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "nested" / "b.json"
    write_artifacts([(a, "content a"), (b, "content b")], echo="content a")
    assert a.read_text() == "content a"
    assert b.read_text() == "content b"


def test_write_artifacts_echoes_and_reports_every_path(tmp_path, capsys):
    a, b = tmp_path / "a.md", tmp_path / "b.json"
    write_artifacts([(a, "content a"), (b, "content b")], echo="content a")
    out = capsys.readouterr().out
    assert "content a" in out
    assert str(a) in out
    assert str(b) in out
