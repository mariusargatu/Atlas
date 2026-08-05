"""The run store: the one thing standing between a measurement and a published table."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from atlas.config import Settings, load_settings
from atlas.contracts import ChunkId, QuestionName
from evals.results import (
    RESULTS_SCHEMA_VERSION,
    MetricRow,
    QuestionRow,
    RunRow,
    append_question,
    append_run,
    read_questions,
    read_runs,
    recorded_questions,
)
from evals.table import retrieval_table

METRICS = ("recall@k", "nDCG@k")
SYSTEMS = ("null: random", "null: best constant", "reranked")


def _row(**overrides) -> RunRow:
    metrics = tuple(
        MetricRow(system=system, metric=metric, value=0.2, ceiling=0.6, questions=97)
        for system in SYSTEMS
        for metric in METRICS
    )
    body = dict(
        schema=RESULTS_SCHEMA_VERSION, run_id="3f9c1a2b7d04",
        recorded="2026-08-02T00:00:00+00:00", commit="abc1234",
        settings={"chunk": {"max_tokens": 256}}, questions=97, k=10,
        generated=False, judged=False, metrics=metrics,
        latency_ms={"vector": {"median": 1.0, "slowest_tenth": 2.0, "mean": 1.2}},
        cost_usd={"embedding": 0.004}, wall_seconds=3.5,
    )
    body.update(overrides)
    return RunRow(**body)


def test_a_run_row_round_trips_through_the_file_on_disk(tmp_path) -> None:
    # Through the file, never through a dict: a RunRow holds a tuple of dataclasses, and
    # JSON gives those back as lists of dicts. A reader that forgets to rebuild them
    # renders every table from the wrong shape.
    path = tmp_path / "runs.jsonl"
    original = _row()
    append_run(path, original)
    assert read_runs(path) == (original,)


def test_the_store_refuses_a_file_written_under_another_schema_version(tmp_path) -> None:
    path = tmp_path / "runs.jsonl"
    append_run(path, _row())
    payload = json.loads(path.read_text(encoding="utf-8").strip())
    payload["schema"] = "9.9.9"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="9.9.9"):
        read_runs(path)


def test_resuming_skips_only_the_questions_already_recorded_at_the_requested_grain(
    tmp_path,
) -> None:
    # A retrieval only ledger records every question; without the grain check a later run
    # asked to generate would find them all present, skip every one, and file a row saying
    # it generated while generating nothing.
    ledger = tmp_path / "run.jsonl"
    for name in ("task_001", "task_002"):
        append_question(ledger, QuestionRow(
            schema=RESULTS_SCHEMA_VERSION, run_id="r", question=QuestionName(name),
            generated=False, judged=False, rankings={"fused": (ChunkId("d#0000"),)},
            correct=(ChunkId("d#0000"),), primary=(), timings_ms={"vector": 1.0},
        ))
    assert len(read_questions(ledger)) == 2
    assert recorded_questions(ledger, generated=False, judged=False) == {"task_001", "task_002"}
    assert recorded_questions(ledger, generated=True, judged=False) == frozenset()
    assert recorded_questions(ledger, generated=True, judged=True) == frozenset()


def test_a_run_cannot_be_recorded_with_a_metric_that_has_no_null_baseline() -> None:
    # The null baseline requirement enforced by the type rather than by discipline: a
    # number with no floor under it is not a measurement, so the object refuses to exist.
    complete = _row()
    assert complete.metrics

    without = tuple(m for m in complete.metrics if m.system != "null: best constant")
    with pytest.raises(ValueError, match="not a measurement"):
        replace(complete, metrics=without)

    with pytest.raises(ValueError, match="measured nothing"):
        replace(complete, metrics=())


def test_the_settings_a_row_recorded_reproduce_the_run_id_it_was_filed_under(
    tmp_path,
) -> None:
    # That the store did not drop, reorder or coerce a leaf on the way through JSON: a row
    # whose settings no longer hash to its own key describes a configuration nobody can
    # reconstruct.
    settings = Settings()
    path = tmp_path / "runs.jsonl"
    append_run(path, _row(run_id=settings.run_id, settings=json.loads(
        json.dumps(__import__("dataclasses").asdict(settings)))))
    recovered = read_runs(path)[0]

    written = tmp_path / "settings.json"
    written.write_text(json.dumps(recovered.settings), encoding="utf-8")
    assert load_settings(written).run_id == recovered.run_id


def test_the_published_table_carries_the_ceiling_and_every_null_row() -> None:
    # A table without a ceiling invites a correct 0.197 to be read as a broken retriever,
    # and one without a row for a ranking that ignores the question invites any number at
    # all to be read as a result.
    rendered = retrieval_table(_row())
    # Named for the arm it describes: a ceiling is per-arm the moment k exceeds what an
    # arm returns, so one unlabelled row is right for the headline and wrong for the rest.
    assert "**ceiling (reranked)**" in rendered and "0.600" in rendered
    assert "attained" in rendered
    for null in ("null: random", "null: best constant"):
        assert null in rendered, f"{null} is missing from a published table"
    assert "questions" in rendered
    assert "Smallest resolvable difference" in rendered
