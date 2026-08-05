"""Where a measurement goes after it is taken.

Two files, because they answer different questions and have different lifetimes.

`data/results/runs.jsonl` is committed: one line per run, keyed by `run_id`. It is what
`just report --render` prints without touching a model, so a reader with no API key can
regenerate every published table.

`.cache/report/{run_id}.jsonl` is not: one line per question, the raw material a run was
summarised from, and the ledger that lets an interrupted run resume rather than pay
twice. It grows with every experiment and would make `git diff` on a run useless.

The append-only shape, the required path argument, and the schema gate that raises are
copied deliberately from `evals.labels`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from atlas.contracts import ChunkId, QuestionName
from evals.ir_metrics import Bounded

RESULTS_SCHEMA_VERSION = "1.0.0"
RUNS_PATH = "data/results/runs.jsonl"
LEDGER_DIR = ".cache/report"

# The two rankings that never read the question. Every published metric must carry a row
# for each: see MetricRow and RunRow.__post_init__.
NULL_SYSTEMS = ("null: random", "null: best constant")


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One metric, for one system, with the ceiling it was measured against.

    `ceiling` is not optional: a metric like recall@10 has a maximum below 1.0 whenever
    the average gold set exceeds k, and without it a correct low score reads as a broken
    retriever. See `evals.ir_metrics.Bounded`, of which this is the disk-persisted form.

    `questions` because not every metric speaks for the whole set: graded nDCG scores only
    22 of the 97 tau2 tasks.
    """

    system: str
    metric: str
    value: float
    ceiling: float
    questions: int

    @property
    def bounded(self) -> Bounded:
        return Bounded(self.value, self.ceiling)


@dataclass(frozen=True, slots=True)
class ContrastRow:
    """What one industry standard scorer said, and how far that is from this repository's
    own judge on the same answers.

    `threshold` travels with the score since the library's score is continuous and the
    judge's verdict is binary: any agreement figure is agreement *at some cut point*.

    `chance_corrected` is the one to read; raw agreement reads high for a scorer that
    rates everything above threshold.
    """

    metric: str
    mean: float
    threshold: float
    raw_agreement: float
    chance_corrected: float
    questions: int
    # The interval around `chance_corrected`, and whether it's wide enough that the point
    # estimate shouldn't be read alone. Optional rather than required because the
    # committed contrast row predates this field and can't be cheaply re-derived; None
    # means "predates the field", not "measured as zero". `evals.table` prints it as such.
    low: float | None = None
    high: float | None = None
    underpowered: bool | None = None
    # Same shape as low/high, but from agreement_with_reference_noise rather than plain
    # agreement: the interval this row's kappa would carry with the judge's own measured
    # self-disagreement (data/noise_floor.json) folded into the resampling. None for a row
    # recorded before this field existed or before a floor had been recorded.
    low_with_judge_noise: float | None = None
    high_with_judge_noise: float | None = None


@dataclass(frozen=True, slots=True)
class RunRow:
    """One run: what it was configured as, what it measured, what it cost.

    `settings` is the whole frozen tree, so a recorded row reproduces the `run_id` it was
    filed under. `commit` travels as provenance but is deliberately not part of the key:
    keying on a sha would split genuinely comparable runs on every unrelated commit.
    """

    schema: str
    run_id: str
    recorded: str  # ISO-8601 UTC, stamped by the caller
    commit: str
    settings: Mapping[str, Any]
    questions: int
    k: int
    generated: bool
    judged: bool
    metrics: tuple[MetricRow, ...]
    latency_ms: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    cost_usd: Mapping[str, float] = field(default_factory=dict)
    wall_seconds: float = 0.0
    # Empty unless --contrast ran. Not part of `metrics`, because the null baseline
    # invariant there is about retrieval rankings and these score written answers.
    contrast: tuple[ContrastRow, ...] = ()

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("a run row with no metrics is a run that measured nothing")
        # A metric published without a row saying what ignoring the question scores is
        # not a measurement; enforced here so a hand-edited store fails to load rather
        # than publish a number with no floor under it.
        if self.generated and not self.latency_ms:
            raise ValueError(
                "a run that generated answers recorded no stage timings at all, which no "
                "real run can do. This is the signature of a summary reading from a source "
                "the run did not execute against."
            )
        published = {row.metric for row in self.metrics}
        for null in NULL_SYSTEMS:
            covered = {row.metric for row in self.metrics if row.system == null}
            missing = sorted(published - covered)
            if missing:
                raise ValueError(
                    f"{missing} would be published with no {null!r} row. A number without "
                    "a baseline that ignores the question is not a measurement."
                )


def _metric_from(payload: Mapping[str, Any]) -> MetricRow:
    return MetricRow(**payload)


def _run_from(payload: Mapping[str, Any]) -> RunRow:
    """Rebuilds the tuples and mappings JSON gives back as lists and dicts.

    A RunRow holds a tuple of dataclasses, so a round trip through JSON returns something
    that compares unequal to what was written unless rebuilt here.
    """
    body = dict(payload)
    body["metrics"] = tuple(_metric_from(row) for row in body.get("metrics", ()))
    body["contrast"] = tuple(ContrastRow(**row) for row in body.get("contrast", ()))
    return RunRow(**body)


def append_run(path: str | Path, row: RunRow) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(row)) + "\n")


def read_runs(path: str | Path) -> tuple[RunRow, ...]:
    """Every recorded run, oldest first. An absent file is an honest empty answer.

    Raises on a row written under another schema version rather than reading it leniently.
    """
    target = Path(path)
    if not target.exists():
        return ()
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("schema") != RESULTS_SCHEMA_VERSION:
            raise ValueError(
                f"the run store holds schema {payload.get('schema')!r}, expected "
                f"{RESULTS_SCHEMA_VERSION!r}"
            )
        rows.append(_run_from(payload))
    return tuple(rows)


def latest_by_run(rows: Sequence[RunRow]) -> tuple[RunRow, ...]:
    """The last recorded row per distinct run, in the order each was first seen.

    The file is append-only and the same key may legitimately appear more than once, so
    the collapse happens on read rather than at write time, leaving the full history on
    disk.

    Keyed on more than `run_id`: `Settings.run_id` is a hash of the settings tree
    (docs/how-runs-are-named.md), but k, generated, and judged are argparse arguments that
    never reach it, so a retrieval-only run and a generate-and-judge run over identical
    settings would otherwise collide on one key.
    """
    latest: dict[tuple[str, int, bool, bool, bool], RunRow] = {}
    for row in rows:
        latest[run_key(row)] = row
    return tuple(latest.values())


def run_key(row: RunRow) -> tuple[str, int, bool, bool, bool]:
    """Everything that makes two rows different measurements rather than a repeat.

    `contrast` is in here because a `--contrast` run and a plain `--judge` run would
    otherwise compute the same key, discarding the more expensive run's own record.
    """
    return (row.run_id, row.k, row.generated, row.judged, bool(row.contrast))


@dataclass(frozen=True, slots=True)
class QuestionRow:
    """One question's raw result, before anything is averaged over it.

    What a resumed run reads to know what it has already paid for.
    """

    schema: str
    run_id: str
    question: QuestionName
    generated: bool
    judged: bool
    rankings: Mapping[str, tuple[ChunkId, ...]]
    correct: tuple[ChunkId, ...]
    primary: tuple[ChunkId, ...]
    timings_ms: Mapping[str, float]
    outcome: str | None = None
    # Stored so a resumed run's contrast can score answers generated in an earlier
    # process rather than producing nothing the second time it's asked for.
    answer_text: str = ""
    cited: tuple[ChunkId, ...] = ()
    shown: tuple[ChunkId, ...] = ()
    violations: tuple[str, ...] = ()
    answer_cost_usd: float = 0.0
    judge_verdict: str | None = None
    judge_cost_usd: float = 0.0
    # Carried so a score computed after the fact can still find the request it belongs to.
    trace_id: str | None = None
    # Defaulted rather than added behind a schema bump, so an old row on disk still loads
    # (an absent key reads as {}) instead of forcing every already-paid-for answer to be
    # re-bought just to stop overpaying for its scoring.
    deepeval_scores: Mapping[str, float] = field(default_factory=dict)


def ledger_path(run_id: str, directory: str | Path = LEDGER_DIR) -> Path:
    return Path(directory) / f"{run_id}.jsonl"


def append_question(path: str | Path, row: QuestionRow) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(row)) + "\n")


def read_questions(path: str | Path) -> tuple[QuestionRow, ...]:
    target = Path(path)
    if not target.exists():
        return ()
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("schema") != RESULTS_SCHEMA_VERSION:
            raise ValueError(
                f"the ledger holds schema {payload.get('schema')!r}, expected "
                f"{RESULTS_SCHEMA_VERSION!r}"
            )
        body = dict(payload)
        body["rankings"] = {k: tuple(v) for k, v in body.get("rankings", {}).items()}
        body["deepeval_scores"] = dict(body.get("deepeval_scores", {}))
        for name in ("correct", "primary", "cited", "shown", "violations"):
            body[name] = tuple(body.get(name, ()))
        rows.append(QuestionRow(**body))
    return tuple(rows)


def latest_questions(path: str | Path) -> dict[QuestionName, QuestionRow]:
    """The last recorded row per question: the ledger is append only, so re-recording a
    question adds a row rather than replacing one."""
    return {row.question: row for row in read_questions(path)}


def recorded_questions(
    path: str | Path, generated: bool, judged: bool
) -> frozenset[QuestionName]:
    """Which questions a resumed run may skip, at the grain it was asked for.

    Grain aware: a retrieval-only ledger records every question, so without checking
    `generated`/`judged` a later run asked to generate would skip everything and file a
    row claiming to have generated nothing new.
    """
    return frozenset(
        row.question
        for row in read_questions(path)
        if row.generated >= generated and row.judged >= judged
    )
