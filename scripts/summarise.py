"""Turning a ledger into the row a chapter quotes, and the deepeval contrast beside it.

Everything that reads recorded questions back and turns them into published numbers
is here, including the deepeval contrast, the single most expensive thing this
repository can do.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from atlas.config import Settings
from atlas.contracts import Chunk, ChunkId, Question, QuestionName, Record
from atlas.corpus.gold import Correct, resolve
from atlas.pipeline import PreparedCorpus
from atlas.trace import score
from evals.baselines import null_rankings
from evals.calibration import NOISE_FLOOR_PATH, recorded_noise_floor
from evals.ir_metrics import Bounded
from evals.report import (
    cost_from_ledger,
    latency_from_ledger,
    metric_rows,
    retrieval_summary,
    scored_retrieval,
)
from evals.results import (
    RESULTS_SCHEMA_VERSION,
    ContrastRow,
    MetricRow,
    QuestionRow,
    RunRow,
    append_question,
)

SYSTEMS = ("vector", "keyword", "fused", "reranked")
ROOT = Path(__file__).resolve().parents[1]


def _commit() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=False, cwd=ROOT)
    return result.stdout.strip() or "unknown"

def _scored_over(
    rows: Sequence[QuestionRow],
    ranked_for: Mapping[QuestionName, tuple[ChunkId, ...]],
    gold: Mapping[QuestionName, Correct],
    k: int,
) -> dict[str, list[Bounded]]:
    scored: dict[str, list[Bounded]] = {}
    for row in rows:
        for metric, bounded in scored_retrieval(ranked_for[row.question], gold[row.question], k).items():
            scored.setdefault(metric, []).append(bounded)
    return scored


def scoreable_traces(
    rows: Sequence[QuestionRow], minted_here: frozenset[str]
) -> tuple[dict[QuestionName, str], int]:
    """Which questions' traces this process may attach a score to, and how many it declined.

    A trace id is a key into whichever Langfuse instance minted it, and `.cache/report`
    outlives that instance: `create_score` against an id no run ever opened is accepted
    and queryable, silently orphaned from any observation. So a score goes only to a
    trace this process opened, the one thing knowable without a lookup that can itself
    go stale.
    """
    traced = {row.question: row.trace_id for row in rows
              if row.trace_id and row.trace_id in minted_here}
    declined = sum(1 for row in rows if row.trace_id and row.trace_id not in minted_here)
    return traced, declined


def _contrast_rows(
    rows: Sequence[QuestionRow], prepared: PreparedCorpus, questions: Sequence[Question],
    minted_here: frozenset[str] = frozenset(), ledger: Path | None = None,
    settings: Settings | None = None,
) -> tuple[ContrastRow, ...]:
    """What the standard industry scorers say about the same answers, beside the judge.

    A second opinion rather than the headline: the number worth reading is the
    chance-corrected agreement between the two, not either score alone.

    Four scorers over every answered question, each at least one model call, so this
    costs several times what the judging run it's compared against costs.
    """
    # Imported locally: deepeval is slow to import and --contrast is off by default,
    # so every other path through this module would otherwise pay that cost.
    from deepeval.test_case import LLMTestCase

    from evals.deepeval_suite import (
        contrast_row,
        contrast_scorers,
        score_contrast,
        scorer_thresholds,
    )

    # None until `just noise-floor` has run; contrast_row handles that as the honest
    # no-floor-yet state.
    judge_flip_floor = (
        recorded_noise_floor(ROOT / NOISE_FLOOR_PATH).at_configured.flip_floor
        if (ROOT / NOISE_FLOOR_PATH).exists() else None
    )

    by_name = {q.name: q for q in questions}

    # Last row per question: persisting scores appends an updated row rather than
    # rewriting, so the same question can legitimately appear twice.
    latest: dict[QuestionName, QuestionRow] = {}
    for row in rows:
        if row.question in by_name:
            latest[row.question] = row
    rows = tuple(latest.values())
    wanted = [row for row in rows if row.answer_text and row.judge_verdict]
    # contrast_scorers() falls back to Settings() when unconfigured, so pass `settings`
    # explicitly: a --settings run naming a different judge model must not score the
    # contrast against the default one while everything else records the configured one.
    scorers = contrast_scorers(settings)
    thresholds = scorer_thresholds(scorers)
    asked_for = set(thresholds)
    verdicts_by_name = {row.question: row.judge_verdict for row in wanted
                        if row.judge_verdict is not None}
    # Banked per question and per scorer, so a partly banked ledger buys only what's
    # missing rather than re-measuring everything because one score is absent.
    banked: dict[str, dict[QuestionName, float]] = {}
    for row in wanted:
        for metric, value in row.deepeval_scores.items():
            if metric in asked_for:
                banked.setdefault(metric, {})[row.question] = value
    unbanked = [row for row in wanted
                if any(row.question not in banked.get(m, {}) for m in asked_for)]
    if wanted and not unbanked:
        print(f"  contrast: reusing recorded scores for all {len(asked_for)} scorers over "
              f"{len(wanted)} answers; nothing re-measured (`--fresh` re-scores)")
        rebuilt = tuple(
            built for metric, scores in sorted(banked.items())
            if (built := contrast_row(metric, scores, verdicts_by_name, thresholds[metric],
                                      judge_flip_floor=judge_flip_floor)) is not None
        )
        if rebuilt:
            return rebuilt
    if banked and unbanked:
        print(f"  contrast: {len(wanted) - len(unbanked)} of {len(wanted)} answers already "
              f"scored; measuring the remaining {len(unbanked)}")
    to_measure = {row.question for row in unbanked}
    cases: dict[QuestionName, LLMTestCase] = {}
    verdicts: dict[QuestionName, str] = {}
    for row in rows:
        # The ledger can hold rows outside this run's question set (e.g. `--limit 3`
        # against a ledger a full run left), so guard the lookup rather than index it.
        if row.question not in by_name:
            continue
        if not row.answer_text or row.judge_verdict is None:
            continue
        if row.question not in to_measure:
            continue
        cases[row.question] = LLMTestCase(
            input=by_name[row.question].text,
            actual_output=row.answer_text,
            # What the writer was shown, never what it claimed to cite: those differ
            # exactly when a citation is fabricated, and handing a faithfulness scorer the
            # answer's own account of its evidence is how a fabricating system scores well.
            retrieval_context=[prepared.by_name[name].text for name in row.shown],
        )
        verdicts[row.question] = row.judge_verdict
    if not cases:
        return ()

    traced, unattached = scoreable_traces(
        [row for row in rows if row.question in by_name], minted_here)
    if unattached:
        print(f"  {unattached} question(s) were resumed, so their deepeval scores stay in "
              "the table and are not attached to a trace. `--fresh` re-traces them.")

    def attach(metric: str, scores: Mapping[QuestionName, float]) -> None:
        for name, value in scores.items():
            trace_id = traced.get(name)
            if trace_id and value == value:  # NaN means the scorer produced nothing
                score(trace_id, f"deepeval: {metric}", value)

    collected: dict[QuestionName, dict[str, float]] = {}

    def keep(metric: str, scores: Mapping[QuestionName, float]) -> None:
        attach(metric, scores)
        for name, value in scores.items():
            if value == value:  # never persist a not-a-number as a measurement
                collected.setdefault(name, {})[metric] = value

    score_contrast(cases, verdicts, scorers=scorers, on_scores=keep)
    for name, scores in collected.items():
        for metric, value in scores.items():
            banked.setdefault(metric, {})[name] = value
    result = tuple(
        built for metric, scores in sorted(banked.items())
        if (built := contrast_row(metric, scores, verdicts_by_name, thresholds[metric],
                                  judge_flip_floor=judge_flip_floor)) is not None
    )
    if ledger is not None and collected:
        by_question = {row.question: row for row in rows}
        for name, scores in collected.items():
            if name in by_question:
                # Merged, not replaced: `scores` holds only what this run measured, and
                # writing it alone would drop every previously banked scorer for the row.
                merged = {**by_question[name].deepeval_scores, **scores}
                append_question(ledger, replace(by_question[name], deepeval_scores=merged))
    return result


def _summarise(
    settings: Settings,
    args: argparse.Namespace,
    rows: Sequence[QuestionRow],
    questions: Sequence[Question],
    chunks: tuple[Chunk, ...],
    records: Sequence[Record],
    embedding_cost: float,
    wall: float,
    contrast: tuple[ContrastRow, ...] = (),
) -> RunRow:
    gold = {q.name: resolve(q, chunks) for q in questions}
    # A ledger can hold rows this summary isn't about (a --limit slice reading a fuller
    # ledger, or a re-run at a higher grain). Rankings can come from any row at least as
    # rich as this run asked for (`>=`), but cost and timings must match the exact grain:
    # taking the richest row there would report an answer/judge cost for a retrieval-only
    # run that spent neither.
    in_scope: dict[QuestionName, QuestionRow] = {}
    for row in rows:
        if row.question in gold and row.generated >= args.generate and row.judged >= args.judge:
            in_scope[row.question] = row
    rows = tuple(in_scope.values())

    metrics: list[MetricRow] = []
    for system in SYSTEMS:
        ranked_for = {row.question: row.rankings.get(system, ()) for row in rows}
        metrics += metric_rows(system, retrieval_summary(_scored_over(rows, ranked_for, gold, args.k)))

    # Both baselines, over the same questions and the same k, so every metric published
    # above has the two rows RunRow refuses to be constructed without.
    for name, ranked_for in null_rankings(chunks, tuple(questions), args.k).items():
        metrics += metric_rows(name, retrieval_summary(_scored_over(rows, ranked_for, gold, args.k)))

    # Cost and timing both read from the ledger (`rows`), not from the in-process
    # `records`: a resumed run's `records` can be empty while the ledger holds the real
    # spend, and reading from `records` produced a canonical row claiming $0.00 in 0.3s
    # over what was actually a $1.41, 808s run. Timings come from every row the metric
    # table covers rather than the exact-grain subset, since a retrieval stage is timed
    # the same regardless of whether the run went on to answer; the stages this grain
    # didn't run are masked out below instead.
    skipped = set()
    if not args.generate:
        skipped.add("answer")
    if not args.judge:
        skipped.add("judge")
    timed = tuple(
        replace(
            row,
            timings_ms={s: ms for s, ms in row.timings_ms.items() if s not in skipped},
            answer_cost_usd=0.0 if "answer" in skipped else row.answer_cost_usd,
            judge_cost_usd=0.0 if "judge" in skipped else row.judge_cost_usd,
        )
        for row in rows
    )
    costs = cost_from_ledger(timed)
    return RunRow(
        schema=RESULTS_SCHEMA_VERSION, run_id=settings.run_id,
        recorded=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        commit=_commit(), settings=asdict(settings), questions=len(rows), k=args.k,
        generated=args.generate, judged=args.judge, metrics=tuple(metrics),
        latency_ms={
            row.stage: {"median": row.median_ms, "slowest_tenth": row.slowest_tenth_ms,
                        "mean": row.mean_ms}
            for row in latency_from_ledger(timed)
        },
        cost_usd={
            "embedding": embedding_cost,
            "answer": costs.by_stage["answer"],
            "judge": costs.by_stage["judge"],
        },
        # Summed per-question stage time, not the in-process wall clock: it stays
        # comparable between a fresh run and one resumed from a mostly-full ledger.
        wall_seconds=sum(sum(r.timings_ms.values()) for r in timed) / 1000.0 or wall,
        contrast=contrast,
    )
