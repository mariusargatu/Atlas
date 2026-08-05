"""Walking the question set through the pipeline, once, and writing the ledger.

There is one implementation of "ask a question and record it" here, and two drivers
over it: the plain loop and the Langfuse dataset run.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from atlas.config import Settings
from atlas.contracts import Answer, ChunkId, Grader, JudgeRecord, Question, Record
from atlas.corpus.gold import GoldIndex
from atlas.pipeline import PreparedCorpus
from evals.baselines import null_rankings
from evals.judge import JudgeModel, judge
from evals.report import rankings
from evals.results import RESULTS_SCHEMA_VERSION, QuestionRow, append_question, recorded_questions


def _grader(settings: Settings) -> Grader:
    client = JudgeModel(settings.judge)

    def grade(
        question: Question, answer: Answer, texts: Mapping[ChunkId, str]
    ) -> JudgeRecord:
        return judge(question, answer, client, settings, texts)

    return grade


def _ask_and_record(
    settings: Settings,
    args: argparse.Namespace,
    prepared: PreparedCorpus,
    question: Question,
    gold: GoldIndex,
    grader: Grader | None,
    ledger: Path,
    records: list[Record],
) -> QuestionRow:
    """One question through every stage, appended to the ledger, returned as a row.

    Shared by the Langfuse dataset driver and the plain loop.
    """
    record = prepared.ask(question, generate=args.generate, grader=grader)
    records.append(record)
    correct = gold.correct(question)
    answer = record.answer
    row = QuestionRow(
        schema=RESULTS_SCHEMA_VERSION, run_id=settings.run_id, question=question.name,
        generated=args.generate, judged=args.judge,
        rankings=rankings(record), correct=tuple(sorted(correct.chunks)),
        primary=tuple(sorted(correct.primary)),
        timings_ms={t.stage: t.wall_ms for t in record.timings},
        outcome=answer.outcome if answer else None,
        cited=answer.cited if answer else (), shown=answer.shown if answer else (),
        violations=answer.violations if answer else (),
        answer_text=answer.text if answer else "",
        answer_cost_usd=answer.usage.cost_usd if answer else 0.0,
        judge_verdict=getattr(record.judge, "verdict", None),
        judge_cost_usd=record.judge.usage.cost_usd if record.judge else 0.0,
        trace_id=record.trace_id,
    )
    append_question(ledger, row)
    return row


def _collect(
    settings: Settings,
    args: argparse.Namespace,
    prepared: PreparedCorpus,
    questions: Sequence[Question],
    ledger: Path,
    all_questions: Sequence[Question],
) -> list[Record]:
    already = recorded_questions(ledger, generated=args.generate, judged=args.judge)
    gold = GoldIndex.build(prepared.chunks)
    grader = _grader(settings) if args.judge else None
    records: list[Record] = []
    pending = [q for q in questions if q.name not in already]
    reused = len(questions) - len(pending)
    if reused:
        print(f"resuming: {reused} of {len(questions)} questions already recorded")

    if args.dataset:
        if not pending:
            print("dataset run skipped: every question was already in the ledger "
                  "(`--fresh` re-runs them)")
            return records
        _dataset_run(settings, args, prepared, all_questions, pending, gold, grader, ledger, records)
        return records

    for index, question in enumerate(pending, start=1):
        _ask_and_record(settings, args, prepared, question, gold, grader, ledger, records)
        if index % 20 == 0:
            print(f"  {index}/{len(pending)} questions", flush=True)
    return records


def _dataset_run(
    settings: Settings,
    args: argparse.Namespace,
    prepared: PreparedCorpus,
    questions: Sequence[Question],
    pending: Sequence[Question],
    gold: GoldIndex,
    grader: Grader | None,
    ledger: Path,
    records: list[Record],
) -> None:
    """Run the question set as a Langfuse dataset run, keyed by `run_id`.

    Deliberately duplicates `runs.jsonl`. See docs/recording-runs.md.
    """
    from langfuse import Langfuse

    from evals.langfuse_dataset import run_experiment

    # A partial run gets a name that says so, so it can't be mistaken for a full
    # one in the comparison view. The count is of items actually in `pending`,
    # not of what was asked for, since that's the denominator of every figure inside.
    run_name = (settings.run_id if len(pending) == len(questions)
                else f"{settings.run_id}-{len(pending)}of{len(questions)}")
    result = run_experiment(
        client=Langfuse(),
        questions=questions,
        pending=pending,
        ask=lambda q: _ask_and_record(settings, args, prepared, q, gold, grader, ledger, records),
        run_name=run_name,
        description=f"{settings.embedding.model_name} / {settings.sparse.scorer}, k={args.k}",
        metadata={"run_id": settings.run_id, "k": str(args.k),
                  "generated": str(args.generate), "judged": str(args.judge)},
        k=args.k,
        nulls=null_rankings(prepared.chunks, tuple(pending), args.k),
    )
    print(result.format())
