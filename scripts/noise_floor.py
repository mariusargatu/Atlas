"""Measure how often the judge disagrees with itself, and record it.

    just noise-floor --limit 20      # start here; ten verdicts a question
    just noise-floor                 # every answer in the current run

Costs money: a floor is `repeats` verdicts per question in each of two columns (ten
verdicts a question at the default `judge.repeats = 5`), roughly $0.055/question at the
recorded rate. `--limit` is worth using first. Reads answers back from a ledger under
`.cache/report/` rather than re-generating them, since only the judge is being measured.

The two columns are `randomness` and zero, run separately rather than copied: zero is
not determinism for a hosted model, and under MODEL_PROVIDER=anthropic the client
cannot send a temperature at all, so the run refuses rather than measure the same
request twice under two labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from atlas.config import Settings
from atlas.contracts import ZERO_USAGE, Answer, ChunkId, Outcome, Question
from atlas.corpus.chunk import cut
from atlas.corpus.registry import get_source
from atlas.models.providers import selected_provider
from evals.calibration import NOISE_FLOOR_PATH, noise_floor, record_noise_floor
from evals.judge import JudgeModel, judge
from evals.results import QuestionRow, latest_questions, ledger_path

ROOT = Path(__file__).resolve().parents[1]


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-judge the current run's recorded answers several times over and "
                    "record how often the judge disagrees with itself.",
        epilog="just noise-floor --limit 20   (ten verdicts a question, about $0.055 each)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="first N answered questions; ten verdicts each")
    parser.add_argument("--repeats", type=int, default=None,
                        help="override judge.repeats for this measurement")
    parser.add_argument("--randomness", type=float, default=0.7,
                        help="the configured column's randomness; the other column is always 0.0")
    parser.add_argument("--out", default=NOISE_FLOOR_PATH, help="where to write the floor")
    return parser.parse_args()


def _answer_from(row: QuestionRow) -> Answer:
    """The recorded answer, rebuilt. `model` and `usage` are filled with placeholders
    since the judge grades text, citations and outcome and never reads those two fields."""
    outcome = row.outcome or "answered"
    if outcome not in ("answered", "refused", "unknown", "not_applicable"):
        raise ValueError(
            f"{row.question} records outcome {outcome!r}, which is not one the pipeline "
            "can produce; the ledger is corrupt rather than this answer being unusual"
        )
    return Answer(
        question=row.question, text=row.answer_text, cited=tuple(row.cited),
        outcome=cast("Outcome", outcome), shown=tuple(row.shown),
        model="from-ledger", usage=ZERO_USAGE, violations=tuple(row.violations),
    )


def main() -> int:
    args = _parse()
    settings = Settings()
    if selected_provider() == "anthropic":
        print(
            "MODEL_PROVIDER=anthropic cannot measure a noise floor: that client sends no "
            "temperature, so the configured column and the zero column would be identical "
            "requests recorded under different labels. Set MODEL_PROVIDER=openai."
        )
        return 1

    # Always the run the current settings describe: every shipped configuration carries
    # an identical `judge` block, so a run selector could only change which answers get
    # re-judged, never which judge does it.
    run_id = settings.run_id
    ledger = ledger_path(run_id, ROOT / ".cache/report")
    latest = latest_questions(ledger)
    if not latest:
        print(f"no judged run at {ledger}. `just report --generate --judge` writes one for "
              f"the current settings, which is the only run a floor is measured against.")
        return 1

    answered = [r for r in latest.values() if r.answer_text]
    if args.limit:
        answered = answered[: args.limit]
    if not answered:
        print(f"{ledger} holds no answers to re-judge. A floor needs a judged run to read.")
        return 1

    source = get_source()
    texts: dict[ChunkId, str] = {
        c.name: c.text for c in cut(source(0).documents, settings.chunk)
    }
    by_name = {q.name: q for q in source.questions()}

    rows_by_question = {r.question: r for r in answered}
    questions = [by_name[r.question] for r in answered if r.question in by_name]
    repeats = args.repeats if args.repeats is not None else settings.judge.repeats

    # One client per randomness value, not one per call: `noise_floor` calls this
    # repeats * len(questions) times per column.
    clients: dict[float, JudgeModel] = {}

    def judge_fn(question: Question, randomness: float) -> str:
        client = clients.get(randomness)
        if client is None:
            client = clients[randomness] = JudgeModel(settings.judge, randomness)
        answer = _answer_from(rows_by_question[question.name])
        return judge(question, answer, client, settings, texts).verdict

    verdicts = repeats * len(questions) * 2
    print(f"re-judging {len(questions)} recorded answers, {repeats} repeats in each of two "
          f"columns: {verdicts} verdicts, roughly ${verdicts * 0.0055:.2f}")

    # Probe with one verdict before spending on the other 199: catches a model that
    # silently drops `temperature`, which would make the two columns the same request
    # twice and record a floor that says the judge never disagrees with itself.
    if args.randomness != 0.0:
        probe = JudgeModel(settings.judge, args.randomness)
        clients[args.randomness] = probe
        judge_fn(questions[0], args.randomness)
        if probe.randomness_applied is False:
            print(
                f"\nNOT RECORDED, and nothing further was spent. {settings.judge.model!r} "
                f"rejects the sampling temperature, so the client retried without it: the "
                f"{args.randomness} column and the 0.0 column would be the same request, and "
                f"a floor measured that way says the judge never disagrees with itself. That "
                f"would set every threshold in tests/measurement/test_judge.py to zero.\n\n"
                f"This model has one sampling setting, so measure the floor at it:\n"
                f"    just noise-floor --randomness 0.0\n"
                f"Both columns are then the provider's default by intent rather than by "
                f"accident, and repeat-to-repeat variation is still real -- a hosted model is "
                f"not deterministic at a fixed setting, which is the thing being measured."
            )
            return 1

    floor = noise_floor(questions, judge_fn, repeats=repeats, randomness=args.randomness)

    record_noise_floor(floor, args.out)
    if args.randomness == 0.0:
        print("\nBoth columns ran at 0.0, so this is one setting measured twice rather than "
              "two settings compared. That is the honest shape for a model with a single "
              "sampling setting; the flip floor below is still a real measurement, because a "
              "hosted model does not repeat itself exactly even at a fixed setting.")

    for label, column in (("configured", floor.at_configured), ("zero", floor.at_zero)):
        print(f"  {label:10} randomness={column.randomness}  "
              f"per-question self-agreement {column.per_question_self_agreement:.3f}  "
              f"flip floor {column.flip_floor:.3f}  spread {column.spread:.3f}")

    column = floor.at_configured
    unstable = column.unstable
    total = len(column.disagreement)
    if unstable:
        worst = column.disagreement[unstable[0]]
        concentrated = len(unstable) <= max(1, total // 4)
        print(f"\n  {len(unstable)} of {total} questions did not repeat, worst {worst} of "
              f"{repeats} repeats against its own majority:")
        for name in unstable[:10]:
            print(f"    {name}  {column.disagreement[name]}/{repeats} against majority")
        if len(unstable) > 10:
            print(f"    ... and {len(unstable) - 10} more")
        print("\n  " + (
            "Concentrated in a minority of questions, which reads as a few genuinely "
            "ambiguous answers rather than a judge that cannot repeat itself."
            if concentrated else
            "Spread across most of the question set. That is not a few hard cases; it "
            "means a single verdict from this judge carries little information, and "
            "anything built on one verdict per answer inherits that."
        ))
    print(f"\nrecorded to {args.out}. `uv run pytest tests/measurement/test_judge.py "
          f"-m reporting` now has a floor to measure against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
