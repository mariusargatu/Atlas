"""Record human verdicts into data/labels.jsonl, one answer at a time.

    just label --sample 25        # 25 unlabelled answers, drawn at random
    just label --report           # agreement between your verdicts and the judge's
    just label --question task_012

Grades the run the current settings describe; there's no flag naming another, since
each label carries the judge's verdict on the same answer.

Shows the question, the passages the writer was shown, the answer, its outcome and
citations, and the scoring guide -- the same evidence the judge sees. It never shows
the judge's own verdict: a rater who can see that isn't independent of it, and the
agreement score would measure anchoring instead.

data/labels.jsonl is the one artefact here that can't be regenerated at any price.
"""

from __future__ import annotations

import argparse
import random
import textwrap
from collections import Counter
from pathlib import Path
from typing import cast

from atlas.config import Settings
from atlas.contracts import ZERO_USAGE, ChunkId
from atlas.corpus.chunk import cut
from atlas.corpus.registry import get_source
from evals.calibration import agreement_against_labels
from evals.judge import JudgeVerdict, Verdict
from evals.labels import HUMAN_RATER, LABEL_SCHEMA_VERSION, Label, append_label, read_labels
from evals.results import QuestionRow, latest_questions, ledger_path

LABELS_PATH = "data/labels.jsonl"
ROOT = Path(__file__).resolve().parents[1]
VERDICTS = ("pass", "fail")
_RULE = "-" * 78


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record human verdicts on the current run's answers into "
                    "data/labels.jsonl, and report their agreement with the judge.",
        epilog="just label --sample 25   |   just label --report   |   just label --question task_012",
    )
    parser.add_argument("--question", default=None, help="grade one named question")
    parser.add_argument("--limit", type=int, default=None, help="stop after N answers")
    parser.add_argument("--sample", type=int, default=None,
                        help="draw N unlabelled answers at random rather than taking them in order")
    parser.add_argument("--seed", type=int, default=0, help="which draw --sample takes")
    parser.add_argument("--report", action="store_true",
                        help="print agreement between the labels already recorded and the judge, and exit")
    return parser.parse_args()


def _report(settings: Settings) -> int:
    """Agreement between recorded labels and the judge, chance-corrected.

    Reads only data/labels.jsonl: each label carries the judge's verdict on the same
    answer, so no ledger or run id lookup is needed.

    Grouped by (run_id, rater) rather than pooled: labels from different runs may have
    been graded under different judge prompts, and a human's verdicts and a model's
    are not the same rater, so a union would average across distinct measurements.
    """
    labels = read_labels(LABELS_PATH)
    if not labels:
        print(f"{LABELS_PATH} is empty. `just label --sample 25` records verdicts.")
        return 1

    by_group: dict[tuple[str, str], list[Label]] = {}
    for label in labels:
        by_group.setdefault((label.run_id, label.rater), []).append(label)
    if len(by_group) > 1:
        print(f"{len(labels)} labels across {len(by_group)} run/rater groups, reported "
              f"separately: one kappa over the union would describe none of them.")

    for (run_id, rater), group in sorted(by_group.items()):
        verdicts = [
            JudgeVerdict(
                question=label.question, verdict=cast("Verdict", label.judge_verdict),
                reason="", rubric_version=label.rubric_version,
                prompt_version=settings.judge.prompt_version, model=settings.judge.model,
                usage=ZERO_USAGE,
            )
            for label in group
        ]
        result = agreement_against_labels(group, verdicts, settings)
        human = Counter(label.verdict for label in group)
        machine = Counter(v.verdict for v in verdicts)
        who = "human" if rater == HUMAN_RATER else rater
        print(f"\n{len(group)} answer(s) from run {run_id}, graded by {who}")
        print(f"  {who} said {dict(human)}, the judge said {dict(machine)}")
        if rater != HUMAN_RATER:
            print("  NOT human agreement. A model grading a model's answers against the "
                  "same guide shares its priors, so this is a second opinion, not a "
                  "ground truth, and it cannot stand in for a labelled set.")
        print(f"  raw agreement    {result.raw:.3f}")
        print(f"  chance-corrected {result.chance_corrected:+.3f}  "
              f"95% interval [{result.low:+.3f}, {result.high:+.3f}]")
        if result.underpowered:
            print("  UNDERPOWERED: the interval is too wide, or one of the two raters "
                  "never varied, so the point estimate above should not be read on its "
                  "own. More labels, or labels covering both verdicts, is the only fix.")
    return 0


def _show(row: QuestionRow, question_text: str, texts: dict[ChunkId, str], rubric: str) -> None:
    print(f"\n{_RULE}\n{row.question}\n{_RULE}")
    print(textwrap.fill(" ".join(question_text.split()), 78))
    print("\npassages the writer was shown:")
    for name in row.shown:
        body = " ".join(texts.get(ChunkId(name), "(missing)").split())
        print(textwrap.fill(f"[{name}] {body}", 78, initial_indent="  ",
                            subsequent_indent="      ")[:600])
    print("\nanswer:")
    print(textwrap.fill(" ".join(row.answer_text.split()), 78, initial_indent="  ",
                        subsequent_indent="  "))
    print(f"\noutcome: {row.outcome}   cited: {', '.join(row.cited) or '(nothing)'}")
    if row.violations:
        print(f"violations: {', '.join(row.violations)}")
    print(f"\nscoring guide:\n{textwrap.indent(rubric.strip(), '  ')}")


def main() -> int:
    args = _parse()
    settings = Settings()
    rubric = Path(settings.judge.rubric_path).read_text(encoding="utf-8").split("---", 2)[2]

    already = {label.question for label in read_labels(LABELS_PATH)}
    source = get_source()
    by_name = {q.name: q for q in source.questions()}

    if args.report:
        return _report(settings)

    run_id = settings.run_id
    latest = latest_questions(ledger_path(run_id, ROOT / ".cache/report"))
    if not latest:
        print(f"the current settings hash to {run_id}, which has no judged run yet.\n")
        labellable = []
        for path in sorted((ROOT / ".cache/report").glob("*.jsonl")):
            answered = sum(1 for r in latest_questions(path).values() if r.answer_text)
            if answered:
                labellable.append((path.stem, answered))
        if labellable:
            print("Other ledgers do hold answers, from settings you are no longer running:")
            for other, answered in labellable:
                print(f"  {other}   {answered} answers")
            print("\nLabelling those would compare you against a judge reading a different\n"
                  "prompt. `just report --generate --judge` writes a run for these settings.")
        else:
            print("`just report --generate --judge` writes one; retrieval-only runs record\n"
                  "rankings and have nothing to grade.")
        return 1

    pending = [r for r in latest.values()
               if r.answer_text and r.judge_verdict and r.question not in already]
    if args.question:
        pending = [r for r in pending if r.question == args.question]
    # A named draw, not the first N: ledger order tracks tau2's task numbering and
    # therefore its topics, so `--limit 20` would sample a slice of the corpus rather
    # than a sample of it. Not stratified by the judge's own verdict either, since that
    # would condition the sample on the thing being measured.
    if args.sample:
        pending = random.Random(args.seed).sample(pending, min(args.sample, len(pending)))
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print(f"nothing to label: {len(already)} already recorded, and every judged "
              f"answer in {run_id} is among them.")
        return 0

    texts = {c.name: c.text for c in cut(source(0).documents, settings.chunk)}

    print(f"{len(pending)} unlabelled answer(s). Blank verdict quits; nothing is written "
          f"until you answer both prompts.")
    written = 0
    for row in pending:
        question = by_name.get(row.question)
        if question is None:
            continue
        _show(row, question.text, texts, rubric)
        verdict = input(f"\nverdict ({'/'.join(VERDICTS)}, blank to stop): ").strip()
        if not verdict:
            break
        if verdict not in VERDICTS:
            print(f"  {verdict!r} is not one of {VERDICTS}; skipped, nothing written")
            continue
        reason = input("reason: ").strip()
        if not reason:
            print("  a verdict with no reason cannot be argued with later; skipped")
            continue
        append_label(LABELS_PATH, Label(
            schema=LABEL_SCHEMA_VERSION, question=row.question,
            rubric_version=settings.judge.rubric_version, verdict=verdict,
            reason=reason, run_id=run_id,
            judge_verdict=row.judge_verdict or "",
            rater=HUMAN_RATER,
        ))
        written += 1
        print(f"  recorded ({written} this session)")

    print(f"\n{written} label(s) appended to {LABELS_PATH}. "
          f"{len(already) + written} in the store.")
    if written:
        _report(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
