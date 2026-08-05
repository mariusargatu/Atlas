"""Score the question set, record the run, print the table a chapter pastes.

    just report                      # retrieval only: free against a warm cache
    just report --generate           # adds a written answer per question
    just report --judge              # adds a verdict per answer, implies --generate
    just report --settings FILE      # a different configuration, and a different run_id
    just report --render             # print what is already recorded; no key needed
    just report --render --all       # every recorded run, one line each

With Langfuse configured every run is also a dataset run, keyed by the same run_id.
See docs/recording-runs.md.

Three phases with different costs, so the expensive one can be skipped or resumed.
Collecting writes one line per question to a gitignored ledger under .cache/report;
summarising turns that into one committed line in data/results/runs.jsonl; rendering
reads that file and touches nothing else. An interrupted run resumes from its ledger
and pays only for what it has not already bought.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from atlas.config import Settings, load_settings
from atlas.corpus.chunk import cut
from atlas.corpus.registry import load
from atlas.models.embed import CachedEmbedder
from atlas.models.pricing import cost_usd
from atlas.models.providers import OpenAIEmbedder
from atlas.pipeline import PreparedCorpus
from atlas.trace import ENABLED as TRACING_ENABLED
from atlas.trace import flush
from evals.results import (
    RUNS_PATH,
    RunRow,
    append_run,
    latest_by_run,
    ledger_path,
    read_questions,
    read_runs,
    run_key,
)
from evals.table import chapter_block, run_index
from scripts.collect import _collect
from scripts.summarise import _contrast_rows, _summarise

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "vectors"
SYSTEMS = ("vector", "keyword", "fused", "reranked")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="write an answer per question")
    parser.add_argument("--judge", action="store_true",
                        help="grade each answer; implies --generate")
    parser.add_argument("--contrast", action="store_true",
                        help="score answers with deepeval too and compare; implies --judge")
    parser.add_argument("--settings", default=None,
                        help="a settings file: a diff from the defaults")
    parser.add_argument("--k", type=int, default=10, help="metric depth")
    parser.add_argument("--limit", type=_positive, default=None,
                        help="first N questions; prints, never records")
    parser.add_argument("--force", action="store_true",
                        help="append a second row for a run_id already recorded")
    parser.add_argument("--fresh", action="store_true",
                        help="discard the resume ledger and pay for every question again")
    parser.add_argument("--render", action="store_true",
                        help="print a recorded run; needs no key")
    parser.add_argument("--run-id", default=None, help="which recorded run to render")
    parser.add_argument("--all", action="store_true", help="with --render: one line per run")
    parser.add_argument("--grain", choices=("retrieval", "generate", "judge", "contrast"),
                        default=None,
                        help="with --render: which recorded grain of a run to print")
    return parser.parse_args()


def _positive(text: str) -> int:
    """0 must be rejected: every `if args.limit` check below treats 0 as falsy, so
    `--limit 0` would silently mean "no limit" instead of "no questions"."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"--limit {value} would mean the whole question set, not none of it. "
            "Use 1 or more, or leave the flag off."
        )
    return value


def _langfuse_reachable() -> bool:
    """Whether the configured Langfuse will actually answer, not merely whether keys
    exist: `run_experiment`'s first call is an HTTP request with no fallback, and
    `.env.example` ships keys pointing at localhost, so keys-set-but-stack-down is the
    ordinary case rather than a corner."""
    try:
        from langfuse import Langfuse

        return bool(Langfuse().auth_check())
    except Exception:  # noqa: BLE001 -- unreachable, unauthorised and misconfigured are one answer here
        return False


def _render(args: argparse.Namespace) -> int:
    """Reads the committed store and prints. No model client is constructed on this
    path, so a reader with no key reaches every published table."""
    rows = latest_by_run(read_runs(ROOT / RUNS_PATH))
    if args.all or not rows:
        print(run_index(rows))
        return 0

    # A run_id can hold several rows (retrieval-only, judged, contrast); `latest_by_run`
    # keys on five fields, so picking one grain to print needs its own logic.
    def grain_of(row: RunRow) -> str:
        if row.contrast:
            return "contrast"
        return "judge" if row.judged else "generate" if row.generated else "retrieval"

    # With no run named, prefer the defaults over whichever configuration was recorded
    # most recently, so `--render` matches what a reader of the default-run docs expects.
    matching = [r for r in rows if r.run_id == args.run_id] if args.run_id else list(rows)
    if not args.run_id:
        default = Settings().run_id
        matching = [r for r in matching if r.run_id == default] or matching
    if args.run_id and not matching:
        print(f"no recorded run {args.run_id!r}. Recorded runs:\n{run_index(rows)}")
        return 1
    if args.grain:
        wanted = [r for r in matching if grain_of(r) == args.grain]
        if not wanted:
            print(f"no {args.grain!r} grain recorded"
                  f"{' for ' + args.run_id if args.run_id else ''}. Recorded: "
                  + ", ".join(sorted({grain_of(r) for r in matching})))
            return 1
        matching = wanted
    chosen = max(matching, key=lambda r: r.recorded) if matching else None
    if args.run_id and len(matching) > 1:
        grains = ", ".join(f"{grain_of(r)} (k={r.k}, {r.recorded[:19]})"
                           for r in sorted(matching, key=lambda r: r.recorded))
        print(f"{args.run_id} has {len(matching)} recorded grains: {grains}\n"
              f"printing the most recent; `--grain` picks one.\n")
    if chosen is None:
        print(f"no recorded run {args.run_id!r}. Recorded runs:\n{run_index(rows)}")
        return 1
    print(chapter_block(chosen))
    return 0


def main() -> int:
    args = _parse()
    if args.render:
        return _render(args)
    if args.contrast:
        args.judge = True
    if args.judge:
        args.generate = True

    # A live default rather than a flag: tracing is decided once from the environment,
    # and the dataset run follows it, costing no extra model call and only ingestion.
    args.dataset = TRACING_ENABLED and _langfuse_reachable()
    if TRACING_ENABLED and not args.dataset:
        print("Langfuse is configured but not answering, so this run records to "
              "data/results/runs.jsonl only. `just up` starts it; everything else here "
              "works without it.\n")

    if args.fresh and args.limit:
        # Refused: the ledger is keyed by run_id, which doesn't vary with --limit, so
        # `--fresh --limit 3` would delete however many questions were banked and write
        # back three. This has cost real money before.
        print("--fresh --limit would delete the whole ledger for this run_id and refill "
              "it with the first N questions. Drop --limit to re-run everything, or drop "
              "--fresh to reuse what is already banked.")
        return 1

    settings = load_settings(args.settings) if args.settings else Settings()
    collection, questions = load()
    chunks = cut(collection.documents, settings.chunk)
    # The full set is kept alongside the limited one: the Langfuse dataset is the
    # question set and must hold all 97 regardless of how few this run executes.
    all_questions = questions
    if args.limit:
        questions = questions[: args.limit]

    # Not a refusal: the run happens whether or not the run_id is already recorded.
    # Only the append is withheld when it's already there.
    recorded = {run_key(row) for row in read_runs(ROOT / RUNS_PATH)}
    already_recorded = (settings.run_id, args.k, args.generate, args.judge,
                        args.contrast) in recorded

    print(f"run:       {settings.run_id}")
    print(f"corpus:    {len(collection.documents)} documents -> {len(chunks)} chunks")
    print(f"questions: {len(questions)}  k={args.k}  "
          f"generate={args.generate} judge={args.judge}\n")

    embedder = OpenAIEmbedder(settings.embedding)
    started = time.perf_counter()
    prepared = PreparedCorpus.build(settings, chunks, embedder=CachedEmbedder(embedder, CACHE))
    embedding_cost = cost_usd(settings.embedding.model_name, embedder.input_tokens, 0)

    ledger = ROOT / ledger_path(settings.run_id)
    # --fresh spends money (rebuilds the ledger); --force only decides whether a second
    # row may be written to runs.jsonl. Kept as two separate flags because collapsing
    # them once meant "record again" also implied "re-buy everything".
    if args.fresh and ledger.exists():
        ledger.unlink()
    records = _collect(settings, args, prepared, questions, ledger, all_questions)
    wall = time.perf_counter() - started

    rows = read_questions(ledger)
    # Every asked-for question must have a ledger row before anything is summarised.
    # On the Langfuse dataset-run path a failed question's exception is swallowed and
    # the run still exits 0, so a silently shrunk denominator is refused here rather
    # than reported as a smaller-but-honest count: the drops are selected by content
    # (the judge raises on a provider decline), which is the direction that flatters
    # the system.
    in_ledger = {r.question for r in rows}
    missing = [q.name for q in questions if q.name not in in_ledger]
    if missing:
        shown = ", ".join(missing[:8]) + (f", and {len(missing) - 8} more" if len(missing) > 8 else "")
        print(
            f"\n{len(missing)} of {len(questions)} questions produced no ledger row and this "
            f"run has not been recorded: {shown}.\n"
            "A run that lost questions is not a run over the set it names. Re-run to retry "
            "just those -- the ledger keeps everything already paid for, so the ones that "
            "succeeded are not bought again."
        )
        flush()
        return 1
    minted_here = frozenset(r.trace_id for r in records if r.trace_id)
    contrast = (_contrast_rows(rows, prepared, questions, minted_here, ledger, settings)
                if args.contrast else ())
    row = _summarise(settings, args, rows, questions, chunks, records, embedding_cost,
                     wall, contrast)

    flush()
    if not TRACING_ENABLED:
        scored = "no scores sent: tracing is off, so there was no trace to attach one to"
    elif not args.dataset:
        scored = ("no scores sent: Langfuse is configured but was not answering when "
                  "this run started")
    elif records:
        scored = ("scores were written by the dataset run's evaluators, on the traces the "
                  "run opened (ingestion is asynchronous; check the dashboard)")
    else:
        scored = ("no dataset run: every question came from the ledger, so there was "
                  "nothing to score. `--fresh` re-runs and re-traces them.")

    if args.limit:
        # A limited run scores a different question set under the same run_id, and two
        # rows filed under one key is the one thing this store exists to prevent.
        print(chapter_block(row))
        print(f"\n{scored}")
        print(f"--limit {args.limit}: printed, not recorded. "
              "A partial question set is not a run.")
        return 0

    if already_recorded and not args.force:
        print(chapter_block(row))
        print(f"\n{scored}")
        print(f"\nthis configuration is already in {RUNS_PATH} at this k and grain, so "
              "nothing was appended. `--force` records it again.")
        return 0

    append_run(ROOT / RUNS_PATH, row)
    print(chapter_block(row))
    print(f"\n{scored}")
    print(f"\nrecorded to {RUNS_PATH}. `just report --render` prints it without a key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
