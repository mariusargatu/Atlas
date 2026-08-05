"""The question set as a Langfuse dataset, and each run as a dataset run.

Deliberately duplicates `data/results/runs.jsonl`: the file is diffable with no container
runtime, the dataset run links a number back to the request that produced it. See
docs/recording-runs.md.

The join between them is `Settings.run_id`, which is both the key in `runs.jsonl` and the
`run_name` here, so a reader can move between the two without a lookup table.

**Gold is documents here, never chunks.** `expected_output` is the task's
`required_documents`. Chunk ids are not stored: they're a property of the chunker
(`atlas.corpus.gold.resolve` recomputes them against the chunk set under test), so a
stored chunk-level answer key would be correct for exactly one `ChunkSettings` and
silently wrong for every other. See docs/where-the-answers-come-from.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

from langfuse.experiment import Evaluation

from atlas.contracts import Question, QuestionName
from atlas.corpus.gold import Correct
from evals.report import RetrievalRow, retrieval_summary, scored_retrieval
from evals.results import QuestionRow

DATASET_NAME = "tau2-banking"

DATASET_DESCRIPTION = (
    "The 97 tasks from Sierra's tau2-bench banking domain, with each task's "
    "required_documents as the answer key. Third-party gold: nobody here chose which "
    "documents are correct, which is what makes a score on it worth anything. "
    "expected_output is documents, never chunks -- chunk-level gold is a property of "
    "the chunker and is resolved at run time against the chunk set under test."
)


def sync_items(client: Any, questions: Sequence[Question]) -> int:
    """Upsert one dataset item per question.

    Keyed on `id=question.name`, so running this twice updates in place rather than
    doubling the dataset. Called at the start of every dataset run so it can never be a
    step a reader forgets to trigger manually.
    """
    client.create_dataset(name=DATASET_NAME, description=DATASET_DESCRIPTION)
    for question in questions:
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=question.name,
            input={"question": question.text},
            expected_output={"required_documents": list(question.required)},
            metadata={
                "kind": question.kind,
                # Separate from required_documents: the graded metric scores only these,
                # and only 22 of 97 tasks name any.
                "primary_documents": list(question.primary),
                "source": "tau2-bench (Sierra), banking domain",
            },
        )
    return len(questions)


def item_evaluations(output: Mapping[str, Any], k: int) -> list[Any]:
    """Every score for one question, computed from the row the task returned.

    Reuses `scored_retrieval`, the same function the printed table averages, so the
    dashboard and the markdown can't disagree about what a metric means. Each ceiling
    travels into the score's comment since Langfuse's score view has no ceiling column.

    NaN is skipped rather than sent: a NaN posted as a score is a number-shaped hole that
    would average into everything downstream.
    """
    correct = Correct(
        question=QuestionName(str(output["question"])),
        chunks=frozenset(output["correct"]),
        primary=frozenset(output["primary"]),
    )
    ranked = tuple(output["rankings"].get("reranked", ()))

    evaluations: list[Any] = []
    for metric, bounded in scored_retrieval(ranked, correct, k).items():
        if bounded.value != bounded.value:  # NaN, and only NaN, is unequal to itself
            continue
        evaluations.append(Evaluation(
            name=metric, value=bounded.value,
            comment=f"ceiling {bounded.ceiling:.3f} on this question",
        ))

    # Free: an answer that refuses while citing sources, or answers while citing none,
    # contradicts itself on its own terms, with no model needed.
    if output.get("generated"):
        violations = tuple(output.get("violations", ()))
        evaluations.append(Evaluation(
            name="citation_violations", value=float(len(violations)),
            comment="; ".join(violations) or "none",
        ))
    if output.get("judge_verdict") is not None:
        evaluations.append(Evaluation(name="judge", value=str(output["judge_verdict"])))
    return evaluations


def run_evaluations(rows: Sequence[RetrievalRow], system: str = "") -> list[Any]:
    """The run's headline numbers, each one sent with the ceiling it was measured against.

    Three scores per metric: `mean` is the number, `ceiling` is the most this question set
    allows, and `attained` is the ratio to compare across runs. Split into three because a
    dashboard has one numeric field per score, unlike `evals.report`'s single printed line.

    `system` names a baseline arm so this view carries a floor beside its ceilings, the
    same invariant `evals.results.RunRow.__post_init__` enforces for the committed file.
    """
    prefix = f"{system} " if system else ""
    evaluations: list[Any] = []
    for row in rows:
        over = f"over {row.questions} question(s)"
        evaluations.append(Evaluation(
            name=f"{prefix}mean {row.metric}", value=row.mean, comment=over))
        evaluations.append(Evaluation(
            name=f"{prefix}ceiling {row.metric}", value=row.mean_ceiling,
            comment=f"the most this question set allows, {over}",
        ))
        evaluations.append(Evaluation(
            name=f"{prefix}attained {row.metric}", value=row.attained,
            comment=f"mean as a fraction of the ceiling -- compare this across runs, {over}",
        ))
    return evaluations


def run_experiment(
    client: Any,
    questions: Sequence[Question],
    pending: Sequence[Question],
    ask: Callable[[Question], QuestionRow],
    run_name: str,
    description: str,
    metadata: Mapping[str, Any],
    k: int,
    nulls: Mapping[str, Mapping[QuestionName, Sequence[str]]] | None = None,
) -> Any:
    """Drive the question set through Langfuse as one dataset run.

    `ask` is the same callable the plain loop uses, so this is a second *driver* over one
    implementation rather than a second implementation of the question loop.

    `questions` is the whole set; `pending` is what this run actually executes. Kept
    separate because syncing only `pending` would silently redefine the dataset under
    `--limit` for any later full run.

    Serial on purpose (`max_concurrency=1`): the ledger is an append-only file written
    from inside `ask` and isn't safe to write from several threads, and per-stage wall
    times need to stay comparable to the plain loop's.
    """
    by_name = {question.name: question for question in pending}

    def task(*, item: Any, **_: Any) -> dict[str, Any]:
        return asdict(ask(by_name[QuestionName(str(item.id))]))

    def evaluate(*, output: Any, **_: Any) -> list[Any]:
        return item_evaluations(output, k)

    def _aggregate(item_results: Sequence[Any], ranking_for: Any) -> list[RetrievalRow]:
        scored: dict[str, list[Any]] = {}
        for result in item_results:
            output = result.output
            if not output:
                continue
            correct = Correct(
                question=QuestionName(str(output["question"])),
                chunks=frozenset(output["correct"]),
                primary=frozenset(output["primary"]),
            )
            for metric, bounded in scored_retrieval(ranking_for(output), correct, k).items():
                scored.setdefault(metric, []).append(bounded)
        return list(retrieval_summary(scored))

    def evaluate_run(*, item_results: Sequence[Any], **_: Any) -> list[Any]:
        out = run_evaluations(_aggregate(item_results, lambda o: tuple(o["rankings"].get("reranked", ()))))
        for name, ranked_for in (nulls or {}).items():
            out += run_evaluations(
                _aggregate(item_results, lambda o, m=ranked_for: tuple(m.get(str(o["question"]), ()))),
                name,
            )
        return out

    sync_items(client, questions)
    # No fetch_items_page_size: the API caps it at 100, so `len(questions)` would 400 on
    # a 101st question, and get_dataset pages through the whole set regardless.
    dataset = client.get_dataset(DATASET_NAME)
    # Filtered rather than passing dataset.items wholesale, since --limit and the resume
    # ledger both narrow what this run should actually execute (and pay for).
    wanted = {question.name for question in pending}
    items = [item for item in dataset.items if str(item.id) in wanted]

    return client.run_experiment(
        name=DATASET_NAME, run_name=run_name, description=description,
        data=items, task=task, evaluators=[evaluate], run_evaluators=[evaluate_run],
        metadata=dict(metadata), max_concurrency=1,
    )
