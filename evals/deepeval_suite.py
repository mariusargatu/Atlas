"""The generation metrics the industry uses, as a contrast to this repository's own judge.

Three of these are the library's own implementations (faithfulness, answer relevancy,
contextual relevancy); the fourth, refusal correctness, is a GEval rubric written here
because the library has no equivalent.

`ContextualPrecisionMetric` and `ContextualRecallMetric` are excluded: both require
`expected_output`, a reference answer, and tau2's gold set never names one (see
docs/where-the-answers-come-from.md). `evals.ir_metrics.precision_at_k` and `recall_at_k`
already compute those quantities directly against the gold set instead of via an LLM's
guess.
"""

from __future__ import annotations

import asyncio
import copy
import math
import statistics
from collections.abc import Callable, Mapping, Sequence

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import GPTModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from atlas.config import Settings
from atlas.contracts import Answer, ChunkId, Question, QuestionName
from atlas.models.pricing import require_priced
from evals.calibration import agreement, agreement_with_reference_noise
from evals.judge import shown_passages
from evals.results import ContrastRow

# Citation validity is not a rubric here: it's `set(cited) <= set(shown)`, which
# `atlas.models.generate.record_violations` computes exactly for free, on every answer.
_REFUSAL_CORRECTNESS_STEPS = [
    "Read the question, the shown passages and the answer's outcome.",
    "Check whether the answer refuses, states unknown, or states not applicable "
    "when the shown passages genuinely lack the fact.",
    "Penalise a confident answer where the correct outcome was a refusal, unknown, "
    "or not applicable.",
]

_PARAMS = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT]


# Set explicitly rather than inherited from the library's default, since a default can
# move between releases and this number decides every agreement figure below.
CONTRAST_THRESHOLD = 0.5

# The judge models this targets reject temperature=0 unless recognized by the library's
# own model registry, so this can't run at 0.0 the way the structured judge does. The
# contrast therefore carries sampling noise the judge doesn't; read it across repeats.
CONTRAST_TEMPERATURE = 1.0

CONCURRENCY = 8


def _scoring_model(model: str) -> GPTModel:
    """The library's own client, constructed rather than named by string: passing `model=`
    a string makes the library build this itself with its default temperature, which the
    configured judge model rejects.

    `require_priced` is called for its refusal, not its return value, so a scorer can't
    silently spend on a model nobody priced. It doesn't close the whole gap: the library's
    tokens are spent inside its own client and never reach the cost ledger (see
    `evals.table.contrast_table`).
    """
    return GPTModel(model=require_priced(model), temperature=CONTRAST_TEMPERATURE)


def _metric(name: str, steps: list[str], model: str) -> GEval:
    return GEval(
        name=name, evaluation_steps=steps, evaluation_params=_PARAMS,
        model=_scoring_model(model), threshold=CONTRAST_THRESHOLD,
    )


def contrast_scorers(settings: Settings | None = None) -> tuple[BaseMetric, ...]:
    """Three metrics the library implements itself and one rubric it has no answer for.

    `refusal_correctness` stays a GEval rubric: whether an answer *should* have refused is
    a property of this corpus, where a question's required documents may simply not carry
    what it asks for, and no library metric expresses that.
    """
    settings = settings or Settings()
    model = settings.judge.model
    return (
        AnswerRelevancyMetric(model=_scoring_model(model), threshold=CONTRAST_THRESHOLD),
        FaithfulnessMetric(model=_scoring_model(model), threshold=CONTRAST_THRESHOLD),
        ContextualRelevancyMetric(model=_scoring_model(model), threshold=CONTRAST_THRESHOLD),
        _metric("refusal_correctness", _REFUSAL_CORRECTNESS_STEPS, model),
    )


def contrast_case(
    question: Question, answer: Answer, texts: Mapping[ChunkId, str]
) -> LLMTestCase:
    """One answer, in the shape this library reads.

    `retrieval_context` is built from `answer.shown`, never from what the answer claims it
    cited: those differ whenever the writer fabricates a citation, and handing a scorer the
    answer's own account of its evidence would let a fabricating system score well on
    faithfulness.
    """
    return LLMTestCase(
        input=question.text,
        actual_output=answer.text,
        retrieval_context=[text for _, text in shown_passages(answer, texts)],
    )


def _metric_name(scorer: BaseMetric) -> str:
    """The library exposes a readable `__name__` on a metric, which mypy reads as the class
    attribute rather than the instance one. Narrowed here so the read happens once."""
    name: object = getattr(scorer, "__name__", None)
    # __qualname__, not __name__, for the fallback: BaseMetric declares __name__ as a
    # property, so `type(scorer).__name__` resolves to the property object itself.
    return name if isinstance(name, str) else type(scorer).__qualname__


async def _measure_concurrently(
    scorer: BaseMetric, cases: Sequence[LLMTestCase]
) -> list[float]:
    """Every case scored, with a failure on one case costing only that one case.

    `asyncio.gather` propagates the first exception and cancels the rest by default, so a
    single unparseable model response would otherwise kill an entire completed, paid-for
    run. Failures are caught per case and returned as NaN, which `score_contrast` already
    excludes from the mean and agreement; failures are counted and printed, never swallowed.
    """
    limit = asyncio.Semaphore(CONCURRENCY)
    failures: list[str] = []

    async def one(case: LLMTestCase) -> float:
        async with limit:
            # A copy per call, not defensive tidiness: `a_measure` writes `self.score` and
            # returns it, so coroutines sharing one metric can interleave between the write
            # and the return and hand back another question's score. Shallow, so every copy
            # still shares the one model client.
            own = copy.copy(scorer)
            try:
                score = await own.a_measure(
                    case, _show_indicator=False, _log_metric_to_confident=False
                )
            except Exception as error:  # noqa: BLE001 -- any failure is one missing score
                failures.append(f"{type(error).__name__}: {error}")
                return math.nan
            return float(score) if score is not None else math.nan

    scores = list(await asyncio.gather(*(one(case) for case in cases)))
    if failures:
        print(f"    {len(failures)} of {len(cases)} raised and scored nothing; "
              f"first was {failures[0][:120]}", flush=True)
    return scores


def _measure_all(scorer: BaseMetric, cases: Sequence[LLMTestCase]) -> list[float]:
    return asyncio.run(_measure_concurrently(scorer, cases))


def contrast_row(
    metric: str,
    scores: Mapping[QuestionName, float],
    verdicts: Mapping[QuestionName, str],
    threshold: float,
    seed: int = 0,
    judge_flip_floor: float | None = None,
) -> ContrastRow | None:
    """One scorer's row, from numbers already in hand. None when nothing was scorable.

    Split out from `score_contrast` so a resumed run can rebuild the comparison from
    `QuestionRow.deepeval_scores` without spending anything again.

    `judge_flip_floor` folds the judge's own measured self-disagreement into a second
    interval via `agreement_with_reference_noise`. That interval is *lower* than the plain
    one rather than wider, so it reads as a ceiling on the row's headline number, not an
    error bar around it.
    """
    # Filtered before pairing rather than after: building `called` from a raw list lets
    # NaN silently become "fail" (NaN is not >= any threshold), recording a scorer that
    # simply failed to score as one that disagreed with a passing judge.
    scored = [(name, value) for name, value in sorted(scores.items())
              if name in verdicts and not math.isnan(value)]
    if not scored:
        return None
    called = ["pass" if value >= threshold else "fail" for _, value in scored]
    reference = [verdicts[name] for name, _ in scored]
    against = agreement(reference, called, seed)
    noisy = (
        agreement_with_reference_noise(reference, called, judge_flip_floor, seed)
        if judge_flip_floor is not None else None
    )
    return ContrastRow(
        metric=metric, mean=statistics.fmean(value for _, value in scored),
        threshold=threshold, raw_agreement=against.raw,
        chance_corrected=against.chance_corrected,
        low=against.low, high=against.high, underpowered=against.underpowered,
        low_with_judge_noise=noisy.low if noisy is not None else None,
        high_with_judge_noise=noisy.high if noisy is not None else None,
        questions=len(scored),
    )


def scorer_thresholds(scorers: Sequence[BaseMetric] | None = None) -> dict[str, float]:
    """Each scorer's cut point, by name. Never the library attribute unguarded: it is typed
    as optional, and a None would compare against nothing and put a hole in the one column
    that says which cut point produced the agreement beside it."""
    return {
        _metric_name(scorer):
            scorer.threshold if scorer.threshold is not None else CONTRAST_THRESHOLD
        for scorer in (scorers if scorers is not None else contrast_scorers())
    }


def score_contrast(
    cases: Mapping[QuestionName, LLMTestCase],
    verdicts: Mapping[QuestionName, str],
    scorers: Sequence[BaseMetric] | None = None,
    seed: int = 0,
    on_scores: Callable[[str, Mapping[QuestionName, float]], None] | None = None,
) -> tuple[ContrastRow, ...]:
    """Scores every case with every scorer and compares each against the judge.

    Paired on question name rather than position, so two sequences of the same length
    over different questions can't silently line up wrong.

    Costs at least one model call per scorer per question, several for scorers that
    decompose an answer into claims first; run deliberately, not in a loop. Scored
    concurrently, bounded by CONCURRENCY, since the work is entirely waiting on an
    endpoint and serial scoring here once took over thirty minutes at 0.6% CPU.
    """
    scorers = list(scorers if scorers is not None else contrast_scorers())
    judged = {name: verdict for name, verdict in verdicts.items() if name in cases}
    names = sorted(judged, key=str)
    if not names:
        return ()

    rows = []
    for position, scorer in enumerate(scorers, start=1):
        # Printed because this stage is slow and silent otherwise, indistinguishable from
        # a hang.
        print(f"  contrast {position}/{len(scorers)}: {_metric_name(scorer)} "
              f"over {len(names)} answers", flush=True)
        scores = _measure_all(scorer, [cases[name] for name in names])
        if on_scores is not None:
            on_scores(_metric_name(scorer), dict(zip(names, scores, strict=True)))
        scored = [(n, s) for n, s in zip(names, scores, strict=True) if not math.isnan(s)]
        if not scored:
            print(f"    {_metric_name(scorer)} produced no score on any of "
                  f"{len(names)} answers and is absent from the table entirely", flush=True)
            continue
        if len(scored) < len(names):
            print(f"    scored {len(scored)} of {len(names)}; the rest are excluded "
                  "from the mean and the agreement", flush=True)
        threshold = scorer.threshold if scorer.threshold is not None else CONTRAST_THRESHOLD
        row = contrast_row(_metric_name(scorer), dict(scored), judged, threshold, seed)
        if row is not None:
            rows.append(row)
    return tuple(rows)
