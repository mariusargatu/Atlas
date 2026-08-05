"""Rendering a comparison as a header a reader can trust: the resolvable
difference travels as a number on the object, not just as text somebody has to
search for.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from atlas.contracts import ChunkId, QuestionName, Record
from atlas.corpus.gold import Correct
from atlas.pipeline import STAGES
from evals.ir_metrics import (
    Bounded,
    bounded_graded_ndcg_at_k,
    bounded_ndcg_at_k,
    bounded_precision_at_k,
    bounded_recall_at_k,
    reciprocal_rank_at_k,
    success_at_k,
)
from evals.results import MetricRow, QuestionRow
from evals.stats import (
    PairedComparison,
    paired_comparison,
    smallest_resolvable_difference,
)


@dataclass(frozen=True, slots=True)
class ComparisonHeader:
    resolvable_difference: float
    text: str


def comparison_header(name_a: str, name_b: str, questions: int) -> ComparisonHeader:
    resolvable = smallest_resolvable_difference(questions)
    text = (
        f"{name_a} vs {name_b}, {questions} questions, "
        f"smallest resolvable difference {resolvable:.2f}"
    )
    return ComparisonHeader(resolvable_difference=resolvable, text=text)


@dataclass(frozen=True, slots=True)
class RetrievalRow:
    """One metric averaged over a question set, with its ceiling beside it.

    `ceiling` is one for a metric that is already normalised, and that is not padding:
    a reader comparing rows needs to see which maxima are one and which are not,
    and a blank in that column reads as "not applicable" rather than as "one".
    """

    metric: str
    mean: float
    mean_ceiling: float
    attained: float
    questions: int

    @property
    def line(self) -> str:
        if self.mean_ceiling >= 1.0 - 1e-9:
            return f"{self.metric:<20} {self.mean:.3f}"
        return (
            f"{self.metric:<20} {self.mean:.3f}   of an achievable "
            f"{self.mean_ceiling:.3f}   ({self.attained:.0%} attained)"
        )


def retrieval_summary(scores: Mapping[str, Sequence[Bounded]]) -> tuple[RetrievalRow, ...]:
    """Averages each metric over the questions it could be computed for.

    Takes Bounded rather than float, with no overload accepting a bare number: a metric
    whose maximum depends on the question can't reach this function without carrying that
    maximum along.

    NaN is excluded rather than propagated; the surviving count travels on the row so a
    figure averaged over three questions can't be mistaken for one averaged over ninety-seven.
    """
    rows = []
    for metric, values in scores.items():
        usable = [b for b in values if not (math.isnan(b.value) or math.isnan(b.ceiling))]
        if not usable:
            continue
        mean = statistics.fmean(b.value for b in usable)
        mean_ceiling = statistics.fmean(b.ceiling for b in usable)
        rows.append(RetrievalRow(
            metric=metric, mean=mean, mean_ceiling=mean_ceiling,
            # Ratio of the means, not mean of the ratios: a question with a tiny ceiling
            # would otherwise dominate, since attaining all of very little scores as 1.0
            # just as attaining all of a lot does.
            attained=mean / mean_ceiling if mean_ceiling else math.nan,
            questions=len(usable),
        ))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class LatencyRow:
    stage: str
    median_ms: float
    slowest_tenth_ms: float
    mean_ms: float


def _slowest_tenth(values: list[float]) -> float:
    # Mean of the slowest tenth, not the p90 value itself: with few recorded questions,
    # interpolating a single percentile point can miss the one call that was actually
    # slow, and reranking cost shows up in the tail rather than the median.
    tail_count = max(1, math.ceil(len(values) / 10))
    tail = sorted(values)[-tail_count:]
    return statistics.fmean(tail)


def latency_from_ledger(rows: Sequence[QuestionRow]) -> tuple[LatencyRow, ...]:
    """One row per stage, in the pipeline's own order, from the persisted ledger.

    Reads from the ledger rather than live `Record`s so a fully resumed run (which
    executes nothing and holds no Records) still reports real latency, matching every
    other figure in the run row.
    """
    per_stage: dict[str, list[float]] = {}
    for row in rows:
        for stage, elapsed_ms in row.timings_ms.items():
            per_stage.setdefault(stage, []).append(elapsed_ms)
    return tuple(
        LatencyRow(stage=stage, median_ms=statistics.median(values),
                   slowest_tenth_ms=_slowest_tenth(values), mean_ms=statistics.fmean(values))
        for stage in STAGES if (values := per_stage.get(stage))
    )


@dataclass(frozen=True, slots=True)
class CostSummary:
    by_stage: dict[str, float]
    total_usd: float


def cost_from_ledger(rows: Sequence[QuestionRow]) -> CostSummary:
    """What a run spent, per question, as recorded rather than as re-observed."""
    by_stage = {"answer": sum(row.answer_cost_usd for row in rows),
                "judge": sum(row.judge_cost_usd for row in rows)}
    return CostSummary(by_stage=by_stage, total_usd=sum(by_stage.values()))


@dataclass(frozen=True, slots=True)
class RunSample:
    """Enough about one side of a before/after comparison to write a verdict: a
    quality score per question, paired by name, plus the two figures a change in
    reranking actually costs."""

    quality: Mapping[QuestionName, float]
    slowest_tenth_ms: float
    cost_usd: float


@dataclass(frozen=True, slots=True)
class RerankingVerdict:
    quality_change: PairedComparison
    slowest_tenth_change_ms: float
    cost_change_usd: float
    earned_its_place: bool


def reranking_verdict(before: RunSample, after: RunSample, seed: int) -> RerankingVerdict:
    """Quality, money and response time in one object: whether a reranker is worth
    keeping isn't answerable from a quality gain alone, since the same gain bought with a
    slower tail or a larger bill is a different decision. Earns its place only when the
    paired quality interval excludes zero on the improving side.
    """
    quality_change = paired_comparison(before.quality, after.quality, seed)
    return RerankingVerdict(
        quality_change=quality_change,
        slowest_tenth_change_ms=after.slowest_tenth_ms - before.slowest_tenth_ms,
        cost_change_usd=after.cost_usd - before.cost_usd,
        earned_its_place=quality_change.low > 0.0,
    )


STAGE_RANKINGS = ("vector", "keyword", "fused", "reranked")


def rankings(record: Record) -> dict[str, tuple[ChunkId, ...]]:
    return {
        "vector": tuple(h.chunk for h in record.vector.hits),
        "keyword": tuple(h.chunk for h in record.keyword.hits),
        "fused": tuple(h.chunk for h in record.fused.hits),
        "reranked": tuple(h.chunk for h in record.reranked.hits),
    }


def scored_retrieval(
    ranked: Sequence[ChunkId], correct: Correct, k: int
) -> dict[str, Bounded]:
    """Every retrieval metric for one ranked list, each carrying its own ceiling."""
    scores = {
        "recall@k": bounded_recall_at_k(ranked, correct.chunks, k),
        "precision@k": bounded_precision_at_k(ranked, correct.chunks, k),
        "MRR@k": Bounded(reciprocal_rank_at_k(ranked, correct.chunks, k), 1.0),
        "nDCG@k": bounded_ndcg_at_k(ranked, correct.chunks, k),
        "success@k": Bounded(success_at_k(ranked, correct.chunks, k), 1.0),
    }
    # NaN when the source can't say which documents carry the answer, never a silent
    # fallback to the flat metric.
    scores["graded nDCG@k"] = bounded_graded_ndcg_at_k(ranked, correct, k)
    return scores


def metric_rows(system: str, rows: Sequence[RetrievalRow]) -> tuple[MetricRow, ...]:
    """Names an already averaged set of rows with the system that produced them."""
    return tuple(
        MetricRow(system=system, metric=row.metric, value=row.mean,
                  ceiling=row.mean_ceiling, questions=row.questions)
        for row in rows
    )
