from __future__ import annotations

import math

import pytest

from evals.ir_metrics import Bounded
from evals.report import (
    cost_from_ledger,
    latency_from_ledger,
    reranking_verdict,
    retrieval_summary,
)
from evals.results import RESULTS_SCHEMA_VERSION, QuestionRow

VERDICT_CASES = (
    ("a clear gain that cost response time", 0.06, 0.02, 120.0, True),
    ("a gain the interval cannot separate from zero", 0.01, 0.15, 120.0, False),
    ("a loss", -0.05, 0.02, -5.0, False),
)


def _as_ledger(records):
    """The Record fixtures as the QuestionRows a run actually summarises from."""
    return tuple(
        QuestionRow(
            schema=RESULTS_SCHEMA_VERSION, run_id="r", question=record.question,
            generated=record.answer is not None, judged=record.judge is not None,
            rankings={}, correct=(), primary=(),
            timings_ms={t.stage: t.wall_ms for t in record.timings},
            answer_cost_usd=record.answer.usage.cost_usd if record.answer else 0.0,
            judge_cost_usd=record.judge.usage.cost_usd if record.judge else 0.0,
        )
        for record in records
    )


def test_every_stage_appears_once_in_the_latency_summary(recorded_runs) -> None:
    rows = latency_from_ledger(_as_ledger(recorded_runs))
    assert [row.stage for row in rows] == ["vector", "keyword", "fuse", "rerank", "answer"]
    assert all(math.isfinite(row.median_ms) and math.isfinite(row.mean_ms) for row in rows)
    assert all(row.slowest_tenth_ms >= row.median_ms >= 0.0 for row in rows)


def test_the_slowest_tenth_is_a_tail_figure_and_not_the_median_again(runs_with_one_slow_question):
    # An implementation returning the median under both names passes on a single recorded
    # question, and whether reranking earned its place is argued on the tail.
    row = next(r for r in latency_from_ledger(_as_ledger(runs_with_one_slow_question))
               if r.stage == "answer")
    assert row.slowest_tenth_ms > row.median_ms and row.mean_ms > row.median_ms


def test_cost_summary_totals_the_usage_it_was_given_and_nothing_else(recorded_runs) -> None:
    summary = cost_from_ledger(_as_ledger(recorded_runs))
    assert set(summary.by_stage) == {"answer", "judge"}
    assert summary.total_usd == pytest.approx(sum(summary.by_stage.values()))
    assert summary.by_stage["answer"] == pytest.approx(
        sum(r.answer.usage.cost_usd for r in recorded_runs if r.answer is not None))
    assert summary.by_stage["judge"] == pytest.approx(
        sum(r.judge.usage.cost_usd for r in recorded_runs if r.judge is not None))


@pytest.mark.parametrize("change,spread,tail_change,earned", [c[1:] for c in VERDICT_CASES],
                         ids=[c[0] for c in VERDICT_CASES])
def test_the_reranking_verdict_reports_quality_money_and_response_time_together(
        change, spread, tail_change, earned, verdict_runs) -> None:
    # Three distinct pairs, one per outcome. Passing the same run in as both sides makes
    # every difference exactly zero and every assertion holds for a verdict that computes
    # all three wrongly, sign included.
    before, after = verdict_runs(change=change, spread=spread, tail_change=tail_change)
    verdict = reranking_verdict(before=before, after=after, seed=0)
    assert verdict.earned_its_place is earned
    assert verdict.quality_change.low <= change <= verdict.quality_change.high
    assert (verdict.quality_change.low > 0.0) is earned
    assert verdict.slowest_tenth_change_ms == pytest.approx(tail_change)
    assert math.isfinite(verdict.cost_change_usd)


def test_swapping_before_and_after_flips_the_sign_of_the_quality_change(verdict_runs) -> None:
    before, after = verdict_runs(change=0.06, spread=0.02, tail_change=120.0)
    forward = reranking_verdict(before=before, after=after, seed=0)
    backward = reranking_verdict(before=after, after=before, seed=0)
    assert forward.quality_change.low > 0.0 > backward.quality_change.high
    assert forward.slowest_tenth_change_ms == pytest.approx(-backward.slowest_tenth_change_ms)
    assert backward.earned_its_place is False


def test_a_capped_metric_cannot_reach_the_summary_without_its_ceiling() -> None:
    # The type doing the work: a caller cannot report recall without saying what recall
    # could have reached.
    with pytest.raises((TypeError, AttributeError)):
        retrieval_summary({"recall@k": [0.2, 0.3]})  # type: ignore[list-item]


def test_the_summary_reports_the_ceiling_and_the_share_attained() -> None:
    # Twenty-one correct chunks and ten slots caps recall at 10/21, so 0.2 is not a fifth
    # of what was possible, it is 42% of it.
    ceiling = 10 / 21
    rows = retrieval_summary({"recall@k": [Bounded(0.2, ceiling), Bounded(0.2, ceiling)]})
    assert len(rows) == 1
    row = rows[0]
    assert row.mean == pytest.approx(0.2)
    assert row.mean_ceiling == pytest.approx(ceiling)
    assert row.attained == pytest.approx(0.2 / ceiling)
    assert row.questions == 2
    assert "of an achievable" in row.line


def test_an_already_normalised_metric_prints_without_a_ceiling_clause() -> None:
    # nDCG's ceiling really is one, and an "of an achievable 1.000" clause beside it trains
    # a reader to skip the clause on the rows where it matters.
    (row,) = retrieval_summary({"nDCG@k": [Bounded(0.38, 1.0)]})
    assert "achievable" not in row.line and "0.380" in row.line


def test_a_value_above_its_own_ceiling_is_a_crash_and_not_a_number() -> None:
    # Otherwise the symptom is an attained share above one, printed as though it made
    # sense. Every ceiling function is deliberately loose for this reason: a sound bound
    # beats a tight one that can be exceeded.
    with pytest.raises(ValueError, match="exceeds its own ceiling"):
        Bounded(0.9, 0.5)


def test_questions_a_metric_cannot_speak_about_are_excluded_and_counted() -> None:
    # Not a number means the question had no correct set to score against, which is a
    # question the metric cannot speak about rather than a zero. Averaging it in as
    # zero would report a worse system; propagating it would report no system at all.
    rows = retrieval_summary({"recall@k": [Bounded(0.4, 0.5), Bounded(math.nan, math.nan)]})
    assert rows[0].questions == 1 and rows[0].mean == pytest.approx(0.4)
