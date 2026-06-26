"""Calibration: measure the measurer against human labels, the chance corrected way.

A judge you have not checked against a known reference is not a metric, it is a vibe with a decimal
point. Calibration is the check: run the judge over the human labelled set, line the two label
columns up, and compute how much they agree once luck is subtracted. The honest number is Cohen's
kappa, never raw percent agreement, because raw agreement gives chance a free ride and a judge that
calls everything "pass" scores 80% while contributing nothing.

The bar is a licensing threshold: kappa >= 0.6 (the moderate/substantial boundary on Landis & Koch,
1977) licenses automating the metric. Below it, fix the rubric or keep the check manual. The report
carries the judge contract, so an agreement number is never read apart from the instrument that
earned it, and the bar gates on the point estimate here while a confidence interval on kappa is the
statistics article's job.
"""
from __future__ import annotations

from dataclasses import dataclass

from evals.judge.contract import JudgeContract
from evals.stats import cohen_kappa

AUTOMATION_BAR = 0.6  # Cohen's kappa floor that licenses automating a metric


@dataclass(frozen=True)
class AgreementRow:
    """One case: what the human said, what the judge said, and whether they landed together."""

    case_id: str
    human: int
    judge: int

    @property
    def agree(self) -> bool:
        return self.human == self.judge


@dataclass(frozen=True)
class CalibrationReport:
    """A judge's agreement with the human labelled set, stamped with the instrument that produced it."""

    contract: JudgeContract
    rows: tuple[AgreementRow, ...]
    bar: float = AUTOMATION_BAR

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def kappa(self) -> float:
        return cohen_kappa([r.human for r in self.rows], [r.judge for r in self.rows])

    @property
    def raw_agreement(self) -> float:
        """The flattering number, reported only to show the gap kappa reveals."""
        return sum(1 for r in self.rows if r.agree) / self.n if self.n else 0.0

    @property
    def licensed(self) -> bool:
        """Whether this judge clears the bar to automate the metric."""
        return self.kappa >= self.bar

    def render(self) -> str:
        verdict = "LICENSED to automate" if self.licensed else "NOT licensed, keep manual / fix the rubric"
        lines = [
            f"judge contract: {self.contract.judge_model_id} / {self.contract.rubric_version} "
            f"/ tmpl:{self.contract.prompt_template_hash[:8]} (fp:{self.contract.fingerprint()[:8]})",
            f"n={self.n}  raw agreement={self.raw_agreement:.0%}  Cohen's kappa={self.kappa:.2f}  "
            f"bar={self.bar:.2f}  -> {verdict}",
        ]
        for r in self.rows:
            mark = "ok " if r.agree else "MISS"
            lines.append(f"  {mark} {r.case_id:<28} human={r.human} judge={r.judge}")
        return "\n".join(lines)


def calibrate(
    contract: JudgeContract,
    case_ids: list[str],
    human_labels: list[int],
    judge_labels: list[int],
    *,
    bar: float = AUTOMATION_BAR,
) -> CalibrationReport:
    """Build the agreement report for one judge contract against the human labels."""
    if not (len(case_ids) == len(human_labels) == len(judge_labels)):
        raise ValueError("case_ids, human_labels and judge_labels must be the same length")
    if not case_ids:
        raise ValueError("a calibration needs at least one case")
    rows = tuple(
        AgreementRow(case_id=c, human=h, judge=j)
        for c, h, j in zip(case_ids, human_labels, judge_labels)
    )
    return CalibrationReport(contract=contract, rows=rows, bar=bar)


def order_swap_flip_rate(pairs: list[tuple[int, int]]) -> float:
    """Position bias gate: the fraction of pairwise comparisons whose winner flips when the order is
    swapped. ``pairs`` is ``[(winner_ab, winner_ba), ...]`` from ``llm_judge.order_swap``. A flip
    means the judge had a reading order artifact, not a preference. Count those cases as ties and a
    flag. A flip rate above threshold says position bias is live and the comparison is not trustworthy.
    """
    if not pairs:
        return 0.0
    flips = sum(1 for ab, ba in pairs if ab != ba)
    return flips / len(pairs)


__all__ = ["AUTOMATION_BAR", "AgreementRow", "CalibrationReport", "calibrate", "order_swap_flip_rate"]
