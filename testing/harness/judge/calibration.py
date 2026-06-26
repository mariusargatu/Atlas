"""Calibration: measure the judge against the human labelled set, the chance corrected way.

Reports Cohen's kappa, not raw percent agreement, because raw agreement gives chance a free ride (a
judge that passes everything can still score well on it). The automation bar (kappa >= 0.6, Landis &
Koch 1977) gates on the interval's lower bound, not the point estimate, so the calibration set is
sized until the floor clears the bar too, not just the point, and the gate itself routes through
`quality.gate.gate_on_lower_bound`, the SAME rule a release uses, never a second hand rolled copy of
it (the digest's own named fix over the pre rewrite module this absorbs).

D15 also names prevalence and Gwet's AC1 as numbers a calibration report carries alongside kappa: AC1
is what stays honest exactly where kappa's own chance model is famous for punishing a genuinely good
judge (the "kappa paradox", extreme label prevalence), and the prevalence index is the number that
explains WHY the two can diverge. All four (kappa, raw agreement, AC1, prevalence) are read together;
none of them replaces another, and only ONE of them ever licenses deployment: kappa's lower bound
against HUMAN gold (the plan's own "kappa honesty" global constraint, this repo's documented prior
failure of exactly this substitution).

KAPPA HONESTY: this module computes a `CalibrationReport` from whatever `human_labels` it is given.
It has no opinion on where those labels came from, and NEITHER does `licensed`: a report built from
registry truth or judge vs judge agreement (SP8 task 3's provisional sources) computes a `licensed`
value exactly like a report built from real human gold, because the arithmetic cannot tell the
difference. The caller is the one place that honesty is enforced: only a report whose `human` column
is real human adjudication may be read as licensing a production deployment (D15). Task 3's
provisional calibration artifact is the module that carries this labelling discipline explicitly;
this module supplies the shared arithmetic both readings are built from.

Absorbed from the pre rewrite `evals/judge/calibration.py` (SP8 task 2, per the planning digest:
"keep, with one real fix... route through `gate_on_lower_bound` once it lives at
`testing/harness/quality/gate.py`, not keep a second hand written copy of the same rule").
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from judge.contract import JudgeContract
from quality.gate import GateDecision, GateVerdict, gate_on_lower_bound
from quality.stats import cohen_kappa, cohen_kappa_interval, gwet_ac1

AUTOMATION_BAR = 0.6  # Cohen's kappa floor that licenses automating a metric (D15, Landis & Koch 1977)

# The CI width a kappa reading must clear before the lower bound gate calls it PASS/FAIL rather than
# QUARANTINE ("too wide to call"). Kappa's own interval lives on a [-1, 1] scale, twice the [0, 1]
# scale a pass rate's Wilson interval uses, so this budget is set wider than `evals/benchmark/
# study.py`'s own 0.20 release gate budget for the same reason that budget is 0.20 and not narrower:
# proportionate to the scale being measured, not copied verbatim from a different metric. 0.5 is
# SP8's own reasoned choice, not a doubling of 0.20 (which would be 0.40): the HLD does not pin a
# number for this (D15 names kappa/agreement/AC1/prevalence, not a variance budget), so this is SP8's
# own call to make and record, the same "no number pinned, SP8 decides and documents" precedent the
# plan already applies to the judge fail rate alert threshold (task 4).
KAPPA_VARIANCE_BUDGET = 0.5


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
    """A judge's agreement with a labelled set, stamped with the instrument that produced it.

    `generated_at` comes from the caller's injected clock (`determinism.sources.FrozenClock` in the
    hermetic lane, a real clock outside it), never `datetime.now()` read inside this module: the
    determinism global constraint applies to this report exactly as it does to the label writer.
    """

    contract: JudgeContract
    rows: tuple[AgreementRow, ...]
    generated_at: datetime
    bar: float = AUTOMATION_BAR
    variance_budget: float = KAPPA_VARIANCE_BUDGET

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def kappa(self) -> float:
        return cohen_kappa([r.human for r in self.rows], [r.judge for r in self.rows])

    @property
    def kappa_ci(self) -> tuple[float, float, float]:
        """The kappa point estimate with its 95% confidence interval, as (point, lo, hi)."""
        return cohen_kappa_interval([r.human for r in self.rows], [r.judge for r in self.rows])

    @property
    def raw_agreement(self) -> float:
        """The flattering number, reported only to show the gap kappa reveals."""
        return sum(1 for r in self.rows if r.agree) / self.n if self.n else 0.0

    @property
    def ac1(self) -> float:
        """Gwet's AC1 (D15): the prevalence robust companion to kappa, read alongside it, never
        instead of it."""
        return gwet_ac1([r.human for r in self.rows], [r.judge for r in self.rows])

    @property
    def prevalence(self) -> float:
        """The prevalence index (Byrt et al. 1993): how skewed the pass/fail calls are toward one
        category, |agreed pass - agreed fail| / n. Kappa is sensitive to exactly this skew (the
        "kappa paradox"); this is the number that explains why kappa and AC1 can read apart on the
        same data, not a licensing quantity of its own (D15 names it as a reported companion, never
        a bar of its own)."""
        if not self.n:
            return 0.0
        both_pass = sum(1 for r in self.rows if r.human == 1 and r.judge == 1)
        both_fail = sum(1 for r in self.rows if r.human == 0 and r.judge == 0)
        return abs(both_pass - both_fail) / self.n

    @property
    def gate_decision(self) -> GateDecision:
        """The kappa lower bound gate, routed through the SAME rule a release uses
        (`quality.gate.gate_on_lower_bound`), never a hand rolled `kappa_ci[1] >= bar` comparison
        (the digest's own named fix over the pre rewrite module this report supersedes)."""
        _, lo, hi = self.kappa_ci
        return gate_on_lower_bound((lo, hi), threshold=self.bar, variance_budget=self.variance_budget)

    @property
    def licensed(self) -> bool:
        """Whether this judge clears the bar to automate the metric. Reads the SHARED gate's
        verdict, not a second hand rolled comparison: only PASS licenses. A QUARANTINE (the interval
        too wide to call) is not a licence either, the same fail closed reading a release gate gives
        an unproven claim, whichever side of the bar its point estimate happens to sit on."""
        return self.gate_decision.verdict is GateVerdict.PASS

    def render(self) -> str:
        gate = self.gate_decision
        verdict = "LICENSED to automate" if self.licensed else "NOT licensed, keep manual / fix the rubric"
        lines = [
            f"judge contract: {self.contract.judge_model_id} / {self.contract.rubric_version} "
            f"/ tmpl:{self.contract.prompt_template_hash[:8]} (fp:{self.contract.fingerprint()[:8]})",
            f"generated_at: {self.generated_at.isoformat()}",
            f"n={self.n}  raw agreement={self.raw_agreement:.0%}  Cohen's kappa={self.kappa:.2f} "
            f"95% CI [{self.kappa_ci[1]:.2f}, {self.kappa_ci[2]:.2f}]  AC1={self.ac1:.2f}  "
            f"prevalence={self.prevalence:.2f}  bar={self.bar:.2f}  -> {verdict}",
            f"gate: {gate.verdict.value} ({gate.reason})",
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
    generated_at: datetime,
    bar: float = AUTOMATION_BAR,
    variance_budget: float = KAPPA_VARIANCE_BUDGET,
) -> CalibrationReport:
    """Build the agreement report for one judge contract against a labelled set.

    `generated_at` is required and has no wall clock fallback: the caller supplies it (the frozen
    clock in every hermetic test, a real clock in a live/burst run), so this function can never
    silently reach for `datetime.now()`.
    """
    if not (len(case_ids) == len(human_labels) == len(judge_labels)):
        raise ValueError("case_ids, human_labels and judge_labels must be the same length")
    if not case_ids:
        raise ValueError("a calibration needs at least one case")
    rows = tuple(
        AgreementRow(case_id=c, human=h, judge=j)
        for c, h, j in zip(case_ids, human_labels, judge_labels)
    )
    return CalibrationReport(
        contract=contract, rows=rows, generated_at=generated_at, bar=bar, variance_budget=variance_budget
    )


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


__all__ = [
    "AUTOMATION_BAR",
    "KAPPA_VARIANCE_BUDGET",
    "AgreementRow",
    "CalibrationReport",
    "calibrate",
    "order_swap_flip_rate",
]
