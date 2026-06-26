"""Provisional judge calibration (SP8 task 3): two signals collected before any real human label
exists, each one clearly stamped with its own source, neither one licensing anything.

KAPPA HONESTY (binding, this repo's own documented prior failure: kappa 0.29 in prose versus 0.21
in the committed artifact). The ONLY number that may ever license a production deployment of the
judge is `judge.calibration.CalibrationReport.licensed`, computed from REAL human labels against
`AUTOMATION_BAR` (kappa >= 0.6) through its confidence interval's lower bound, routed through
`quality.gate.gate_on_lower_bound`. This module never imports that gate, `AUTOMATION_BAR`, or
anything named `licensed`: there is nothing here for a careless caller to accidentally compare
against the deployment bar, and `testing/tests/test_judge_provisional_honesty.py` proves that by
inspecting this module's own source, not by trusting this paragraph.

Two provisional sources, both computed here:

1. REGISTRY TRUTH AGREEMENT. `manufactured_cases` consumes SP7's registry contradictions
   (`corpus/registry/core.yaml`'s `conflict-daniel-contract` and `conflict-promo-price-north`, each
   a `winning_fact`/`losing_fact` pair). Every pair yields one true case, a hand authored template
   stating the winning fact, cited against ONLY the winning fact's own entity, so it is grounded by
   construction, and one plausible false case, the SAME template stating the losing fact instead,
   cited against that SAME winning only context, so it is ungrounded by construction. Zero new
   corruption logic invented: the false answer is a real registry fact, just the wrong one for this
   question, exactly what a model that read the wrong document would say.
   `registry_truth_agreement` compares the judge's verdicts against this constructed ground truth.
   This is plain agreement (accuracy), never Cohen's kappa: the ground truth here has no
   independent marginal distribution to correct for, unlike two human raters.

2. JUDGE VS JUDGE KAPPA. `judge_vs_judge_kappa` compares TWO judge contracts' verdicts on the same
   set, with no ground truth on either side. This measures whether two instruments agree with each
   other, chance corrected (`quality.stats.cohen_kappa`), never whether either one is right.

`provisional_calibration_artifact` reports both side by side, each stamped with its own `SOURCE`
constant, plus an explicit statement that neither licenses deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from corpus_tools.registry import Entity, Registry, load_registry

from judge.contract import JudgeContract
from quality.stats import cohen_kappa

CORE_REGISTRY = Path("corpus/registry/core.yaml")


@dataclass(frozen=True)
class ManufacturedCase:
    """One manufactured case: a question, the cited context (the winning fact's own entity, and
    ONLY that entity), an answer, and the ground truth that answer's construction guarantees
    (1 grounded / 0 ungrounded)."""

    case_id: str
    contradiction_id: str
    question: str
    context: str
    answer: str
    ground_truth: int


def _phrase_question(hint: str, fallback: str) -> str:
    text = hint.strip() or fallback
    text = text[0].upper() + text[1:]
    if not text.endswith(("?", ".", "!")):
        text += "?"
    return text


def _cited_chunk(entity: Entity, field: str, value: object) -> str:
    """The one fact a correctly retrieved chunk would cite, nothing else. The whole manufactured
    design rests on this string: the losing fact never rides along in it."""
    name = entity.fields.get("name", entity.id)
    return f"{name} ({entity.id}): {field}={value}"


def _answer_sentence(entity: Entity, field: str, value: object) -> str:
    """The one hand authored template every manufactured answer uses. The only thing that differs
    between a contradiction's true and false case is which fact (winning or losing) it states here;
    no corruption logic, negation, or perturbation is ever applied to a value."""
    label = field.replace("_", " ")
    name = entity.fields.get("name", entity.id)
    return f"According to {name}, {label} is {value}."


def manufactured_cases(reg: Registry | None = None) -> tuple[ManufacturedCase, ...]:
    """Every registry contradiction, walked in file order (deterministic), each yielding exactly
    one true and one false `ManufacturedCase`. `reg` defaults to the committed registry
    (`corpus/registry/core.yaml`); hermetic tests may pass a fixture registry instead."""
    if reg is None:
        reg = load_registry([CORE_REGISTRY])
    cases: list[ManufacturedCase] = []
    for c in reg.contradictions:
        winning_entity_id, _, winning_field = c.winning_fact.partition(":")
        losing_entity_id, _, losing_field = c.losing_fact.partition(":")
        winning_entity = reg.entity(winning_entity_id)
        losing_entity = reg.entity(losing_entity_id)
        winning_value = winning_entity.fields[winning_field]
        losing_value = losing_entity.fields[losing_field]
        question = _phrase_question(c.question_hint, c.id)
        context = _cited_chunk(winning_entity, winning_field, winning_value)
        cases.append(
            ManufacturedCase(
                case_id=f"manufactured-{c.id}-true",
                contradiction_id=c.id,
                question=question,
                context=context,
                answer=_answer_sentence(winning_entity, winning_field, winning_value),
                ground_truth=1,
            )
        )
        cases.append(
            ManufacturedCase(
                case_id=f"manufactured-{c.id}-false",
                contradiction_id=c.id,
                question=question,
                context=context,
                answer=_answer_sentence(losing_entity, losing_field, losing_value),
                ground_truth=0,
            )
        )
    return tuple(cases)


# ---- registry truth agreement: known ground truth, plain accuracy, never Cohen's kappa -------------


@dataclass(frozen=True)
class RegistryTruthRow:
    case_id: str
    ground_truth: int
    judge: int

    @property
    def agree(self) -> bool:
        return self.ground_truth == self.judge


@dataclass(frozen=True)
class RegistryTruthAgreement:
    """PROVISIONAL. Source: manufactured failures with ground truth known by construction (see the
    module docstring). Never human gold; never routed through the deployment gate (there is
    deliberately no `licensed` property here)."""

    SOURCE: ClassVar[str] = "registry_truth_manufactured_ground_truth_by_construction"

    contract: JudgeContract
    rows: tuple[RegistryTruthRow, ...]
    generated_at: datetime

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def agreement(self) -> float:
        if not self.n:
            return 0.0
        return sum(1 for r in self.rows if r.agree) / self.n

    def render(self) -> str:
        lines = [
            f"registry truth agreement  (source: {self.SOURCE})",
            f"judge contract: {self.contract.judge_model_id} / {self.contract.rubric_version} "
            f"(fp:{self.contract.fingerprint()[:8]})",
            f"generated_at: {self.generated_at.isoformat()}",
            f"n={self.n}  agreement={self.agreement:.0%}",
        ]
        for r in self.rows:
            mark = "ok " if r.agree else "MISS"
            lines.append(f"  {mark} {r.case_id:<32} ground_truth={r.ground_truth} judge={r.judge}")
        return "\n".join(lines)


def registry_truth_agreement(
    contract: JudgeContract,
    cases: list[ManufacturedCase],
    judge_labels: list[int],
    *,
    generated_at: datetime,
) -> RegistryTruthAgreement:
    """Compare a judge's verdicts against the manufactured cases' ground truth, known by
    construction, never by a second rater. `generated_at` is required, no wall clock fallback."""
    if len(cases) != len(judge_labels):
        raise ValueError("cases and judge_labels must be the same length")
    if not cases:
        raise ValueError("registry truth agreement needs at least one manufactured case")
    rows = tuple(
        RegistryTruthRow(case_id=c.case_id, ground_truth=c.ground_truth, judge=j)
        for c, j in zip(cases, judge_labels)
    )
    return RegistryTruthAgreement(contract=contract, rows=rows, generated_at=generated_at)


# ---- judge vs judge: Cohen's kappa between two judges, no ground truth on either side ---------------


@dataclass(frozen=True)
class JudgeVsJudgeRow:
    case_id: str
    label_a: int
    label_b: int

    @property
    def agree(self) -> bool:
        return self.label_a == self.label_b


@dataclass(frozen=True)
class JudgeVsJudgeAgreement:
    """PROVISIONAL. Source: two judge contracts' verdicts on the same set, no ground truth on
    either side at all. Never human gold; never routed through the deployment gate."""

    SOURCE: ClassVar[str] = "judge_vs_judge_no_ground_truth"

    contract_a: JudgeContract
    contract_b: JudgeContract
    rows: tuple[JudgeVsJudgeRow, ...]
    generated_at: datetime

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def kappa(self) -> float:
        """Cohen's kappa BETWEEN TWO JUDGES on the same set, chance corrected. Measures whether the
        two instruments agree with each other, never whether either one is right; contrast
        `judge.calibration.CalibrationReport.kappa`, which does compare against a reference (real
        human gold, once collected)."""
        return cohen_kappa([r.label_a for r in self.rows], [r.label_b for r in self.rows])

    @property
    def raw_agreement(self) -> float:
        if not self.n:
            return 0.0
        return sum(1 for r in self.rows if r.agree) / self.n

    def render(self) -> str:
        lines = [
            f"judge vs judge kappa  (source: {self.SOURCE})",
            f"judge A: {self.contract_a.judge_model_id} / {self.contract_a.rubric_version}",
            f"judge B: {self.contract_b.judge_model_id} / {self.contract_b.rubric_version}",
            f"generated_at: {self.generated_at.isoformat()}",
            f"n={self.n}  raw agreement={self.raw_agreement:.0%}  kappa={self.kappa:.2f}",
        ]
        for r in self.rows:
            mark = "ok " if r.agree else "MISS"
            lines.append(f"  {mark} {r.case_id:<32} a={r.label_a} b={r.label_b}")
        return "\n".join(lines)


def judge_vs_judge_kappa(
    contract_a: JudgeContract,
    contract_b: JudgeContract,
    case_ids: list[str],
    labels_a: list[int],
    labels_b: list[int],
    *,
    generated_at: datetime,
) -> JudgeVsJudgeAgreement:
    """Cohen's kappa between two judge contracts' labels on the same case set. No ground truth
    argument at all, on purpose: this is an agreement reading, not a correctness reading."""
    if not (len(case_ids) == len(labels_a) == len(labels_b)):
        raise ValueError("case_ids, labels_a and labels_b must be the same length")
    if not case_ids:
        raise ValueError("judge vs judge kappa needs at least one case")
    rows = tuple(
        JudgeVsJudgeRow(case_id=c, label_a=a, label_b=b) for c, a, b in zip(case_ids, labels_a, labels_b)
    )
    return JudgeVsJudgeAgreement(
        contract_a=contract_a, contract_b=contract_b, rows=rows, generated_at=generated_at
    )


# ---- one artifact, both numbers, each labeled by its real source, neither licensing anything --------


@dataclass(frozen=True)
class ProvisionalCalibrationArtifact:
    """Both provisional readings, side by side, each stamped with its OWN `SOURCE` constant, plus
    an explicit statement that neither licenses a production deployment.

    Deliberately carries no `licensed` property and no `gate_decision`: the deployment gate is
    `judge.calibration.CalibrationReport.licensed`, computed from real human labels, and nothing
    else. See the module docstring's KAPPA HONESTY paragraph and
    `testing/tests/test_judge_provisional_honesty.py` for the machine enforced version of that
    rule.
    """

    registry_truth: RegistryTruthAgreement
    judge_vs_judge: JudgeVsJudgeAgreement
    generated_at: datetime

    def render(self) -> str:
        lines = [
            "# Provisional judge calibration (registry truth + judge vs judge, NEITHER human gold)",
            f"generated_at: {self.generated_at.isoformat()}",
            "",
            "## 1. Registry truth agreement",
            self.registry_truth.render(),
            "",
            "## 2. Judge vs judge kappa",
            self.judge_vs_judge.render(),
            "",
            "## Honesty statement",
            (
                "This provisional artifact does not license a production deployment of this judge: "
                "neither the registry truth agreement above nor the judge vs judge kappa above may "
                "ever stand in for it. The production deployment gate is Cohen's kappa >= 0.6, its "
                "confidence interval's lower bound, against REAL human gold labels only "
                "(judge.calibration.CalibrationReport.licensed, D15). These two numbers are "
                "provisional signals collected before any human labeled set exists; read them as a "
                "sanity check on the judge's wiring, never as a calibration."
            ),
        ]
        return "\n".join(lines)


def provisional_calibration_artifact(
    registry_truth: RegistryTruthAgreement,
    judge_vs_judge: JudgeVsJudgeAgreement,
    *,
    generated_at: datetime,
) -> ProvisionalCalibrationArtifact:
    return ProvisionalCalibrationArtifact(
        registry_truth=registry_truth, judge_vs_judge=judge_vs_judge, generated_at=generated_at
    )


__all__ = [
    "CORE_REGISTRY",
    "JudgeVsJudgeAgreement",
    "JudgeVsJudgeRow",
    "ManufacturedCase",
    "ProvisionalCalibrationArtifact",
    "RegistryTruthAgreement",
    "RegistryTruthRow",
    "judge_vs_judge_kappa",
    "manufactured_cases",
    "provisional_calibration_artifact",
    "registry_truth_agreement",
]
