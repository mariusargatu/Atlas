"""Calibrated-judge machinery: the instrument, and how to check it.

The judge is a measuring instrument with a serial number (``JudgeContract``), a versioned rubric
(``rubric``), a gateway-routed call so a model grades a model on tape (``llm_judge``), an
agreement check against human labels the chance-corrected way (``calibration``), and a disjoint-
family panel whose disagreement is the signal (``panel``). Metric SELECTION lives with the metrics
half of the article; this is the half that grades the grader.
"""
from __future__ import annotations

from evals.judge.calibration import (
    AUTOMATION_BAR,
    AgreementRow,
    CalibrationReport,
    calibrate,
    order_swap_flip_rate,
)
from evals.judge.contract import JudgeContract
from evals.judge.llm_judge import LlmJudgeGrader, judge_label, order_swap
from evals.judge.panel import PanelVote, panel_vote
from evals.judge.rubric import RUBRIC_V1, RUBRIC_V2, Rubric, prompt, template_hash

__all__ = [
    "AUTOMATION_BAR",
    "AgreementRow",
    "CalibrationReport",
    "JudgeContract",
    "LlmJudgeGrader",
    "PanelVote",
    "RUBRIC_V1",
    "RUBRIC_V2",
    "Rubric",
    "calibrate",
    "judge_label",
    "order_swap",
    "order_swap_flip_rate",
    "panel_vote",
    "prompt",
    "template_hash",
]
