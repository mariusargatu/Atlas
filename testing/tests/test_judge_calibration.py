"""P6 judge calibration: the instrument has a serial number, and it must be checked.

These cover the calibrated-judge machinery end to end on the hermetic REPLAY lane: the judge
contract (a versioned triple), the gateway-routed judge (a model grading a model on tape), the
before/after agreement study (κ rises above the bar after one documented rubric correction), the
order-swap position-bias gate, and the panel's disagreement signal.
"""
from __future__ import annotations

import tempfile

import pytest

from replay.cassette_store import seed_cassette
from replay.gateway import GatewayChatModel

from evals.datasets.judge_calibration import (
    CALIBRATION,
    case_ids,
    corrected_labels,
    human_labels,
    naive_labels,
)
from evals.judge.calibration import calibrate, order_swap_flip_rate
from evals.judge.contract import JudgeContract
from evals.judge.llm_judge import LlmJudgeGrader, judge_label, order_swap
from evals.judge.panel import panel_vote
from evals.judge.rubric import RUBRIC_V1, RUBRIC_V2, compare_prompt, prompt, template_hash
from evals.evalkit.graders import GradeContext
from tracing import InMemoryTracer

_JUDGE_ID = "gpt-judge"


# ---- the judge contract: change any field and you have a new instrument ----

def test_contract_fingerprint_is_stable_and_field_sensitive():
    base = JudgeContract(_JUDGE_ID, RUBRIC_V2.version, template_hash(RUBRIC_V2))
    same = JudgeContract(_JUDGE_ID, RUBRIC_V2.version, template_hash(RUBRIC_V2))
    assert base.fingerprint() == same.fingerprint()
    # a rubric swap, a model swap, or a template edit each void the calibration
    assert base.fingerprint() != JudgeContract(_JUDGE_ID, RUBRIC_V1.version, template_hash(RUBRIC_V2)).fingerprint()
    assert base.fingerprint() != JudgeContract("claude-judge", RUBRIC_V2.version, template_hash(RUBRIC_V2)).fingerprint()
    assert template_hash(RUBRIC_V1) != template_hash(RUBRIC_V2)


# ---- the judge is a gateway call: a model grading a model, replayed ----

def test_judge_routes_through_the_gateway_and_parses_a_verdict():
    with tempfile.TemporaryDirectory(prefix="judge-") as cdir:
        case = CALIBRATION[0]
        seed_cassette(cdir, prompt(RUBRIC_V2, case.question, case.answer),
                      {"content": "PASS", "tool_calls": []}, _JUDGE_ID)
        gateway = GatewayChatModel(model_id=_JUDGE_ID, cassette_dir=cdir, mode="replay")
        assert judge_label(gateway, RUBRIC_V2, case.question, case.answer) == 1


def test_an_unparseable_verdict_fails_closed():
    with tempfile.TemporaryDirectory(prefix="judge-") as cdir:
        case = CALIBRATION[0]
        seed_cassette(cdir, prompt(RUBRIC_V2, case.question, case.answer),
                      {"content": "well, it depends", "tool_calls": []}, _JUDGE_ID)
        gateway = GatewayChatModel(model_id=_JUDGE_ID, cassette_dir=cdir, mode="replay")
        # not a clear PASS, so the judge does not vouch for the answer
        assert judge_label(gateway, RUBRIC_V2, case.question, case.answer) == 0


# ---- the before/after study: one documented correction clears the bar ----

def test_the_naive_judge_fails_the_bar_and_the_corrected_judge_clears_it():
    ids, humans = case_ids(), human_labels()
    naive = calibrate(
        JudgeContract(_JUDGE_ID, RUBRIC_V1.version, template_hash(RUBRIC_V1)),
        ids, humans, naive_labels(),
    )
    corrected = calibrate(
        JudgeContract(_JUDGE_ID, RUBRIC_V2.version, template_hash(RUBRIC_V2)),
        ids, humans, corrected_labels(),
    )
    # the lying judge: barely better than chance, in the Landis-Koch "fair" band, fails 0.6
    assert 0.2 <= naive.kappa < 0.45
    assert not naive.licensed
    # after the correction: almost-perfect agreement (Landis-Koch 0.81-1.0), clears the automation bar
    assert corrected.kappa >= 0.6
    assert corrected.licensed
    # the correction is real movement, not noise on the raw number
    assert corrected.kappa - naive.kappa > 0.3
    # pin the headline numbers the docstrings and the artifact quote, so the data and the prose
    # cannot drift apart again (a stale 0.71 once slipped through these loose bands). Both kappa AND
    # raw agreement are pinned, because the study prose names the 64% raw figure too.
    assert round(naive.kappa, 2) == 0.29
    assert round(corrected.kappa, 2) == 0.85
    assert round(naive.raw_agreement, 2) == 0.64
    assert round(corrected.raw_agreement, 2) == 0.93


def test_raw_agreement_hides_what_kappa_reveals():
    ids, humans = case_ids(), human_labels()
    naive = calibrate(
        JudgeContract(_JUDGE_ID, RUBRIC_V1.version, template_hash(RUBRIC_V1)),
        ids, humans, naive_labels(),
    )
    # raw agreement looks far healthier than the chance-corrected number
    assert naive.raw_agreement > naive.kappa + 0.25


# ---- bias gates ----

def test_order_swap_flip_rate_counts_only_inconsistent_pairs():
    # (winner_ab, winner_ba): equal means consistent, unequal means an order artifact
    assert order_swap_flip_rate([(0, 0), (1, 1)]) == 0.0
    assert order_swap_flip_rate([(0, 1), (1, 1)]) == 0.5
    assert order_swap_flip_rate([]) == 0.0


def test_panel_reads_the_disagreement_and_ties_fail_closed():
    assert panel_vote([1, 1, 1]).label == 1
    assert panel_vote([1, 1, 1]).disagreed is False
    split = panel_vote([1, 0, 1])
    assert split.label == 1 and split.disagreed is True   # majority pass, but flag for a human
    tie = panel_vote([1, 0])
    assert tie.label == 0 and tie.disagreed is True        # a split panel is not evidence of good


def test_panel_needs_at_least_one_judge():
    with pytest.raises(ValueError):
        panel_vote([])


# ---- the gateway-routed order-swap detects a real position flip ----

def _seed_compare(cdir, rubric, q, first, second, pick):
    seed_cassette(cdir, compare_prompt(rubric, q, first, second), {"content": pick, "tool_calls": []}, _JUDGE_ID)


def test_order_swap_catches_a_judge_that_always_picks_the_first_shown():
    q, a, b = "Any cap?", "Your plan is uncapped.", "There is no data limit on your plan."
    with tempfile.TemporaryDirectory(prefix="judge-") as cdir:
        _seed_compare(cdir, RUBRIC_V2, q, a, b, "A")   # (a,b) -> picks first (a)
        _seed_compare(cdir, RUBRIC_V2, q, b, a, "A")   # (b,a) -> picks first (b): a flip
        gateway = GatewayChatModel(model_id=_JUDGE_ID, cassette_dir=cdir, mode="replay")
        ab, ba = order_swap(gateway, RUBRIC_V2, q, a, b)
    assert ab != ba   # the winner flipped with the order: a position artifact, not a preference


# ---- the judge as a grader in the eval stack ----

def test_llm_judge_grader_reads_the_question_from_the_trace_and_grades():
    case = CALIBRATION[9]  # the cold-open answer
    tracer = InMemoryTracer()
    tracer.open("turn", "turn", input=case.question)
    ctx = GradeContext(customer_id=case.customer_id, final_response=case.answer, trace=tuple(tracer.spans))
    with tempfile.TemporaryDirectory(prefix="judge-") as cdir:
        seed_cassette(cdir, prompt(RUBRIC_V2, case.question, case.answer),
                      {"content": "FAIL", "tool_calls": []}, _JUDGE_ID)
        gateway = GatewayChatModel(model_id=_JUDGE_ID, cassette_dir=cdir, mode="replay")
        verdict = LlmJudgeGrader(gateway, RUBRIC_V2).grade(ctx)
    assert verdict.passed is False and verdict.grader == "llm-judge"


# ---- report rendering and input validation ----

def test_calibration_report_renders_the_contract_and_verdict():
    ids, humans = case_ids(), human_labels()
    report = calibrate(
        JudgeContract(_JUDGE_ID, RUBRIC_V1.version, template_hash(RUBRIC_V1)),
        ids, humans, naive_labels(),
    )
    text = report.render()
    assert _JUDGE_ID in text and "NOT licensed" in text and "kappa" in text


def test_calibrate_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        calibrate(
            JudgeContract(_JUDGE_ID, RUBRIC_V1.version, template_hash(RUBRIC_V1)),
            ["only-one-id"], [1, 0], [1],
        )
