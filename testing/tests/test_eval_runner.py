"""The eval harness, tested on the REPLAY lane (the second machine).

This is the eval harness testing ITSELF: the same runner the nightly LIVE lane uses, run on the
pinned gateway so the test is deterministic and has zero egress. On REPLAY every trial is identical,
so a rate is 0 or 1. That is the point, it proves the driver, the grader stack, the planner seam,
and the aggregation WIRING without a live model. The concrete graders (oracle rules, the judge)
belong to later articles. Here the grader is the trivial ``PredicateGrader``. The report itself is
held to the statistics article's law: no rate ships without its interval, and the gate reads the floor.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from evals.evalkit.case import EvalCase
from evals.evalkit.graders import Composite, GradeContext, PredicateGrader, Verdict
from evals.evalkit.planner import StaticPlanner
from evals.evalkit.report import build_report, run_suite
from evals.evalkit.runner import CaseResult, TrialResult, _drive, _trial_passed, run_case
from evals.gate import GateVerdict
from evals.scaffold import build_replay_graph
from evals.stats import wilson_interval

_MODEL_ID = "claude-test"


def _build_factory(cassette_dir):
    """A REPLAY graph-build hook for the runner: fresh graph + tracer per trial, model pinned."""
    return lambda: build_replay_graph(cassette_dir, model_id=_MODEL_ID)


def _seed(seed_cassette, cassette_dir, utterance: str, answer: str) -> None:
    seed_cassette(cassette_dir, [HumanMessage(utterance)], {"content": answer, "tool_calls": []}, _MODEL_ID)


def _ctx(text: str) -> GradeContext:
    return GradeContext(customer_id="cust_current", final_response=text, trace=())


_SHIPPED = PredicateGrader("shipped", lambda ctx: bool(ctx.final_response.strip()))


# ---- the runner drives a real turn and aggregates (REPLAY: rate is 0 or 1) ----

@pytest.mark.asyncio
async def test_run_case_drives_a_real_turn_and_reports_a_full_pass(tmp_path, seed_cassette):
    utterance = "Tell me about my plan"
    _seed(seed_cassette, tmp_path, utterance, "Your current plan is flexible; happy to help further.")
    case = EvalCase(id="benign", turns=(utterance,), customer_id="cust_current")

    result = await run_case(case, _build_factory(tmp_path), [_SHIPPED], k=5)

    assert isinstance(result, CaseResult)
    assert result.passes == 5 and result.k == 5 and result.rate == 1.0  # REPLAY: every trial identical
    assert all(isinstance(t, TrialResult) and t.passed for t in result.trials)


@pytest.mark.asyncio
async def test_run_case_rate_is_zero_when_the_grader_fails_every_trial(tmp_path, seed_cassette):
    utterance = "Tell me about my plan"
    _seed(seed_cassette, tmp_path, utterance, "Your current plan is flexible.")
    case = EvalCase(id="never", turns=(utterance,), customer_id="cust_current")
    never = PredicateGrader("never", lambda ctx: False)

    result = await run_case(case, _build_factory(tmp_path), [never], k=4)

    assert result.passes == 0 and result.rate == 0.0  # deterministic on REPLAY, never flaky


@pytest.mark.asyncio
async def test_run_case_rejects_zero_k(tmp_path):
    case = EvalCase(id="x", turns=("hi",), customer_id="cust_current")
    with pytest.raises(ValueError):
        await run_case(case, _build_factory(tmp_path), [_SHIPPED], k=0)


@pytest.mark.asyncio
async def test_drive_runs_every_turn_on_one_thread():
    # A case with multiple turns is a real conversation: all turns share one thread_id, so under the
    # checkpointer turn 2 resumes turn 1's state instead of starting cold.
    seen = []

    class _Graph:
        async def ainvoke(self, state, config):
            seen.append(config["configurable"]["thread_id"])
            return {"final_response": "ok"}

    case = EvalCase(id="multi", turns=("first", "second", "third"), customer_id="cust_current")
    await _drive(_Graph(), case, thread_id="multi-trial0")
    assert seen == ["multi-trial0", "multi-trial0", "multi-trial0"]


# ---- the grader stack (machinery only, concrete graders arrive with later articles) ----

def test_predicate_grader_reports_pass_and_fail():
    assert PredicateGrader("p", lambda ctx: True).grade(_ctx("x")).passed
    assert not PredicateGrader("p", lambda ctx: False).grade(_ctx("x")).passed


def test_composite_short_circuits_at_the_first_failure():
    calls = []

    class _Spy:
        name = "spy"

        def grade(self, ctx):
            calls.append("spy")
            return Verdict("spy", passed=True, reason="")

    failing = PredicateGrader("fail", lambda ctx: False)
    verdicts = Composite([failing, _Spy()]).grade(_ctx("x"))
    assert len(verdicts) == 1 and not verdicts[0].passed and calls == []  # spy never ran


def test_trial_passes_only_when_a_grader_ran_and_none_failed():
    assert _trial_passed([Verdict("a", True, ""), Verdict("b", True, "")])
    assert not _trial_passed([Verdict("a", True, ""), Verdict("b", False, "")])
    assert not _trial_passed([])  # no grader ran -> not a free pass


# ---- run_suite drives many cases on REPLAY and aggregates into one report ----

@pytest.mark.asyncio
async def test_run_suite_aggregates_many_cases_into_one_report(tmp_path, seed_cassette):
    _seed(seed_cassette, tmp_path, "Tell me about my plan", "Your current plan is flexible.")
    _seed(seed_cassette, tmp_path, "What is a data cap?", "A data cap is a monthly usage limit.")
    cases = [
        EvalCase(id="plan", turns=("Tell me about my plan",), customer_id="cust_current"),
        EvalCase(id="datacap", turns=("What is a data cap?",), customer_id="cust_current"),
    ]

    report = await run_suite(cases, _build_factory(tmp_path), [_SHIPPED], k=3)

    assert report.total_trials == 6 and report.total_passes == 6 and report.overall_rate == 1.0
    assert {c.case_id for c in report.cases} == {"plan", "datacap"}


@pytest.mark.asyncio
async def test_run_suite_resolves_per_case_graders_from_a_registry(tmp_path, seed_cassette):
    # A {name: Grader} registry: each case is graded by ONLY the grader(s) it declares, so a
    # mixed risk suite is expressible through run_suite (not just a single uniform grader list).
    _seed(seed_cassette, tmp_path, "Tell me about my plan", "Your current plan is flexible.")
    _seed(seed_cassette, tmp_path, "What is a data cap?", "")  # empty -> the "shipped" grader fails
    registry = {"shipped": _SHIPPED}
    cases = [
        EvalCase(id="plan", turns=("Tell me about my plan",), customer_id="cust_current", graders=("shipped",)),
        EvalCase(id="empty", turns=("What is a data cap?",), customer_id="cust_current", graders=("shipped",)),
    ]

    report = await run_suite(cases, _build_factory(tmp_path), registry, k=2)

    by_id = {c.case_id: c for c in report.cases}
    assert by_id["plan"].passes == 2          # graded by its declared rule, and shipped text
    assert by_id["empty"].passes == 0         # graded by the same rule, no text shipped


# ---- the three agent harness: planner (designs) -> generator (graph) -> evaluator (graders) ----

def test_static_planner_returns_its_fixed_case_set():
    cases = (EvalCase(id="a", turns=("hi",), customer_id="cust_current"),)
    assert StaticPlanner(cases).plan() == cases


@pytest.mark.asyncio
async def test_three_roles_stay_separate_planner_drives_generator_graded_by_evaluator(tmp_path, seed_cassette):
    # The planner designs the tasks. The generator (the Atlas graph) produces the runs. The
    # evaluator (the grader stack) grades them. Three roles, never one agent marking its own exam.
    _seed(seed_cassette, tmp_path, "Tell me about my plan", "Your current plan is flexible.")
    planner = StaticPlanner([EvalCase(id="plan", turns=("Tell me about my plan",), customer_id="cust_current")])

    report = await run_suite(planner.plan(), _build_factory(tmp_path), [_SHIPPED], k=2)

    assert report.total_trials == 2 and report.total_passes == 2
    assert report.cases[0].case_id == "plan"


# ---- aggregation with synthetic results (mixed rates the REPLAY lane can't produce) ----

def _synthetic_case(case_id: str, passes: int, k: int) -> CaseResult:
    trials = tuple(
        TrialResult(i, passed=i < passes, verdicts=(Verdict("p", i < passes, ""),)) for i in range(k)
    )
    return CaseResult(case_id, passes, k, trials)


def test_report_aggregates_rate_across_cases():
    report = build_report([_synthetic_case("a", 7, 10), _synthetic_case("b", 9, 10)])
    assert report.total_passes == 16 and report.total_trials == 20 and report.overall_rate == 0.8


def test_report_serializes_to_json_friendly_dict():
    report = build_report([_synthetic_case("a", 5, 10)])
    body = report.as_dict()
    assert body["overall"]["trials"] == 10 and body["overall"]["rate"] == 0.5
    assert body["cases"][0]["id"] == "a" and body["cases"][0]["rate"] == 0.5


# ---- no bare point estimates: every rate the report ships carries its interval ----


def test_every_rate_ships_with_its_wilson_interval():
    report = build_report([_synthetic_case("a", 7, 10), _synthetic_case("b", 9, 10)])
    body = report.as_dict()
    assert body["overall"]["ci95"] == list(wilson_interval(16, 20))
    assert body["cases"][0]["ci95"] == list(wilson_interval(7, 10))
    assert body["cases"][1]["ci95"] == list(wilson_interval(9, 10))


def test_reporter_lint_no_dict_carries_a_rate_without_an_interval():
    # The reporting law as a meta test (the analog of "no silent caps"): walk everything
    # the report serializes, and any node that quotes a rate must quote its uncertainty.
    # A future field added to the trend row cannot silently ship a bare point estimate.
    def walk(node):
        if isinstance(node, dict):
            if "rate" in node:
                lo, hi = node["ci95"]
                assert 0.0 <= lo <= node["rate"] <= hi <= 1.0
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(build_report([_synthetic_case("a", 7, 10), _synthetic_case("b", 0, 5)]).as_dict())


def test_empty_report_admits_it_knows_nothing():
    # Zero trials is the widest interval, never a confident 0% with false certainty.
    body = build_report([]).as_dict()
    assert body["overall"]["rate"] == 0.0
    assert body["overall"]["ci95"] == [0.0, 1.0]


def test_render_header_carries_the_interval():
    out = build_report([_synthetic_case("a", 7, 10), _synthetic_case("b", 9, 10)]).render()
    lo, hi = wilson_interval(16, 20)
    assert f"Wilson 95% CI [{lo:.3f}, {hi:.3f}]" in out


# ---- the report gates on the floor, never the point ----


def test_report_gates_on_the_lower_bound_not_the_point():
    # 16/20 is a 0.80 point with a floor near 0.58: it must NOT clear a 0.75 bar.
    report = build_report([_synthetic_case("a", 7, 10), _synthetic_case("b", 9, 10)])
    decision = report.gate(threshold=0.75, variance_budget=0.5)
    assert decision.verdict is GateVerdict.FAIL
    assert report.overall_rate >= 0.75  # the point clears, but the gate still holds the release


def test_report_gate_passes_when_the_floor_clears():
    report = build_report([_synthetic_case("a", 96, 100), _synthetic_case("b", 98, 100)])
    assert report.gate(threshold=0.90, variance_budget=0.10).verdict is GateVerdict.PASS


def test_report_gate_quarantines_an_interval_wider_than_the_budget():
    report = build_report([_synthetic_case("a", 4, 5)])  # n=5: honest width is huge
    assert report.gate(threshold=0.5, variance_budget=0.2).verdict is GateVerdict.QUARANTINE


# ---- the readable surface: name/risk flow through, render() reads as outcomes ----

@pytest.mark.asyncio
async def test_case_name_and_risk_flow_into_the_result(tmp_path, seed_cassette):
    _seed(seed_cassette, tmp_path, "Tell me about my plan", "Your current plan is flexible.")
    case = EvalCase(
        id="c1", turns=("Tell me about my plan",), customer_id="cust_current",
        name="customer asks about their plan", risk="answer-accuracy",
    )
    result = await run_case(case, _build_factory(tmp_path), [_SHIPPED], k=2)
    assert result.name == "customer asks about their plan" and result.risk == "answer-accuracy"


def _result_with_reason(case_id, passes, k, risk, reason):
    trials = tuple(
        TrialResult(i, i < passes, (Verdict("g", i < passes, "" if i < passes else reason),))
        for i in range(k)
    )
    return CaseResult(case_id, passes, k, trials, name=case_id, risk=risk)


def test_render_labels_pass_fail_and_flaky_with_the_failure_reason():
    report = build_report([
        _result_with_reason("ok", 5, 5, "fee-claim-safety", ""),
        _result_with_reason("broken", 0, 4, "data-isolation", "leaked another customer's name"),
        _result_with_reason("coinflip", 3, 5, "answer-accuracy", "answer contradicts the account"),
    ])
    out = report.render()
    assert "PASS  fee-claim-safety" in out
    assert "FAIL  data-isolation" in out and "leaked another customer's name" in out
    assert "FLAKY answer-accuracy" in out and "contradicts the account" in out


def test_first_failure_reason_surfaces_the_red_line():
    c = _result_with_reason("x", 0, 2, "r", "the reason an SDET reads")
    assert c.first_failure_reason() == "the reason an SDET reads"
    passing = _result_with_reason("y", 2, 2, "r", "unused")  # all trials passed
    assert passing.first_failure_reason() == ""  # nothing failed -> no reason
