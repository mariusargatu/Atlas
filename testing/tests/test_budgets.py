"""P7 (the deterministic slice of monitoring, hermetic): most of production monitoring is online and
non-deterministic (an LLM judge on sampled live traffic, a trend not a gate). But two seams are
exact recorded numbers and so CAN gate, cost/latency/token and the CALL BUDGET. This gates the call
budget over a recorded tool-call sequence: no retry storm, no reading the same record forty times in
a loop the model cannot feel itself running. Deterministic, so it belongs in the PR lane; the same
number trended live is the operator lane's job.
"""
from __future__ import annotations

from evals.monitor.budget import DEFAULT_RETRIEVAL_TOOLS, Budget, check_budget

RETRIEVAL_TOOLS = DEFAULT_RETRIEVAL_TOOLS  # single home for the budgeting retrieval set (budget.py)


def test_a_healthy_turn_stays_within_budget():
    report = check_budget(
        ["search_knowledge", "get_account_summary"],
        Budget(max_tool_calls=5, max_retrieval_rounds=2),
        retrieval_tools=RETRIEVAL_TOOLS,
    )
    assert report.ok is True
    assert report.tool_calls == 2
    assert report.retrieval_rounds == 1


def test_a_retrieval_retry_storm_blows_the_budget():
    report = check_budget(
        ["search_knowledge"] * 5,
        Budget(max_tool_calls=10, max_retrieval_rounds=2),
        retrieval_tools=RETRIEVAL_TOOLS,
    )
    assert report.ok is False
    assert any("retrieval" in r.lower() for r in report.reasons)


def test_too_many_tool_calls_is_caught():
    report = check_budget(
        ["get_bill"] * 6,
        Budget(max_tool_calls=5, max_retrieval_rounds=3),
        retrieval_tools=RETRIEVAL_TOOLS,
    )
    assert report.ok is False
    assert report.tool_calls == 6


def test_at_the_limit_is_within_budget_not_over_it():
    # exactly at both limits passes: the budget uses strict '>', so == limit is fine
    report = check_budget(
        ["search_knowledge", "search_knowledge", "get_bill", "get_bill", "get_equipment"],
        Budget(max_tool_calls=5, max_retrieval_rounds=2),
        retrieval_tools=RETRIEVAL_TOOLS,
    )
    assert report.ok is True
    assert report.tool_calls == 5
    assert report.retrieval_rounds == 2


def test_both_budgets_exceeded_reports_both_reasons():
    report = check_budget(
        ["search_knowledge"] * 6,
        Budget(max_tool_calls=5, max_retrieval_rounds=2),
        retrieval_tools=RETRIEVAL_TOOLS,
    )
    assert report.ok is False
    assert len(report.reasons) == 2
