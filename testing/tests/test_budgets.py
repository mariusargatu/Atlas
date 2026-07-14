"""The call budget gate, the deterministic slice of monitoring: exact over a recorded tool-call
sequence (no retry storm, no unbounded call count), so it can gate the hermetic lane. The same
counts trended over live traffic are the operator lane's non-gating alarm.
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


def test_budget_lives_in_the_domain_and_the_monitor_module_reexports_it():
    # the runtime (graph) and the eval lanes must gate on the SAME objects, not parallel copies
    import atlas.domain.budget as domain_budget
    import evals.monitor.budget as monitor_budget

    assert monitor_budget.Budget is domain_budget.Budget
    assert monitor_budget.BudgetReport is domain_budget.BudgetReport
    assert monitor_budget.check_budget is domain_budget.check_budget
    assert monitor_budget.DEFAULT_BUDGET is domain_budget.DEFAULT_BUDGET
    assert monitor_budget.DEFAULT_RETRIEVAL_TOOLS is domain_budget.DEFAULT_RETRIEVAL_TOOLS


def test_runtime_recursion_limit_is_derived_from_the_default_budget():
    from atlas.domain.budget import DEFAULT_BUDGET, RECURSION_LIMIT

    # A budget-legal turn of max_tool_calls single-call reads costs 2 supersteps per read round trip
    # plus a two-superstep answer tail (final agent + pre_render_guard) = 2*max_tool_calls + 2. langgraph
    # raises when the step count exceeds recursion_limit, so the limit sits one above that: +3. The
    # end-to-end proof that this value lets the last legal turn COMPLETE lives in test_atlas_graph.
    assert RECURSION_LIMIT == 2 * DEFAULT_BUDGET.max_tool_calls + 3
