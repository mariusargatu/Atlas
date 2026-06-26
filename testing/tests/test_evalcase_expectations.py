"""Structured expectations survive the trip from case to grader.

`EvalCase.expected` is prose. A case-parameterised grader (retrieval ids, tool calls) needs the
machine-readable form, so both `EvalCase` and `GradeContext` carry it and `run_case` populates the
context from the case.
"""
from __future__ import annotations

from evals.evalkit.case import EvalCase
from evals.evalkit.graders import GradeContext


def test_eval_case_defaults_expectations_to_empty():
    case = EvalCase(id="c1", turns=("hi",), customer_id="cust_current")
    assert case.expected_doc_ids == ()
    assert case.expected_tool_calls == ()


def test_eval_case_carries_structured_expectations():
    case = EvalCase(
        id="c1",
        turns=("hi",),
        customer_id="cust_current",
        expected_doc_ids=("chunk-a", "chunk-b"),
        expected_tool_calls=({"tool": "knowledge.search_knowledge", "args": {}},),
    )
    assert case.expected_doc_ids == ("chunk-a", "chunk-b")
    assert case.expected_tool_calls[0]["tool"] == "knowledge.search_knowledge"


def test_grade_context_defaults_expectations_to_empty():
    ctx = GradeContext(customer_id="cust_current", final_response="ok", trace=())
    assert ctx.expected_doc_ids == ()
    assert ctx.expected_tool_calls == ()


def test_grade_context_carries_structured_expectations():
    ctx = GradeContext(
        customer_id="cust_current",
        final_response="ok",
        trace=(),
        expected_doc_ids=("chunk-a",),
        expected_tool_calls=({"tool": "actions.change_plan", "args": {}},),
    )
    assert ctx.expected_doc_ids == ("chunk-a",)
    assert ctx.expected_tool_calls[0]["tool"] == "actions.change_plan"
