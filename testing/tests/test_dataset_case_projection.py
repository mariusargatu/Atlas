"""The dataset case -> EvalCase projection. Every field derived, none hand mapped."""
from __future__ import annotations

import pytest

from evals.evalkit.dataset_case import NEUTRAL_SESSION, dataset_case_to_eval_case, graders_for


def _case(**overrides) -> dict:
    base = {
        "case_id": "gen-fact-region-north-coverage_type",
        "split": "dev",
        "origin": "synthetic",
        "intent": "troubleshooting",
        "answerable": True,
        "turns": [{"user": "What is the coverage_type of region-north?"}],
        "expected_facts": [{"fact_id": "region-north:coverage_type", "value": "urban and suburban"}],
        "expected_doc_ids": ["8892cb9848b46ae4"],
    }
    base.update(overrides)
    return base


def test_identity_falls_back_to_the_neutral_session():
    assert dataset_case_to_eval_case(_case()).customer_id == NEUTRAL_SESSION


def test_declared_identity_wins():
    case = dataset_case_to_eval_case(_case(customer_id="cust_legacy_term"))
    assert case.customer_id == "cust_legacy_term"


def test_id_name_and_turns_are_derived():
    ec = dataset_case_to_eval_case(_case())
    assert ec.id == ec.name == "gen-fact-region-north-coverage_type"
    assert ec.turns == ("What is the coverage_type of region-north?",)


def test_expected_prose_is_rendered_from_expected_facts():
    ec = dataset_case_to_eval_case(_case())
    assert ec.expected == "region-north:coverage_type = urban and suburban"


def test_structured_expectations_ride_along():
    ec = dataset_case_to_eval_case(_case())
    assert ec.expected_doc_ids == ("8892cb9848b46ae4",)


def test_risk_prefers_adversarial_then_failure_then_intent():
    assert dataset_case_to_eval_case(_case(adversarial_class="grounded_not_true")).risk == "grounded_not_true"
    assert dataset_case_to_eval_case(_case(failure_class="hallucination")).risk == "hallucination"
    assert dataset_case_to_eval_case(_case()).risk == "troubleshooting"


@pytest.mark.parametrize(
    "overrides, expected_grader",
    [
        ({"end_state": {"account_assertions": [{"path": "plan_id", "equals": "p"}]}}, "write-applied-after-confirm"),
        ({"expected_doc_ids": ["abc"]}, "retrieval-ids-recalled"),
        ({"expected_tool_calls": [{"tool": "knowledge.search_knowledge", "args": {}}]}, "tool-calls-match"),
    ],
)
def test_graders_are_derived_from_case_shape(overrides, expected_grader):
    assert expected_grader in graders_for(_case(**overrides))


def test_graders_for_applies_both_write_and_read_graders_when_a_case_declares_both():
    """`end_state.account_assertions` and `expected_facts` are independent shape signals: a case
    carrying both must project to BOTH graders. An `elif` here would silently drop
    `answer-true-vs-account` the moment a case combines them (0 of today's cases do, but the schema
    permits it), which is exactly the "runs green while checking less" defect this function guards
    against everywhere else."""
    case = _case(end_state={"account_assertions": [{"path": "plan_id", "equals": "p"}]})
    assert "expected_facts" in case  # the fixture's own base already carries expected_facts
    names = graders_for(case)
    assert "write-applied-after-confirm" in names
    assert "answer-true-vs-account" in names


def test_every_derived_grader_name_resolves_in_the_registry():
    from evals.evalkit.metric_graders import GOLDEN_GRADERS

    shapes = [
        _case(),
        _case(end_state={"account_assertions": [{"path": "plan_id", "equals": "p"}]}),
        _case(expected_tool_calls=[{"tool": "actions.change_plan", "args": {}}]),
    ]
    for case in shapes:
        for name in graders_for(case):
            assert name in GOLDEN_GRADERS, f"{name} is not registered"
