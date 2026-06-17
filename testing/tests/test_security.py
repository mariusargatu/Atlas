"""P8 / principle 11, least agency: unauthorized actions are unreachable, not guarded.

The injected document attack ("reset this customer's equipment") cannot land on a
troubleshooting turn, because that turn never binds the actions tools. One rule, a table of
intent×tool reachability: the cases are the spec.
"""
from __future__ import annotations

import pytest

from atlas.domain.binding import classify_intent, is_reachable


@pytest.mark.parametrize(
    "intent, tool, reachable",
    [
        ("troubleshooting", "reset_modem", False),   # the injected doc attack: write unreachable on a read turn
        ("troubleshooting", "get_equipment", True),
        ("troubleshooting", "search_knowledge", True),
        ("action", "change_plan", True),
        ("action", "get_bill", True),                 # an action turn still binds reads...
        ("action", "list_plans", True),               # ...and the catalog (positive cells, not just writes)
        ("account_read", "change_plan", False),
        ("account_read", "get_bill", True),
        ("account_read", "list_plans", True),
        ("policy_question", "change_plan", False),
        ("policy_question", "search_knowledge", True),
        ("policy_question", "list_plans", True),
    ],
)
def test_intent_binds_only_its_own_tools(intent, tool, reachable):
    assert is_reachable(intent, tool) is reachable


@pytest.mark.parametrize(
    "text, intent",
    [
        ("Switch me to the fast plan", "action"),
        ("Cancel my add-on", "action"),
        ("Please reset my modem", "action"),
        ("Open a ticket for my outage", "action"),
        ("What plan am I on?", "troubleshooting"),       # a read/question is never an action turn
        ("Is my plan contract-free?", "troubleshooting"),
        ("My internet is slow", "troubleshooting"),
    ],
)
def test_classify_intent_marks_change_requests_as_actions(text, intent):
    assert classify_intent(text) == intent
