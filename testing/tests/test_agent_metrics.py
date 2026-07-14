"""Deterministic generation side metrics (SP7 Task 3), hand worked. Every value below is computed
by hand in the test comment, the same discipline `test_ir_metrics.py` and `test_stats.py` use for
their sibling modules: a formula pinned against arithmetic worked independently of the module under
test, not merely re-derived from it. Edge cases named by the plan (zero expected tool calls, an
empty citation set, a single class confusion matrix) each get their own guard test, plus a symmetry
test for the counterfactual equivalence checker SP7 Task 5 reuses.
"""
from __future__ import annotations

from dataset_tools import generator

from quality.agent_metrics import (
    CitationMetrics,
    ConfusionMatrix,
    RefusalRates,
    ToolCallMetrics,
    answer_correctness_rate,
    build_confusion_matrix,
    citation_precision_recall,
    confusion_matrix_accuracy,
    counterfactual_equivalent,
    expected_entity_ids,
    is_fact_grounded,
    refusal_rates,
    tool_call_metrics,
)

# ---- tool_call_metrics ----


def test_tool_call_metrics_right_tools_wrong_argument():
    # Both expected tools get called (selection perfect); the second call's args drift, so exactly
    # one of the two expected calls has a fully matching observed call.
    expected = [
        {"tool": "catalog.get_plan", "args": {"plan_id": "fiber-500"}},
        {"tool": "actions.change_plan", "args": {"plan_id": "fiber-500"}},
    ]
    observed = [
        {"tool": "catalog.get_plan", "args": {"plan_id": "fiber-500"}},
        {"tool": "actions.change_plan", "args": {"plan_id": "WRONG"}},
    ]
    metrics = tool_call_metrics(expected, observed)
    assert metrics == ToolCallMetrics(precision=1.0, recall=1.0, argument_exact_match=0.5)


def test_tool_call_metrics_extra_unrequested_call_hurts_precision_only():
    expected = [{"tool": "account.get_contract", "args": {}}]
    observed = [
        {"tool": "account.get_contract", "args": {}},
        {"tool": "catalog.get_plan", "args": {"plan_id": "x"}},
    ]
    metrics = tool_call_metrics(expected, observed)
    # 1 of 2 observed calls was wanted -> precision 0.5; the 1 wanted call was made -> recall 1.0;
    # that same call's args matched exactly -> argument_exact_match 1.0.
    assert metrics == ToolCallMetrics(precision=0.5, recall=1.0, argument_exact_match=1.0)


def test_tool_call_metrics_missed_call_hurts_recall_only():
    expected = [{"tool": "account.get_contract", "args": {}}]
    observed = [{"tool": "catalog.get_plan", "args": {"plan_id": "x"}}]
    metrics = tool_call_metrics(expected, observed)
    assert metrics == ToolCallMetrics(precision=0.0, recall=0.0, argument_exact_match=0.0)


def test_tool_call_metrics_zero_expected_tool_calls_is_guarded_not_a_crash():
    # A case that requires no tool call at all (most of Task 1's generated factoid cases): the
    # expected-side denominator is empty for recall and argument_exact_match, so both are 0.0 by
    # this module's guarded convention (same "empty denominator -> 0.0, never ZeroDivisionError"
    # rule `ir_metrics.recall_at_k`/`average_precision_at_k` use for an empty relevant set).
    # Callers that only care about agentic cases should skip this metric on cases whose contract
    # entry omits `expected_tool_calls` entirely, the same way a retrieval golden slice is curated
    # to always label at least one relevant chunk.
    assert tool_call_metrics([], []) == ToolCallMetrics(precision=0.0, recall=0.0, argument_exact_match=0.0)
    observed = [{"tool": "account.get_contract", "args": {}}]
    metrics = tool_call_metrics([], observed)
    assert metrics == ToolCallMetrics(precision=0.0, recall=0.0, argument_exact_match=0.0)


def test_tool_call_metrics_observed_empty_is_guarded():
    expected = [{"tool": "account.get_contract", "args": {}}]
    metrics = tool_call_metrics(expected, [])
    assert metrics == ToolCallMetrics(precision=0.0, recall=0.0, argument_exact_match=0.0)


def test_tool_call_metrics_ignores_arg_ordering_within_a_dict():
    # Canonical JSON comparison (sort_keys) means key order in the args dict never matters.
    expected = [{"tool": "catalog.get_plan", "args": {"plan_id": "x", "region": "north"}}]
    observed = [{"tool": "catalog.get_plan", "args": {"region": "north", "plan_id": "x"}}]
    metrics = tool_call_metrics(expected, observed)
    assert metrics == ToolCallMetrics(precision=1.0, recall=1.0, argument_exact_match=1.0)


def test_tool_call_metrics_treats_missing_and_empty_args_the_same():
    expected = [{"tool": "account.get_contract"}]
    observed = [{"tool": "account.get_contract", "args": {}}]
    metrics = tool_call_metrics(expected, observed)
    assert metrics == ToolCallMetrics(precision=1.0, recall=1.0, argument_exact_match=1.0)


# ---- citation_precision_recall / expected_entity_ids ----


def test_citation_precision_recall_hand_worked():
    cited = ["plan-fiber-500", "fee-early-termination"]
    expected = frozenset({"plan-fiber-500"})
    assert citation_precision_recall(cited, expected) == CitationMetrics(precision=0.5, recall=1.0)


def test_citation_precision_recall_empty_citation_set_is_guarded():
    metrics = citation_precision_recall([], frozenset({"plan-fiber-500"}))
    assert metrics == CitationMetrics(precision=0.0, recall=0.0)


def test_citation_precision_recall_empty_expected_set_is_guarded():
    metrics = citation_precision_recall(["plan-fiber-500"], frozenset())
    assert metrics == CitationMetrics(precision=0.0, recall=0.0)


def test_citation_precision_recall_both_empty_is_guarded():
    assert citation_precision_recall([], frozenset()) == CitationMetrics(precision=0.0, recall=0.0)


def test_citation_precision_recall_perfect_overlap():
    cited = ["a", "b"]
    expected = frozenset({"a", "b"})
    assert citation_precision_recall(cited, expected) == CitationMetrics(precision=1.0, recall=1.0)


def test_expected_entity_ids_splits_the_entity_id_field_convention():
    facts = [
        {"fact_id": "contract_term-daniel-2025:contract_months", "value": 12},
        {"fact_id": "fee-early-termination:amount", "value": "150"},
        {"fact_id": "contract_term-daniel-2025:vintage_year", "value": 2025},  # same entity twice
    ]
    assert expected_entity_ids(facts) == frozenset({"contract_term-daniel-2025", "fee-early-termination"})


def test_expected_entity_ids_degrades_gracefully_on_a_colonless_fact_id():
    # A hand curated case predating the entity_id:field convention would use an opaque label like
    # this; gc-0001 and gc-0002 (contracts/dataset/examples) both since reconciled to the real
    # convention (d899464 and the SP7 Task 3 fix round), but the degradation path itself must stay
    # correct for any future hand authored case that has not been reconciled yet: it must not
    # raise, it simply cannot resolve to a real registry entity id.
    facts = [{"fact_id": "fact-opaque-label", "value": "some value"}]
    assert expected_entity_ids(facts) == frozenset({"fact-opaque-label"})


def test_expected_entity_ids_of_no_facts_is_empty():
    assert expected_entity_ids([]) == frozenset()


# ---- confusion matrix ----


def test_confusion_matrix_hand_worked_accuracy():
    pairs = [
        ("troubleshooting", "troubleshooting"),
        ("troubleshooting", "troubleshooting"),
        ("troubleshooting", "troubleshooting"),
        ("plan_change", "plan_change"),
        ("plan_change", "plan_change"),
        ("plan_change", "troubleshooting"),
    ]
    matrix = build_confusion_matrix(pairs)
    assert matrix.labels == ("plan_change", "troubleshooting")
    assert matrix.counts[("troubleshooting", "troubleshooting")] == 3
    assert matrix.counts[("plan_change", "plan_change")] == 2
    assert matrix.counts[("plan_change", "troubleshooting")] == 1
    assert confusion_matrix_accuracy(matrix) == 5 / 6


def test_confusion_matrix_single_class_perfect():
    # Edge case named by the plan: only one distinct intent appears at all.
    matrix = build_confusion_matrix([("troubleshooting", "troubleshooting")] * 4)
    assert matrix.labels == ("troubleshooting",)
    assert confusion_matrix_accuracy(matrix) == 1.0


def test_confusion_matrix_single_true_class_with_misclassification():
    # Edge case named by the plan, the other reading: one true intent, imperfectly observed.
    matrix = build_confusion_matrix(
        [("troubleshooting", "troubleshooting"), ("troubleshooting", "plan_change")]
    )
    assert confusion_matrix_accuracy(matrix) == 0.5


def test_confusion_matrix_of_no_pairs_is_guarded():
    matrix = build_confusion_matrix([])
    assert matrix == ConfusionMatrix(labels=(), counts={})
    assert confusion_matrix_accuracy(matrix) == 0.0


# ---- refusal rates ----


def test_refusal_rates_hand_worked():
    # (answerable, observed_refused)
    outcomes = [
        (True, False),   # answerable, answered: correct
        (True, True),    # answerable, refused: FALSE refusal
        (False, True),   # unanswerable, refused: correct
        (False, False),  # unanswerable, answered: MISSED refusal (the hallucination shaped one)
    ]
    assert refusal_rates(outcomes) == RefusalRates(missed_refusal_rate=0.5, false_refusal_rate=0.5)


def test_refusal_rates_no_unanswerable_cases_is_guarded():
    assert refusal_rates([(True, False), (True, True)]) == RefusalRates(
        missed_refusal_rate=0.0, false_refusal_rate=0.5
    )


def test_refusal_rates_no_answerable_cases_is_guarded():
    assert refusal_rates([(False, True), (False, False)]) == RefusalRates(
        missed_refusal_rate=0.5, false_refusal_rate=0.0
    )


def test_refusal_rates_of_no_outcomes_is_guarded():
    assert refusal_rates([]) == RefusalRates(missed_refusal_rate=0.0, false_refusal_rate=0.0)


def test_refusal_rates_all_correct_is_zero_both_ways():
    outcomes = [(True, False), (True, False), (False, True), (False, True)]
    assert refusal_rates(outcomes) == RefusalRates(missed_refusal_rate=0.0, false_refusal_rate=0.0)


# ---- reference based answer correctness ----


def test_is_fact_grounded_matches_response_text_with_typed_coercion():
    # value is an int in the registry (contract_months: 12); the SAME str() coercion
    # corpus_tools.verify._registry_value applies must find it as a literal token in prose.
    fact = {"fact_id": "contract_term-daniel-2025:contract_months", "value": 12}
    assert is_fact_grounded(fact, "Your plan runs on a 12 month contract.") is True


def test_is_fact_grounded_matches_tool_result_when_absent_from_text():
    fact = {"fact_id": "plan-fiber-500:monthly_price", "value": "39.99"}
    text, results = "I don't have that information.", [{"monthly_price": "39.99"}]
    assert is_fact_grounded(fact, text, tool_results=results) is True


def test_is_fact_grounded_matches_a_numeric_tool_result_leaf_with_typed_coercion():
    fact = {"fact_id": "contract_term-daniel-2025:contract_months", "value": 12}
    assert is_fact_grounded(fact, "no mention here", tool_results=[{"contract_months": 12}]) is True


def test_is_fact_grounded_false_when_neither_source_carries_the_value():
    fact = {"fact_id": "plan-fiber-500:monthly_price", "value": "39.99"}
    text, results = "I don't have that information.", [{"monthly_price": "19.99"}]
    assert is_fact_grounded(fact, text, tool_results=results) is False


def test_answer_correctness_rate_hand_worked():
    expected_facts = [
        {"fact_id": "a:b", "value": "X"},
        {"fact_id": "c:d", "value": "Y"},
    ]
    rate = answer_correctness_rate(expected_facts, "only X appears here", tool_results=[])
    assert rate == 0.5


def test_answer_correctness_rate_of_no_expected_facts_is_guarded():
    assert answer_correctness_rate([], "anything at all") == 0.0


def test_answer_correctness_rate_perfect():
    expected_facts = [{"fact_id": "a:b", "value": "X"}, {"fact_id": "c:d", "value": "Y"}]
    assert answer_correctness_rate(expected_facts, "X and Y both appear here") == 1.0


# ---- counterfactual equivalence (Task 5 reuse) ----


def _case(facts, refusal_class=None, **extra):
    return {"expected_facts": facts, "refusal_class": refusal_class, **extra}


def test_counterfactual_equivalent_true_when_facts_and_refusal_class_match():
    facts = [{"fact_id": "plan-fiber-500:monthly_price", "value": "39.99"}]
    a = _case(facts, persona={"name": "sarah", "style": "direct"})
    b = _case(facts, persona={"name": "chen", "style": "formal"})
    assert counterfactual_equivalent(a, b) is True


def test_counterfactual_equivalent_is_indifferent_to_fact_list_order():
    facts_a = [{"fact_id": "a:b", "value": "1"}, {"fact_id": "c:d", "value": "2"}]
    facts_b = [{"fact_id": "c:d", "value": "2"}, {"fact_id": "a:b", "value": "1"}]
    assert counterfactual_equivalent(_case(facts_a), _case(facts_b)) is True


def test_counterfactual_equivalent_false_on_diverging_value():
    a = _case([{"fact_id": "a:b", "value": "1"}])
    b = _case([{"fact_id": "a:b", "value": "2"}])
    assert counterfactual_equivalent(a, b) is False


def test_counterfactual_equivalent_false_on_diverging_refusal_class():
    facts = [{"fact_id": "a:b", "value": "1"}]
    a = _case(facts, refusal_class=None)
    b = _case(facts, refusal_class="out_of_scope")
    assert counterfactual_equivalent(a, b) is False


def test_counterfactual_equivalent_true_when_both_cases_omit_optional_fields():
    assert counterfactual_equivalent({}, {}) is True


def test_counterfactual_equivalence_is_symmetric():
    facts = [{"fact_id": "a:b", "value": "1"}]
    pairs = [
        (_case(facts), _case(facts)),
        (_case(facts, refusal_class="out_of_scope"), _case(facts)),
        (_case([{"fact_id": "a:b", "value": "1"}]), _case([{"fact_id": "a:b", "value": "2"}])),
        (_case([]), _case(facts)),
        ({}, _case(facts, refusal_class="out_of_scope")),
    ]
    for a, b in pairs:
        assert counterfactual_equivalent(a, b) == counterfactual_equivalent(b, a)


# ---- grounded against Task 1's real generated cases (read only import, not a hand fixture) ----


def test_expected_entity_ids_and_answer_correctness_over_a_real_generated_case():
    cases = generator.generate_cases()
    one_hop = next(
        c for c in cases if c.get("hop_count") == 1 and c["adversarial_class"] is None
    )
    facts = one_hop["expected_facts"]
    entity_ids = expected_entity_ids(facts)
    # every expected_facts fact_id the generator emits follows entity_id:field, so the derived
    # entity id set is never empty and never the raw fact_id string.
    assert entity_ids and all(":" not in entity_id for entity_id in entity_ids)

    fact_value = str(facts[0]["value"])
    response_text = f"Here is what I found: {fact_value}."
    assert answer_correctness_rate(facts, response_text) == 1.0
    assert answer_correctness_rate(facts, "I have no information on that.") == 0.0


def test_answer_correctness_over_a_real_contradiction_case_with_an_int_value():
    cases = generator.generate_cases()
    contradiction = next(c for c in cases if c["adversarial_class"] == "grounded_not_true")
    fact = contradiction["expected_facts"][0]
    assert isinstance(fact["value"], int)  # contract_months, the typed coercion case in point
    response_text = f"Your plan runs on a {fact['value']} month contract."
    assert is_fact_grounded(fact, response_text) is True
