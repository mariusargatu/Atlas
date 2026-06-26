"""The promotion loop (SP8 Task 5.1, D34): judge fail spans and end user thumbs down feedback,
joined against a turn's own question/answer/retrieved content (the same shape
`labeling.generate_label_set.generate_label_items` already produces), promoted into `origin:
promoted` dataset cases -- the seam `dataset_tools.generator.validate_case` already accepts (SP7).
Promotion REQUIRES a known taxonomy label; a candidate with none, or an unknown one, is rejected,
never guessed.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from dataset_tools import taxonomy
from tracing import InMemoryTracer

from atlas.adapters.label_store import LabelRecord

from judge.contract import JudgeContract
from judge.emission import emit_verdict
from judge.promotion import (
    PromotionCandidate,
    PromotionError,
    candidates_from_trace_ids,
    judge_fail_trace_ids,
    promote,
    thumbs_down_trace_ids,
)


def _contract() -> JudgeContract:
    return JudgeContract("gpt-judge", "groundedness-v1", "abc123")


def _label(trace_id: str, role: str, verdict: str) -> LabelRecord:
    return LabelRecord(
        trace_id=trace_id, role=role, verdict=verdict, critique="c",
        created_at=datetime(2026, 6, 15).isoformat(),
    )


@pytest.fixture(scope="module")
def tax() -> taxonomy.Taxonomy:
    return taxonomy.load_taxonomy()


# ---- judge_fail_trace_ids: reading judge spans -------------------------------------------------------


def test_judge_fail_trace_ids_reads_an_ungrounded_verdict_span():
    tracer = InMemoryTracer()
    root = tracer.open("turn", "turn", input="q", intent="troubleshooting", customer_id="cust_1")
    emit_verdict(tracer, root, _contract(), "ungrounded")
    assert judge_fail_trace_ids(tracer.spans) == [str(root)]


def test_judge_fail_trace_ids_excludes_a_grounded_verdict_span():
    tracer = InMemoryTracer()
    root = tracer.open("turn", "turn", input="q", intent="troubleshooting", customer_id="cust_1")
    emit_verdict(tracer, root, _contract(), "grounded")
    assert judge_fail_trace_ids(tracer.spans) == []


def test_judge_fail_trace_ids_covers_multiple_turns_in_span_order():
    tracer = InMemoryTracer()
    root_a = tracer.open("turn", "turn", input="a", intent="troubleshooting", customer_id="cust_a")
    emit_verdict(tracer, root_a, _contract(), "ungrounded")
    root_b = tracer.open("turn", "turn", input="b", intent="troubleshooting", customer_id="cust_b")
    emit_verdict(tracer, root_b, _contract(), "grounded")
    root_c = tracer.open("turn", "turn", input="c", intent="troubleshooting", customer_id="cust_c")
    emit_verdict(tracer, root_c, _contract(), "ungrounded")
    assert judge_fail_trace_ids(tracer.spans) == [str(root_a), str(root_c)]


def test_judge_fail_trace_ids_deduplicates_a_repeated_verdict_on_the_same_turn():
    tracer = InMemoryTracer()
    root = tracer.open("turn", "turn", input="q", intent="troubleshooting", customer_id="cust_1")
    emit_verdict(tracer, root, _contract(), "ungrounded")
    emit_verdict(tracer, root, _contract(), "ungrounded")
    assert judge_fail_trace_ids(tracer.spans) == [str(root)]


def test_judge_fail_trace_ids_ignores_a_parentless_judge_span():
    tracer = InMemoryTracer()
    emit_verdict(tracer, None, _contract(), "ungrounded")
    assert judge_fail_trace_ids(tracer.spans) == []


# ---- thumbs_down_trace_ids: reading end user feedback ------------------------------------------------


def test_thumbs_down_trace_ids_reads_end_user_fail_records():
    records = [_label("t1", "end_user", "fail")]
    assert thumbs_down_trace_ids(records) == ["t1"]


def test_thumbs_down_trace_ids_excludes_adjudicator_role():
    records = [_label("t1", "adjudicator", "fail")]
    assert thumbs_down_trace_ids(records) == []


def test_thumbs_down_trace_ids_excludes_a_thumbs_up():
    records = [_label("t1", "end_user", "pass")]
    assert thumbs_down_trace_ids(records) == []


def test_thumbs_down_trace_ids_deduplicates_and_preserves_order():
    records = [_label("t1", "end_user", "fail"), _label("t2", "end_user", "fail"), _label("t1", "end_user", "fail")]
    assert thumbs_down_trace_ids(records) == ["t1", "t2"]


# ---- candidates_from_trace_ids: joining trace ids against turn content -------------------------------

_ITEM = {
    "case_id": "fixture-plan-contract-free",
    "trace_id": "0",
    "question": "Is my plan contract-free?",
    "answer": "Yes, your current plan is contract-free with no minimum term.",
    "retrieved_chunks": [{"doc_id": "plan-current-page", "chunk_id": "plan-current-page", "text": "...", "score": 0.0}],
    "registry_facts": [{"fact_id": "plan-current:contract_free", "value": "contract-free"}],
}


def test_candidates_from_trace_ids_joins_content_by_trace_id():
    items = {"0": _ITEM}
    candidates = candidates_from_trace_ids(["0"], items, source="judge_fail")
    assert candidates == (
        PromotionCandidate(
            trace_id="0", question=_ITEM["question"], answer=_ITEM["answer"], failure_source="judge_fail",
            retrieved_chunks=tuple(_ITEM["retrieved_chunks"]), registry_facts=tuple(_ITEM["registry_facts"]),
        ),
    )


def test_candidates_from_trace_ids_skips_an_unresolved_trace_id():
    candidates = candidates_from_trace_ids(["missing"], {}, source="judge_fail")
    assert candidates == ()


def test_candidates_from_trace_ids_stamps_the_given_failure_source():
    items = {"0": _ITEM}
    candidates = candidates_from_trace_ids(["0"], items, source="end_user_thumbs_down")
    assert candidates[0].failure_source == "end_user_thumbs_down"


# ---- promote: taxonomy gated, produces a valid origin: promoted case ---------------------------------


def _candidate(trace_id: str = "0") -> PromotionCandidate:
    return PromotionCandidate(
        trace_id=trace_id, question="Is my plan contract-free?",
        answer="Yes, totally free forever.", failure_source="judge_fail",
    )


def test_promote_without_a_failure_class_is_rejected(tax: taxonomy.Taxonomy):
    with pytest.raises(PromotionError, match="taxonomy label"):
        promote(_candidate(), failure_class=None, taxonomy=tax)


def test_promote_with_an_empty_failure_class_is_rejected(tax: taxonomy.Taxonomy):
    with pytest.raises(PromotionError, match="taxonomy label"):
        promote(_candidate(), failure_class="", taxonomy=tax)


def test_promote_with_an_unknown_failure_class_is_rejected(tax: taxonomy.Taxonomy):
    with pytest.raises(taxonomy.TaxonomyError, match="not_a_real_code"):
        promote(_candidate(), failure_class="not_a_real_code", taxonomy=tax)


def test_promote_with_a_known_failure_class_produces_a_valid_promoted_case(tax: taxonomy.Taxonomy):
    case = promote(_candidate(), failure_class="ungrounded_claim", taxonomy=tax)
    assert case["origin"] == "promoted"
    assert case["source_trace_id"] == "0"
    assert case["failure_class"] == "ungrounded_claim"
    assert case["candidate_source"] == "judge_fail"
    assert case["turns"] == [{"user": "Is my plan contract-free?"}]


def test_promote_case_validates_against_the_dataset_schema(tax: taxonomy.Taxonomy):
    import jsonschema
    from contract_tools import loader

    case = promote(_candidate(), failure_class="ungrounded_claim", taxonomy=tax)
    jsonschema.validate(case, loader.load_schema("dataset"))


def test_promote_case_id_defaults_to_a_deterministic_value_from_the_trace_id(tax: taxonomy.Taxonomy):
    case = promote(_candidate("42"), failure_class="ungrounded_claim", taxonomy=tax)
    assert "42" in case["case_id"]


def test_promote_accepts_an_explicit_case_id(tax: taxonomy.Taxonomy):
    case = promote(_candidate(), failure_class="ungrounded_claim", taxonomy=tax, case_id="my-case-1")
    assert case["case_id"] == "my-case-1"


# ---- end to end: judge fail span -> content join -> taxonomy gated promotion -------------------------


def test_end_to_end_judge_fail_promotion(tax: taxonomy.Taxonomy):
    tracer = InMemoryTracer()
    root = tracer.open("turn", "turn", input=_ITEM["question"], intent="troubleshooting", customer_id="cust_1")
    emit_verdict(tracer, root, _contract(), "ungrounded")
    trace_id = str(root)
    items = {trace_id: {**_ITEM, "trace_id": trace_id}}

    fail_ids = judge_fail_trace_ids(tracer.spans)
    candidates = candidates_from_trace_ids(fail_ids, items, source="judge_fail")
    assert len(candidates) == 1
    case = promote(candidates[0], failure_class="ungrounded_claim", taxonomy=tax)
    assert case["origin"] == "promoted"
    assert case["source_trace_id"] == trace_id


def test_end_to_end_thumbs_down_promotion(tax: taxonomy.Taxonomy):
    records = [_label("0", "end_user", "fail")]
    thumbs_ids = thumbs_down_trace_ids(records)
    candidates = candidates_from_trace_ids(thumbs_ids, {"0": _ITEM}, source="end_user_thumbs_down")
    assert len(candidates) == 1
    case = promote(candidates[0], failure_class="false_refusal", taxonomy=tax)
    assert case["origin"] == "promoted"
    assert case["candidate_source"] == "end_user_thumbs_down"
