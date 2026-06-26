"""`judge.emission`, hermetic (SP8 task 1): the one place a computed verdict crosses the trace
boundary. Exercised against `InMemoryTracer` (the CI adapter) directly, at the INFORMAL attribute
name level (`judge_id`/`rubric_version`/`verdict`) -- the translation of those informal names into
the frozen `atlas.judge.*` wire names is `backend/atlas/adapters/trace_translation.py`'s own concern,
cross checked in `testing/tests/test_trace_translation.py` and the end to end integration test in
`test_judge_trace_integration.py`, never re-asserted here.
"""
from __future__ import annotations

import pytest

from tracing import InMemoryTracer

from atlas import metrics

from judge.contract import JudgeContract
from judge.emission import emit_verdict


def _contract() -> JudgeContract:
    return JudgeContract("gpt-judge", "groundedness-v1", "abc123")


def test_emit_verdict_opens_a_judge_kind_span():
    tracer = InMemoryTracer()
    seq = emit_verdict(tracer, None, _contract(), "grounded")
    span = tracer.spans[0]
    assert span.seq == seq
    assert span.kind == "judge"
    assert span.name == "judge_verdict"


def test_emit_verdict_carries_the_fingerprint_rubric_version_and_verdict():
    tracer = InMemoryTracer()
    contract = _contract()
    emit_verdict(tracer, None, contract, "grounded")
    span = tracer.spans[0]
    assert span.attributes["judge_id"] == contract.fingerprint()
    assert span.attributes["rubric_version"] == "groundedness-v1"
    assert span.attributes["verdict"] == "grounded"


def test_emit_verdict_carries_an_ungrounded_verdict_too():
    tracer = InMemoryTracer()
    emit_verdict(tracer, None, _contract(), "ungrounded")
    assert tracer.spans[0].attributes["verdict"] == "ungrounded"


def test_emit_verdict_nests_under_the_given_parent():
    tracer = InMemoryTracer()
    root = tracer.open("turn", "turn", input="q", intent="troubleshooting", customer_id="cust_1")
    emit_verdict(tracer, root, _contract(), "grounded")
    judge_span = next(s for s in tracer.spans if s.kind == "judge")
    assert judge_span.parent == root


def test_emit_verdict_fails_closed_on_a_non_wire_verdict():
    tracer = InMemoryTracer()
    with pytest.raises(ValueError, match="grounded"):
        emit_verdict(tracer, None, _contract(), "PASS")  # the judge's own prompt vocabulary, not the wire one


def test_emit_verdict_fails_closed_on_an_empty_verdict():
    tracer = InMemoryTracer()
    with pytest.raises(ValueError):
        emit_verdict(tracer, None, _contract(), "")


# ---- SP8 Task 4 remainder: the atlas_judge_pass_total/atlas_judge_fail_total pair, incremented
# right here, the ONE place a computed verdict crosses the trace boundary ---------------------------


def test_emit_verdict_increments_the_judge_pass_counter_on_a_grounded_verdict():
    tracer = InMemoryTracer()
    emit_verdict(tracer, None, _contract(), "grounded")
    body = metrics.render()
    assert "atlas_judge_pass_total 1" in body
    assert "atlas_judge_fail_total 0" in body


def test_emit_verdict_increments_the_judge_fail_counter_on_an_ungrounded_verdict():
    tracer = InMemoryTracer()
    emit_verdict(tracer, None, _contract(), "ungrounded")
    body = metrics.render()
    assert "atlas_judge_fail_total 1" in body
    assert "atlas_judge_pass_total 0" in body


def test_emit_verdict_accumulates_across_several_verdicts():
    tracer = InMemoryTracer()
    emit_verdict(tracer, None, _contract(), "grounded")
    emit_verdict(tracer, None, _contract(), "grounded")
    emit_verdict(tracer, None, _contract(), "ungrounded")
    body = metrics.render()
    assert "atlas_judge_pass_total 2" in body
    assert "atlas_judge_fail_total 1" in body


def test_emit_verdict_fails_closed_before_touching_any_counter():
    """A non wire verdict raises before the span even opens (the existing fail closed test above) --
    this asserts neither counter moves either, so a caller passing the judge's own PASS/FAIL prompt
    vocabulary by mistake never silently pollutes a metric alongside the exception it already
    raises."""
    tracer = InMemoryTracer()
    with pytest.raises(ValueError):
        emit_verdict(tracer, None, _contract(), "PASS")
    body = metrics.render()
    assert "atlas_judge_pass_total 0" in body
    assert "atlas_judge_fail_total 0" in body
