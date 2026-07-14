"""`matrix.cost_emission`, hermetic (SP9 task 5): the one place a computed generator cost crosses
the trace boundary. Exercised against `InMemoryTracer` (the CI adapter) directly, at the INFORMAL
attribute name level (`model`/`input_tokens`/`output_tokens`/`usd`) -- the translation of those
informal names into the frozen `atlas.cost.*` wire names is `backend/atlas/adapters/
trace_translation.py`'s own concern, cross checked in `testing/tests/test_trace_translation.py` and
the end to end integration test in `test_matrix_cost_trace_integration.py`, never asserted again here.
"""
from __future__ import annotations

from tracing import InMemoryTracer

from matrix.cost_emission import emit_cost


def test_emit_cost_opens_an_llm_kind_span():
    tracer = InMemoryTracer()
    seq = emit_cost(tracer, None, model_id="anthropic:claude-sonnet-5", input_tokens=120, output_tokens=45, usd=0.00081)
    span = tracer.spans[0]
    assert span.seq == seq
    assert span.kind == "llm"
    assert span.name == "generator_cost"


def test_emit_cost_carries_the_model_and_token_and_dollar_figures():
    tracer = InMemoryTracer()
    emit_cost(tracer, None, model_id="openai:gpt-5.6-sol", input_tokens=200, output_tokens=80, usd=0.0013)
    span = tracer.spans[0]
    assert span.attributes["model"] == "openai:gpt-5.6-sol"
    assert span.attributes["input_tokens"] == 200
    assert span.attributes["output_tokens"] == 80
    assert span.attributes["usd"] == 0.0013


def test_emit_cost_nests_under_the_given_parent():
    tracer = InMemoryTracer()
    root = tracer.open("turn", "turn", input="q", intent="troubleshooting", customer_id="cust_1")
    emit_cost(tracer, root, model_id="ollama:qwen2.5:7b", input_tokens=10, output_tokens=5, usd=0.0)
    cost_span = next(s for s in tracer.spans if s.name == "generator_cost")
    assert cost_span.parent == root


def test_emit_cost_reports_zero_dollars_for_ollama_honestly_not_omitted():
    """Ollama always runs free: the $0 figure is still a REAL, present number on the span (never
    omitted), distinct from an old cassette's "cost unavailable" (matrix.spend_gate.cost_from_usage
    returning None) -- $0 IS known, it is simply zero."""
    tracer = InMemoryTracer()
    emit_cost(tracer, None, model_id="ollama:qwen2.5:7b", input_tokens=10, output_tokens=5, usd=0.0)
    span = tracer.spans[0]
    assert span.attributes["usd"] == 0.0
    assert "usd" in span.attributes
