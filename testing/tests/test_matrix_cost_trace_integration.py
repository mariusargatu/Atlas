"""End to end (SP9 task 5, hermetic): a computed generator cost, translated at the trace boundary,
lands on a REAL exported span with the frozen ``atlas.cost.*`` vocabulary. Exercises the REAL
`OtelTracer`/`trace_translation` production path (an injected in memory exporter, no network), the
same proof `test_judge_trace_integration.py` already holds for the judge's own attribute trio --
this file is the one place all three ``atlas.cost.*`` attributes are proven present TOGETHER, on a
real span, from a real (here: fixture) cost computation.
"""
from __future__ import annotations

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from atlas.adapters.otel_tracer import OtelTracer

from matrix.cost_emission import emit_cost
from matrix.spend_gate import cost_from_usage, generation_cost_usd


def _tracer():
    exporter = InMemorySpanExporter()
    tracer = OtelTracer(endpoint="http://example.invalid:4318", config_hash="hash123", exporter=exporter)
    return tracer, exporter


def test_a_real_generator_cost_lands_on_a_real_span_with_all_three_cost_attributes():
    tracer, exporter = _tracer()
    root = tracer.open("turn", "turn", input="what is a data cap", intent="troubleshooting", customer_id="cust_beta")
    usage = {"input_tokens": 120, "output_tokens": 45, "total_tokens": 165}
    usd = cost_from_usage("anthropic", usage)
    emit_cost(tracer, root, model_id="anthropic:claude-sonnet-5", input_tokens=120, output_tokens=45, usd=usd)

    spans = {s.name: s for s in exporter.get_finished_spans()}
    cost_span = spans["generator_cost"]
    assert cost_span.attributes["atlas.cost.input_tokens"] == 120
    assert cost_span.attributes["atlas.cost.output_tokens"] == 45
    assert cost_span.attributes["atlas.cost.usd"] == generation_cost_usd("anthropic", 120, 45)
    assert cost_span.attributes["gen_ai.request.model"] == "anthropic:claude-sonnet-5"


def test_ollama_cost_span_carries_a_real_zero_not_an_absent_attribute():
    tracer, exporter = _tracer()
    emit_cost(tracer, None, model_id="ollama:qwen2.5:7b", input_tokens=30, output_tokens=12, usd=0.0)
    span = exporter.get_finished_spans()[0]
    assert span.attributes["atlas.cost.usd"] == 0.0
