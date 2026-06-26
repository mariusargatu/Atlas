"""End to end (SP8 task 1, hermetic): a REPLAY judge run's verdict, translated at the trace
boundary, lands on a REAL exported span with the frozen ``atlas.judge.*`` vocabulary -- and the same
turn carries a real HMAC pseudonym of its ``customer_id``, never anything the model produced.
Exercises the REAL `OtelTracer`/`trace_translation` production path (an injected in memory exporter,
no network), the same proof `test_otel_tracer.py`/`test_trace_translation.py` already hold for every
other attribute -- this file is the one place all four `atlas.judge.*`/`atlas.subject.pseudonym`
attributes are proven present TOGETHER, on real spans, from a seeded REPLAY judge run.
"""
from __future__ import annotations

import tempfile

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from replay.cassette_store import seed_cassette
from replay.gateway import GatewayChatModel

from atlas.adapters.otel_tracer import OtelTracer

from judge.contract import JudgeContract
from judge.emission import emit_verdict
from judge.llm_judge import judge_label, translate_verdict
from judge.rubric import RUBRIC_GROUNDEDNESS, prompt

_MODEL_ID = "gpt-judge"


def _tracer():
    exporter = InMemorySpanExporter()
    tracer = OtelTracer(endpoint="http://example.invalid:4318", config_hash="hash123", exporter=exporter)
    return tracer, exporter


def _seeded_gateway(cdir, question, answer, context, reply):
    seed_cassette(
        cdir, prompt(RUBRIC_GROUNDEDNESS, question, answer, context),
        {"content": reply, "tool_calls": []}, _MODEL_ID,
    )
    return GatewayChatModel(model_id=_MODEL_ID, cassette_dir=cdir, mode="replay")


def test_a_seeded_replay_run_produces_a_grounded_verdict_with_all_four_attributes_on_the_span():
    tracer, exporter = _tracer()
    root = tracer.open(
        "turn", "turn", input="Is my plan contract-free?", intent="troubleshooting",
        customer_id="cust_alpha",
    )
    question, answer, context = "Is my plan contract-free?", "No, it has a cancellation fee.", "chunk: fee applies"
    with tempfile.TemporaryDirectory(prefix="judge-trace-") as cdir:
        gateway = _seeded_gateway(cdir, question, answer, context, "PASS")
        label = judge_label(gateway, RUBRIC_GROUNDEDNESS, question, answer, context)
    verdict = translate_verdict(label)
    contract = JudgeContract(_MODEL_ID, RUBRIC_GROUNDEDNESS.version, "abc123")
    emit_verdict(tracer, root, contract, verdict)

    spans = {s.name: s for s in exporter.get_finished_spans()}
    judge_span = spans["judge_verdict"]
    assert judge_span.attributes["atlas.judge.id"] == contract.fingerprint()
    assert judge_span.attributes["atlas.judge.rubric_version"] == RUBRIC_GROUNDEDNESS.version
    assert judge_span.attributes["atlas.judge.verdict"] == "grounded"
    assert spans["turn"].attributes["atlas.subject.pseudonym"]  # the fourth attribute, non empty


def test_a_seeded_replay_run_produces_an_ungrounded_verdict():
    tracer, exporter = _tracer()
    root = tracer.open(
        "turn", "turn", input="Is my plan contract-free?", intent="troubleshooting",
        customer_id="cust_beta",
    )
    question, answer, context = "Is my plan contract-free?", "Yes, totally free forever.", "chunk: fee applies"
    with tempfile.TemporaryDirectory(prefix="judge-trace-") as cdir:
        gateway = _seeded_gateway(cdir, question, answer, context, "FAIL")
        label = judge_label(gateway, RUBRIC_GROUNDEDNESS, question, answer, context)
    verdict = translate_verdict(label)
    contract = JudgeContract(_MODEL_ID, RUBRIC_GROUNDEDNESS.version, "abc123")
    emit_verdict(tracer, root, contract, verdict)

    judge_span = next(s for s in exporter.get_finished_spans() if s.name == "judge_verdict")
    assert judge_span.attributes["atlas.judge.verdict"] == "ungrounded"


def test_judge_span_kind_is_evaluator_on_the_wire():
    tracer, exporter = _tracer()
    contract = JudgeContract(_MODEL_ID, RUBRIC_GROUNDEDNESS.version, "abc123")
    emit_verdict(tracer, None, contract, "grounded")
    span = exporter.get_finished_spans()[0]
    assert span.attributes["openinference.span.kind"] == "EVALUATOR"


# ---- pseudonym: a real HMAC of customer_id, never model sourced -----------------------------------


def test_pseudonym_is_stable_per_customer_and_never_the_raw_id_or_message_content():
    tracer, exporter = _tracer()
    tracer.open("turn", "turn", input="q1", intent="troubleshooting", customer_id="cust_same")
    tracer.open(
        "turn", "turn", input="q2 mentions cust_decoy right here in the text",
        intent="troubleshooting", customer_id="cust_same",
    )
    spans = [s for s in exporter.get_finished_spans() if s.name == "turn"]
    pseudonyms = {s.attributes["atlas.subject.pseudonym"] for s in spans}
    assert len(pseudonyms) == 1  # same customer_id -> same pseudonym, regardless of message content
    pseudonym = pseudonyms.pop()
    assert pseudonym != "cust_same"  # never the raw id, verbatim
    assert "cust_decoy" not in pseudonym  # never derived from message content either


def test_pseudonym_differs_for_a_different_customer():
    tracer, exporter = _tracer()
    tracer.open("turn", "turn", input="q", intent="troubleshooting", customer_id="cust_one")
    tracer.open("turn", "turn", input="q", intent="troubleshooting", customer_id="cust_two")
    pseudonyms = [s.attributes["atlas.subject.pseudonym"] for s in exporter.get_finished_spans() if s.name == "turn"]
    assert pseudonyms[0] != pseudonyms[1]
