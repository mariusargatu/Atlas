"""SP10 task 3: `judge.full_sweep`, hermetic. Every grading/report computation here is proven with
an injected REPLAY agent graph and a REPLAY judge gateway -- zero keys, zero egress. Only
`build_live_agent`/`build_live_judge_gateway` (both `judge.live_pr_lane`'s own, reused unchanged)
and `main` reach for a real live provider or a real Postgres/TEI retriever, and neither of those
live bodies ever run in this file (mirrors `test_judge_live_pr_lane.py`'s own scope).
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from determinism.canonical import serialize_tool_result
from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from replay.cassette_store import seed_cassette
from replay.gateway import GatewayChatModel
from tracing import InMemoryTracer

from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.domain.actions import ActionsBackend
from atlas.domain.retrieval import RetrievalConfig
from atlas.orchestration.atlas_graph import build_atlas_graph

from judge.rubric import RUBRIC_GROUNDEDNESS, prompt as judge_prompt

from judge.full_sweep import (
    TREND_PATH,
    grade_case,
    render_report,
    run,
    select_frontier_judge_tier,
)

from matrix.live_driver import MissingEnvVarError

_QUERY = "plan contract term cancel fee"


# ---- grade_case ---------------------------------------------------------------------------------


def _case(case_id: str, **fields) -> dict:
    return {"case_id": case_id, "turns": [{"user": "q"}], "answerable": True, **fields}


def _seed_judge_cassette(cassette_dir, question: str, answer: str, context: str, verdict: str) -> None:
    messages = judge_prompt(RUBRIC_GROUNDEDNESS, question, answer, context)
    seed_cassette(cassette_dir, messages, {"content": verdict, "tool_calls": []}, model_id="test-judge")


def test_grade_case_a_grounded_pass_reads_as_a_single_passing_trial(tmp_path):
    item = {"case_id": "c1", "question": "q1", "answer": "Fiber 500 costs 35.00.", "retrieved_chunks": []}
    _seed_judge_cassette(tmp_path, "q1", "Fiber 500 costs 35.00.", "", "PASS")
    gateway = GatewayChatModel(model_id="test-judge", cassette_dir=tmp_path, mode="replay")

    result = grade_case(_case("c1", intent="troubleshooting"), item, judge_gateway=gateway)

    assert result.case_id == "c1"
    assert result.passes == 1
    assert result.k == 1
    assert result.rate == 1.0
    assert result.name == "troubleshooting"
    assert result.risk == "none"  # no adversarial_class on this case


def test_grade_case_an_ungrounded_fail_reads_as_a_single_failing_trial(tmp_path):
    item = {"case_id": "c1", "question": "q1", "answer": "It is the Ultra plan.", "retrieved_chunks": []}
    _seed_judge_cassette(tmp_path, "q1", "It is the Ultra plan.", "", "FAIL")
    gateway = GatewayChatModel(model_id="test-judge", cassette_dir=tmp_path, mode="replay")

    result = grade_case(
        _case("c1", adversarial_class="hallucination_bait"), item, judge_gateway=gateway
    )

    assert result.passes == 0
    assert result.k == 1
    assert result.risk == "hallucination_bait"
    assert result.trials[0].verdicts[0].passed is False


def test_grade_case_a_cassette_miss_reads_as_a_failing_trial_not_a_crash(tmp_path):
    """A per item exception (here: an unseeded cassette) is recorded as a failed trial and the item
    stays in the sample -- never silently dropped, the same discipline
    `judge.live_pr_lane.judge_the_items` already holds itself to."""
    item = {"case_id": "c1", "question": "unseeded", "answer": "a", "retrieved_chunks": []}
    gateway = GatewayChatModel(model_id="test-judge", cassette_dir=tmp_path, mode="replay")

    result = grade_case(_case("c1"), item, judge_gateway=gateway)

    assert result.passes == 0
    assert result.k == 1


# ---- select_frontier_judge_tier -------------------------------------------------------------------


def test_select_frontier_judge_tier_prefers_openai_when_both_keys_are_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert select_frontier_judge_tier() == ("openai", "gpt-5.6-sol")


def test_select_frontier_judge_tier_falls_back_to_anthropic(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert select_frontier_judge_tier() == ("anthropic", "claude-opus-4-8")


def test_select_frontier_judge_tier_raises_when_neither_key_is_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingEnvVarError, match="OPENAI_API_KEY"):
        select_frontier_judge_tier()


def test_select_frontier_judge_tier_is_the_top_tier_not_the_cheap_one():
    """The one deliberate difference from the Live PR lane's own cheap tier pick (D15's 'calibrated
    frontier judge'): the model ids here must NOT be the Live PR lane's own cheap ones."""
    from judge.live_pr_lane import _JUDGE_TIERS as cheap_tiers

    from judge.full_sweep import _FRONTIER_JUDGE_TIERS as frontier_tiers

    cheap_models = {model_id for _, model_id in cheap_tiers}
    frontier_models = {model_id for _, model_id in frontier_tiers}
    assert cheap_models.isdisjoint(frontier_models)


# ---- run(): the full dependency injected assembly, end to end, hermetic ---------------------------


def _seed_agent_turn(cassette_dir, question: str, answer: str) -> None:
    """A two step search_knowledge turn (decision call + synthesis call), the SAME technique
    `test_judge_live_pr_lane.py`'s own `_seed_agent_turn` already establishes, so the second
    cassette key matches what the real `InMemoryRetriever` produces, never a guess."""
    user = HumanMessage(question)
    toolcall = [{"name": "search_knowledge", "args": {"query": _QUERY}, "id": "k1"}]
    seed_cassette(cassette_dir, [user], {"content": "", "tool_calls": toolcall}, model_id="claude-test")

    chunks = InMemoryRetriever().search_chunks(_QUERY, config=RetrievalConfig())
    passages = serialize_tool_result(
        [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": c.score, "text": c.text} for c in chunks]
    )
    ai = AIMessage(content="", tool_calls=toolcall)
    tool_msg = ToolMessage(content=passages, tool_call_id="k1", name="search_knowledge")
    seed_cassette(cassette_dir, [user, ai, tool_msg], {"content": answer, "tool_calls": []}, model_id="claude-test")


def _agent_graph(cassette_dir):
    tracer = InMemoryTracer()
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=cassette_dir, mode="replay")
    backend = ActionsBackend(IdFactory("ref"))
    graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer(), tracer=tracer)
    return graph


_RUN_CASES = [
    {
        "case_id": f"full-sweep-case-{i}",
        "turns": [{"user": f"What is the name of plan-fiber-{i}?"}],
        "answerable": True,
        "adversarial_class": None,
        "intent": "factoid",
    }
    for i in (500, 100, 200)
]


def test_run_end_to_end_hermetic_builds_an_evalkit_report_with_provenance(tmp_path):
    agent_cassettes = tmp_path / "agent"
    judge_cassettes = tmp_path / "judge"
    real_chunks = InMemoryRetriever().search_chunks(_QUERY, config=RetrievalConfig())
    ctx = "\n".join(f"[{c.doc_id}] {c.text}" for c in real_chunks)

    for i, case in enumerate(_RUN_CASES):
        question = case["turns"][0]["user"]
        answer = f"It is called Fiber {(500, 100, 200)[i]}."
        _seed_agent_turn(agent_cassettes, question, answer)
        # Two PASS, one FAIL: a genuine mixed rate, never a degenerate 0 or 1.
        verdict = "FAIL" if i == 2 else "PASS"
        _seed_judge_cassette(judge_cassettes, question, answer, ctx, verdict)

    graph = _agent_graph(agent_cassettes)
    judge_gateway = GatewayChatModel(model_id="test-judge", cassette_dir=judge_cassettes, mode="replay")

    report = run(
        _RUN_CASES, graph=graph, judge_gateway=judge_gateway,
        judge_provider="openai", judge_model_id="gpt-5.6-sol",
    )

    assert report.lane == "live"
    assert report.model_id == "openai:gpt-5.6-sol"
    assert report.total_trials == 3
    assert report.total_passes == 2
    assert report.overall_rate == pytest.approx(2 / 3)
    lo, hi = report.overall_ci95
    assert lo < report.overall_rate < hi

    as_dict = report.as_dict()
    assert as_dict["provenance"] == {"lane": "live", "model_id": "openai:gpt-5.6-sol"}
    assert as_dict["overall"]["passes"] == 2
    assert as_dict["overall"]["trials"] == 3
    assert len(as_dict["cases"]) == 3


def test_run_never_gates_a_report_of_all_failures_is_still_a_plain_report(tmp_path):
    """The defining property of this lane (D18): even a red run raises nothing and exits nothing --
    `run()` has no gate concept at all, unlike `judge.live_pr_lane.run`'s own `floors_pass`."""
    agent_cassettes = tmp_path / "agent"
    judge_cassettes = tmp_path / "judge"
    real_chunks = InMemoryRetriever().search_chunks(_QUERY, config=RetrievalConfig())
    ctx = "\n".join(f"[{c.doc_id}] {c.text}" for c in real_chunks)
    case = _RUN_CASES[0]
    question = case["turns"][0]["user"]
    answer = "It is called Fiber 500."
    _seed_agent_turn(agent_cassettes, question, answer)
    _seed_judge_cassette(judge_cassettes, question, answer, ctx, "FAIL")

    graph = _agent_graph(agent_cassettes)
    judge_gateway = GatewayChatModel(model_id="test-judge", cassette_dir=judge_cassettes, mode="replay")

    report = run(
        [case], graph=graph, judge_gateway=judge_gateway,
        judge_provider="anthropic", judge_model_id="claude-opus-4-8",
    )

    assert report.overall_rate == 0.0
    assert report.total_passes == 0
    assert report.total_trials == 1
    # `EvalReport.gate()` must be called explicitly (a threshold/variance_budget argument pair);
    # this module's own source never calls it, the textual proof that a red run has no gate path to
    # trip.
    import inspect

    from judge import full_sweep

    assert ".gate(" not in inspect.getsource(full_sweep)


def test_render_report_names_never_gates_and_the_76_case_honesty_note(tmp_path):
    agent_cassettes = tmp_path / "agent"
    judge_cassettes = tmp_path / "judge"
    real_chunks = InMemoryRetriever().search_chunks(_QUERY, config=RetrievalConfig())
    ctx = "\n".join(f"[{c.doc_id}] {c.text}" for c in real_chunks)
    case = _RUN_CASES[0]
    question = case["turns"][0]["user"]
    answer = "It is called Fiber 500."
    _seed_agent_turn(agent_cassettes, question, answer)
    _seed_judge_cassette(judge_cassettes, question, answer, ctx, "PASS")
    graph = _agent_graph(agent_cassettes)
    judge_gateway = GatewayChatModel(model_id="test-judge", cassette_dir=judge_cassettes, mode="replay")

    report = run(
        [case], graph=graph, judge_gateway=judge_gateway,
        judge_provider="openai", judge_model_id="gpt-5.6-sol",
    )
    rendered = render_report(report, judge_provider="openai", judge_model_id="gpt-5.6-sol")

    assert "NEVER GATES" in rendered
    assert "76 case" in rendered
    assert "openai:gpt-5.6-sol" in rendered


def test_trend_path_is_a_distinct_file_from_the_evalkit_demo_trend():
    from evals.evalkit.report import TREND_PATH as DEMO_TREND_PATH

    assert TREND_PATH != DEMO_TREND_PATH
    assert TREND_PATH.name == "trend.jsonl"
    assert "full_sweep" in str(TREND_PATH)
