"""SP10 task 4: `judge.simulator_lane`, hermetic. Every piece of the pass^k computation, the cross
model tier selection, and the whole persona driven episode loop is proven with injected REPLAY
gateways on every one of the three roles (persona player, evaluator, SUT agent) -- zero keys, zero
egress. Only `build_live_persona_gateway`/`build_live_sut_graph_factory`/`main` reach for a real live
provider or a real Postgres/TEI retriever, and none of those live bodies ever run in this file
(mirrors `test_judge_live_pr_lane.py`'s own "every ASSEMBLY function is proven with an injected stub"
scope).
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from replay.cassette_store import seed_cassette
from replay.gateway import GatewayChatModel
from tracing import InMemoryTracer

from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph

from evals.simulation.personas import PERSONAS, Persona

from judge.rubric import RUBRIC_PERSONA_ADHERENCE, RUBRIC_TASK_SUCCESS, prompt as judge_prompt

from judge.simulator_lane import (
    K,
    EpisodeResult,
    PersonaPassKReport,
    _persona_messages,
    drive_persona_episode,
    grade_persona_adherence,
    grade_task_success,
    next_persona_turn,
    persona_adherence_question,
    persona_system_prompt,
    run_persona,
    run_simulator_lane,
    select_driver_and_evaluator_tiers,
    task_success_question,
)

from matrix.live_driver import MissingEnvVarError

_PERSONA = Persona(name="mind-changer", disposition="reverses direction mid conversation", goal="switch plans")


def _seeded_persona_gateway(tmp_path, persona: Persona, transcript, reply: str, model_id: str = "test-persona"):
    seed_cassette(tmp_path, _persona_messages(persona, transcript), {"content": reply, "tool_calls": []}, model_id=model_id)
    return GatewayChatModel(model_id=model_id, cassette_dir=tmp_path, mode="replay")


def _seeded_evaluator_gateway(tmp_path, question: str, answer: str, verdict: str, model_id: str = "test-evaluator"):
    rubric = RUBRIC_PERSONA_ADHERENCE if question == persona_adherence_question(_PERSONA) else RUBRIC_TASK_SUCCESS
    seed_cassette(tmp_path, judge_prompt(rubric, question, answer, ""), {"content": verdict, "tool_calls": []}, model_id=model_id)
    return GatewayChatModel(model_id=model_id, cassette_dir=tmp_path, mode="replay")


# ---- select_driver_and_evaluator_tiers ---------------------------------------------------------------


def test_select_driver_and_evaluator_tiers_needs_both_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    driver, evaluator = select_driver_and_evaluator_tiers()
    assert driver[0] != evaluator[0], "the driver and the evaluator must be two different provider families"
    assert driver == ("openai", "gpt-5.4-nano")
    assert evaluator == ("anthropic", "claude-haiku-4-5-20251001")


def test_select_driver_and_evaluator_tiers_raises_when_only_openai_is_set(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingEnvVarError, match="ANTHROPIC_API_KEY"):
        select_driver_and_evaluator_tiers()


def test_select_driver_and_evaluator_tiers_raises_when_only_anthropic_is_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with pytest.raises(MissingEnvVarError, match="OPENAI_API_KEY"):
        select_driver_and_evaluator_tiers()


def test_select_driver_and_evaluator_tiers_raises_when_neither_is_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(MissingEnvVarError, match="OPENAI_API_KEY.*ANTHROPIC_API_KEY|ANTHROPIC_API_KEY.*OPENAI_API_KEY"):
        select_driver_and_evaluator_tiers()


# ---- next_persona_turn / _persona_messages -----------------------------------------------------------


def test_next_persona_turn_strips_the_end_token_and_reports_done(tmp_path):
    gateway = _seeded_persona_gateway(tmp_path, _PERSONA, (), "When does my contract end? [END]")
    utterance, done = next_persona_turn(gateway, _PERSONA, ())
    assert utterance == "When does my contract end?"
    assert done is True


def test_next_persona_turn_without_the_end_token_reports_not_done(tmp_path):
    gateway = _seeded_persona_gateway(tmp_path, _PERSONA, (), "Tell me about the faster plan.")
    utterance, done = next_persona_turn(gateway, _PERSONA, ())
    assert utterance == "Tell me about the faster plan."
    assert done is False


def test_persona_messages_inverts_roles_for_the_persona_players_own_view():
    transcript = (("customer", "hi"), ("agent", "how can I help?"))
    messages = _persona_messages(_PERSONA, transcript)
    assert messages[0].content == persona_system_prompt(_PERSONA)
    assert isinstance(messages[1], AIMessage) and messages[1].content == "hi"
    assert isinstance(messages[2], HumanMessage) and messages[2].content == "how can I help?"


def test_next_persona_turn_reads_the_transcript_so_far(tmp_path):
    transcript = (("customer", "hi"), ("agent", "how can I help?"))
    gateway = _seeded_persona_gateway(tmp_path, _PERSONA, transcript, "Actually never mind. [END]")
    utterance, done = next_persona_turn(gateway, _PERSONA, transcript)
    assert utterance == "Actually never mind."
    assert done is True


# ---- grade_persona_adherence / grade_task_success ------------------------------------------------------


def test_grade_persona_adherence_parses_pass(tmp_path):
    transcript = (("customer", "I need the fast plan now."), ("agent", "Sure, let me help."))
    gateway = _seeded_evaluator_gateway(tmp_path, persona_adherence_question(_PERSONA), "I need the fast plan now.", "PASS")
    assert grade_persona_adherence(gateway, _PERSONA, transcript) == 1


def test_grade_persona_adherence_parses_fail(tmp_path):
    transcript = (("customer", "As an AI language model, I cannot want a plan."), ("agent", "ok"))
    gateway = _seeded_evaluator_gateway(
        tmp_path, persona_adherence_question(_PERSONA), "As an AI language model, I cannot want a plan.", "FAIL"
    )
    assert grade_persona_adherence(gateway, _PERSONA, transcript) == 0


def test_grade_task_success_folds_actions_into_the_question(tmp_path):
    transcript = (("customer", "switch me to the fast plan"), ("agent", "done, switched you over"))
    actions = (("change_plan", {"plan_id": "plan-fast"}),)
    full_text = "customer: switch me to the fast plan\nagent: done, switched you over"
    gateway = _seeded_evaluator_gateway(tmp_path, task_success_question(_PERSONA, actions), full_text, "PASS")
    assert grade_task_success(gateway, _PERSONA, transcript, actions) == 1


def test_grade_task_success_a_cassette_miss_reads_as_fail_not_a_crash(tmp_path):
    gateway = GatewayChatModel(model_id="test-evaluator", cassette_dir=tmp_path, mode="replay")
    with pytest.raises(Exception):
        # unseeded: judge_label itself does not swallow the miss (that discipline lives in the
        # per item caller, e.g. judge.live_pr_lane.judge_the_items); this asserts the cassette miss
        # really is a hard fail here, not a silent pass.
        grade_task_success(gateway, _PERSONA, (), ())


# ---- EpisodeResult / PersonaPassKReport: pure pass^k statistics ---------------------------------------


def _episode(trial: int, *, adherent: bool, task_pass: bool) -> EpisodeResult:
    return EpisodeResult(
        persona=_PERSONA.name, trial=trial, transcript=(), actions=(), adherent=adherent, task_pass=task_pass
    )


def test_persona_pass_k_report_all_valid_all_pass():
    report = PersonaPassKReport(_PERSONA.name, k=4, episodes=tuple(_episode(i, adherent=True, task_pass=True) for i in range(4)))
    assert report.invalid_count == 0
    assert report.valid_pass_rate == 1.0
    assert report.pass_k_estimate == pytest.approx(1.0)
    assert report.all_valid_trials_passed is True


def test_persona_pass_k_report_mixed_pass_fail():
    episodes = tuple(_episode(i, adherent=True, task_pass=(i < 3)) for i in range(4))  # 3 of 4 pass
    report = PersonaPassKReport(_PERSONA.name, k=4, episodes=episodes)
    assert report.valid_pass_rate == 0.75
    assert report.pass_k_estimate == pytest.approx(0.75**4)
    assert report.all_valid_trials_passed is False


def test_persona_pass_k_report_invalid_episodes_are_excluded_not_counted_as_failures():
    episodes = (
        _episode(0, adherent=False, task_pass=False),  # invalid: persona broke character
        _episode(1, adherent=True, task_pass=True),
        _episode(2, adherent=True, task_pass=True),
        _episode(3, adherent=False, task_pass=False),  # invalid
    )
    report = PersonaPassKReport(_PERSONA.name, k=4, episodes=episodes)
    assert report.invalid_count == 2
    assert len(report.valid_episodes) == 2
    assert report.valid_pass_rate == 1.0  # both SURVIVING trials passed; the invalid ones never count
    assert report.all_valid_trials_passed is True


def test_persona_pass_k_report_every_trial_invalidated_reads_none_not_a_false_reading():
    episodes = tuple(_episode(i, adherent=False, task_pass=False) for i in range(4))
    report = PersonaPassKReport(_PERSONA.name, k=4, episodes=episodes)
    assert report.valid_pass_rate is None
    assert report.pass_k_estimate is None
    assert report.all_valid_trials_passed is None
    assert report.valid_pass_rate_ci95 == (0.0, 1.0)


def test_persona_pass_k_report_render_and_as_dict_are_well_formed():
    episodes = tuple(_episode(i, adherent=True, task_pass=True) for i in range(4))
    report = PersonaPassKReport(_PERSONA.name, k=4, episodes=episodes)
    rendered = report.render()
    assert _PERSONA.name in rendered
    assert "pass^4" in rendered
    d = report.as_dict()
    assert d["k"] == 4
    assert d["valid_count"] == 4
    assert d["invalid_count"] == 0
    assert len(d["episodes"]) == 4


# ---- drive_persona_episode / run_persona / run_simulator_lane: end to end, hermetic --------------------


def _sut_graph_and_backend(cassette_dir):
    gw = GatewayChatModel(model_id="claude-test", cassette_dir=cassette_dir, mode="replay")
    backend = ActionsBackend(IdFactory("ref"))
    graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer(), tracer=InMemoryTracer())
    return graph, backend


def test_drive_persona_episode_single_round_no_action(tmp_path):
    persona_cassettes = tmp_path / "persona"
    agent_cassettes = tmp_path / "agent"

    seed_cassette(
        persona_cassettes, _persona_messages(_PERSONA, ()), {"content": "Is my plan contract-free? [END]", "tool_calls": []},
        model_id="test-persona",
    )
    seed_cassette(
        agent_cassettes, [HumanMessage("Is my plan contract-free?")],
        {"content": "Yes, no fixed contract.", "tool_calls": []}, model_id="claude-test",
    )
    persona_gateway = GatewayChatModel(model_id="test-persona", cassette_dir=persona_cassettes, mode="replay")
    graph, backend = _sut_graph_and_backend(agent_cassettes)

    import asyncio
    transcript, actions = asyncio.run(
        drive_persona_episode(
            _PERSONA, sut_graph=graph, backend=backend, persona_gateway=persona_gateway,
            thread_id="test-episode-1", customer_id="cust_current",
        )
    )

    assert transcript == (
        ("customer", "Is my plan contract-free?"),
        ("agent", "Yes, no fixed contract."),
    )
    assert actions == ()


def test_drive_persona_episode_stops_after_max_turns_if_the_persona_never_signals_done(tmp_path):
    """A live persona player model can fail to emit the end token; the loop must still bound itself
    (`MAX_TURNS`), never spin unboundedly against a real, billed provider."""
    persona_cassettes = tmp_path / "persona"
    agent_cassettes = tmp_path / "agent"

    transcript_so_far: tuple[tuple[str, str], ...] = ()
    for i in range(4):
        seed_cassette(
            persona_cassettes, _persona_messages(_PERSONA, transcript_so_far),
            {"content": f"question number {i}", "tool_calls": []}, model_id="test-persona",
        )
        agent_reply = f"answer number {i}"
        # Seed the agent's reply for exactly the message history this turn will present.
        history: list = []
        for who, text in transcript_so_far:
            history.append(HumanMessage(text) if who == "customer" else AIMessage(text))
        seed_cassette(
            agent_cassettes, history + [HumanMessage(f"question number {i}")],
            {"content": agent_reply, "tool_calls": []}, model_id="claude-test",
        )
        transcript_so_far = transcript_so_far + (("customer", f"question number {i}"), ("agent", agent_reply))

    persona_gateway = GatewayChatModel(model_id="test-persona", cassette_dir=persona_cassettes, mode="replay")
    graph, backend = _sut_graph_and_backend(agent_cassettes)

    import asyncio
    transcript, actions = asyncio.run(
        drive_persona_episode(
            _PERSONA, sut_graph=graph, backend=backend, persona_gateway=persona_gateway,
            thread_id="test-episode-bounded", customer_id="cust_current", max_turns=4,
        )
    )
    assert len(transcript) == 8  # 4 rounds x (customer, agent), never a 5th
    assert actions == ()


def test_run_persona_k_trials_all_adherent_all_pass(tmp_path):
    persona_cassettes = tmp_path / "persona"
    agent_cassettes = tmp_path / "agent"
    evaluator_cassettes = tmp_path / "evaluator"

    seed_cassette(
        persona_cassettes, _persona_messages(_PERSONA, ()), {"content": "Is my plan contract-free? [END]", "tool_calls": []},
        model_id="test-persona",
    )
    seed_cassette(
        agent_cassettes, [HumanMessage("Is my plan contract-free?")],
        {"content": "Yes, no fixed contract.", "tool_calls": []}, model_id="claude-test",
    )
    expected_transcript = (("customer", "Is my plan contract-free?"), ("agent", "Yes, no fixed contract."))
    seed_cassette(
        evaluator_cassettes,
        judge_prompt(RUBRIC_PERSONA_ADHERENCE, persona_adherence_question(_PERSONA), "Is my plan contract-free?", ""),
        {"content": "PASS", "tool_calls": []}, model_id="test-evaluator",
    )
    seed_cassette(
        evaluator_cassettes,
        judge_prompt(
            RUBRIC_TASK_SUCCESS, task_success_question(_PERSONA, ()),
            "customer: Is my plan contract-free?\nagent: Yes, no fixed contract.", "",
        ),
        {"content": "PASS", "tool_calls": []}, model_id="test-evaluator",
    )

    persona_gateway = GatewayChatModel(model_id="test-persona", cassette_dir=persona_cassettes, mode="replay")
    evaluator_gateway = GatewayChatModel(model_id="test-evaluator", cassette_dir=evaluator_cassettes, mode="replay")

    def sut_graph_factory():
        return _sut_graph_and_backend(agent_cassettes)

    report = run_persona(_PERSONA, k=K, sut_graph_factory=sut_graph_factory, persona_gateway=persona_gateway, evaluator_gateway=evaluator_gateway)

    assert report.k == K
    assert len(report.episodes) == K
    assert report.invalid_count == 0
    assert report.valid_pass_rate == 1.0
    assert report.pass_k_estimate == pytest.approx(1.0)
    assert all(e.transcript == expected_transcript for e in report.episodes)


def test_run_persona_invalid_episode_skips_the_task_success_call(tmp_path):
    """When adherence FAILS, `grade_task_success` must never be called (no cassette seeded for it);
    a cassette miss there would raise and fail the test, proving the skip actually happens."""
    persona_cassettes = tmp_path / "persona"
    agent_cassettes = tmp_path / "agent"
    evaluator_cassettes = tmp_path / "evaluator"

    seed_cassette(
        persona_cassettes, _persona_messages(_PERSONA, ()), {"content": "As an AI I cannot want a plan. [END]", "tool_calls": []},
        model_id="test-persona",
    )
    seed_cassette(
        agent_cassettes, [HumanMessage("As an AI I cannot want a plan.")],
        {"content": "I'm not sure I understand.", "tool_calls": []}, model_id="claude-test",
    )
    seed_cassette(
        evaluator_cassettes,
        judge_prompt(RUBRIC_PERSONA_ADHERENCE, persona_adherence_question(_PERSONA), "As an AI I cannot want a plan.", ""),
        {"content": "FAIL", "tool_calls": []}, model_id="test-evaluator",
    )
    # deliberately NO task success cassette seeded: proves grade_task_success is skipped

    persona_gateway = GatewayChatModel(model_id="test-persona", cassette_dir=persona_cassettes, mode="replay")
    evaluator_gateway = GatewayChatModel(model_id="test-evaluator", cassette_dir=evaluator_cassettes, mode="replay")

    def sut_graph_factory():
        return _sut_graph_and_backend(agent_cassettes)

    report = run_persona(_PERSONA, k=1, sut_graph_factory=sut_graph_factory, persona_gateway=persona_gateway, evaluator_gateway=evaluator_gateway)

    assert report.invalid_count == 1
    assert report.episodes[0].adherent is False
    assert report.episodes[0].task_pass is False
    assert report.valid_pass_rate is None


def test_run_simulator_lane_runs_every_persona_in_the_roster(tmp_path):
    """A cheap wiring proof over the REAL roster (`evals.simulation.personas.PERSONAS`): every
    persona gets its own [END]-only turn seeded, so this proves `run_simulator_lane` iterates the
    whole roster, not a hardcoded subset."""
    persona_cassettes = tmp_path / "persona"
    agent_cassettes = tmp_path / "agent"
    evaluator_cassettes = tmp_path / "evaluator"

    for persona in PERSONAS:
        seed_cassette(
            persona_cassettes, _persona_messages(persona, ()), {"content": "hello [END]", "tool_calls": []},
            model_id="test-persona",
        )
        seed_cassette(
            agent_cassettes, [HumanMessage("hello")], {"content": "hi there", "tool_calls": []}, model_id="claude-test",
        )
        seed_cassette(
            evaluator_cassettes,
            judge_prompt(RUBRIC_PERSONA_ADHERENCE, persona_adherence_question(persona), "hello", ""),
            {"content": "PASS", "tool_calls": []}, model_id="test-evaluator",
        )
        seed_cassette(
            evaluator_cassettes,
            judge_prompt(RUBRIC_TASK_SUCCESS, task_success_question(persona, ()), "customer: hello\nagent: hi there", ""),
            {"content": "PASS", "tool_calls": []}, model_id="test-evaluator",
        )

    persona_gateway = GatewayChatModel(model_id="test-persona", cassette_dir=persona_cassettes, mode="replay")
    evaluator_gateway = GatewayChatModel(model_id="test-evaluator", cassette_dir=evaluator_cassettes, mode="replay")

    def sut_graph_factory():
        return _sut_graph_and_backend(agent_cassettes)

    reports = run_simulator_lane(PERSONAS, k=1, sut_graph_factory=sut_graph_factory, persona_gateway=persona_gateway, evaluator_gateway=evaluator_gateway)

    assert len(reports) == len(PERSONAS)
    assert {r.persona for r in reports} == {p.name for p in PERSONAS}
    assert all(r.valid_pass_rate == 1.0 for r in reports)
