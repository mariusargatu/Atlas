"""Agent simulation, hermetic: the failures that scare you build across a conversation, not
within one turn. A persona-driven conversation is recorded once and replayed as a
deterministic fixture through the real atlas_graph (ADR-019), and the multi-turn assertions grade the
whole conversation, not the last reply. The cold open is the mind-changer: she reverses direction, and
the agent must confirm the plan she landed on, not the one she walked back two turns ago. That failure
does not exist in any single turn, which is exactly why the static golden set never caught it.
"""
from __future__ import annotations

import pytest

from evals.datasets.simulation_golden import MIND_CHANGER, MIND_CHANGER_WRONG, Conversation, Turn
from evals.simulation.driver import drive_conversation
from evals.simulation.grade import grade_conversation
from evals.simulation.model import ConversationOutcome
from evals.simulation.personas import PERSONAS


# --- pure conversation grading, with teeth ---

def test_confirming_the_settled_plan_passes_the_walked_back_one_fails():
    settled = ConversationOutcome(actions=(("change_plan", {"plan_id": MIND_CHANGER.settled_plan_id}),))
    assert grade_conversation(settled, MIND_CHANGER).sound is True

    # the cold-open failure: the agent confirms the plan she talked herself out of
    walked_back = ConversationOutcome(actions=(("change_plan", {"plan_id": MIND_CHANGER.walked_back_plan_id}),))
    report = grade_conversation(walked_back, MIND_CHANGER)
    assert report.sound is False and report.matches_settled is False


def test_more_than_one_action_across_a_conversation_is_unsound():
    two = ConversationOutcome(actions=(
        ("change_plan", {"plan_id": MIND_CHANGER.settled_plan_id}),
        ("reset_modem", {}),
    ))
    report = grade_conversation(two, MIND_CHANGER)
    assert report.sound is False and report.single_action is False


def test_an_agent_that_went_silent_when_a_change_was_due_is_unsound():
    silent = ConversationOutcome(actions=())          # settled intent set, but no action taken
    report = grade_conversation(silent, MIND_CHANGER)
    assert report.sound is False and report.matches_settled is False
    assert "took no action" in report.reasons[0]


def test_a_chat_only_conversation_must_take_no_action():
    chat_only = Conversation(persona="confused", customer_id="cust_current", turns=(), settled_plan_id=None)
    assert grade_conversation(ConversationOutcome(actions=()), chat_only).sound is True
    took_action = ConversationOutcome(actions=(("change_plan", {"plan_id": "plan_current_fast"}),))
    report = grade_conversation(took_action, chat_only)
    assert report.sound is False and "nobody asked for" in report.reasons[0]


def test_the_persona_roster_is_populated_and_named():
    names = {p.name for p in PERSONAS}
    assert "mind-changer" in names
    assert len(PERSONAS) >= 4
    for persona in PERSONAS:
        assert persona.disposition and persona.goal


# --- through the real graph: the mind-changer replayed as a fixture ---

@pytest.mark.asyncio
async def test_the_mind_changer_confirms_the_plan_she_landed_on(build_replay_graph, seed_cassette, tmp_path):
    """Drive the reversal conversation through the real atlas_graph. Across the whole conversation the
    agent takes exactly one action, on the plan she settled on (Fast), never the cheaper one she
    considered and rejected. One green assertion where the cold open was an angry customer."""
    graph, _tracer, backend = build_replay_graph()
    outcome = await drive_conversation(MIND_CHANGER, graph, backend, seed_cassette, tmp_path, thread_id="sim-mc")

    report = grade_conversation(outcome, MIND_CHANGER)
    assert report.action_count == 1                                      # the conversation may wander; the action may not
    assert outcome.actions[0] == ("change_plan", {"plan_id": MIND_CHANGER.settled_plan_id})
    assert "Your reference is" in (outcome.final_responses[-1] or "")    # the confirm gate executed the write
    assert report.sound is True


@pytest.mark.asyncio
async def test_the_agent_that_confirms_the_walked_back_plan_is_caught_end_to_end(build_replay_graph, seed_cassette, tmp_path):
    """The cold-open failure, driven through the REAL graph: the agent loses the thread and confirms the
    plan she walked back. The driver surfaces the wrong-plan action from the audit log and the grader
    fails it closed. This is the negative direction the baked-in cassette cannot express above."""
    graph, _tracer, backend = build_replay_graph()
    outcome = await drive_conversation(MIND_CHANGER_WRONG, graph, backend, seed_cassette, tmp_path, thread_id="sim-mc-bad")

    assert outcome.actions == (("change_plan", {"plan_id": MIND_CHANGER_WRONG.walked_back_plan_id}),)
    report = grade_conversation(outcome, MIND_CHANGER_WRONG)
    assert report.sound is False and report.matches_settled is False


@pytest.mark.asyncio
async def test_the_driver_rejects_a_non_terminal_action_turn():
    """A tool-call turn must be terminal; an out-of-scope fixture fails loudly, not as a cassette miss."""
    bad = Conversation(
        persona="x", customer_id="cust_current", settled_plan_id=None,
        turns=(
            Turn("switch me now", tool_calls=({"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"},)),
            Turn("and one more thing"),
        ),
    )
    with pytest.raises(ValueError):   # raised before any graph call, so the dummies are never used
        await drive_conversation(bad, graph=None, backend=None, seed_cassette=None, cassette_dir=None, thread_id="x")
