"""Replay a recorded ``Conversation`` through the real ``atlas_graph``, one turn at a time on a single
thread, and collect the actions it took across the whole conversation.

Each turn's cassette is keyed on the accumulated message history (the same history the checkpointer
persists), so the gateway resolves the recorded reply for that turn and no live model runs. An action
turn pauses at the confirmation interrupt; the driver resumes with a typed CONFIRM, exactly as a
customer would. The executed actions are read from the stateful backend's audit log, the record of
what actually happened, not the prose the agent ended on. Handles answer turns followed by a terminal
action turn (the mind-changer shape); a conversation with actions mid-way would need per-turn history
reconstruction beyond this.
"""
from __future__ import annotations

from evals.simulation.model import ConversationOutcome


async def drive_conversation(conversation, graph, backend, seed_cassette, cassette_dir, *, thread_id: str) -> ConversationOutcome:
    from langchain_core.messages import AIMessage, HumanMessage  # lazy: pure-grade tests never build a graph
    from langgraph.types import Command

    from atlas.orchestration.atlas_graph import thread_config  # lazy for the same reason (pulls in langgraph)

    history: list = []
    finals: list = []
    # same config the product edge uses: recursion limit tied to the call budget (finding 2)
    config = thread_config(thread_id)

    turns = conversation.turns
    for index, turn in enumerate(turns):
        if turn.tool_calls and index != len(turns) - 1:
            # out of scope by construction: history for an action turn (its tool-call AIMessage + the
            # tool message + the confirmation) is not reconstructed, so a later turn's cassette would
            # mis-key. Fail loudly here instead of surfacing a misleading generic cassette miss.
            raise ValueError(
                "a tool-call (action) turn must be terminal; mid-conversation actions are out of scope"
            )
        user = HumanMessage(turn.user)
        # seed the reply the gateway will replay for this turn's model call (history + this user message)
        seed_cassette(cassette_dir, history + [user], {"content": turn.content, "tool_calls": list(turn.tool_calls)})
        out = await graph.ainvoke(
            {"messages": [user], "session": {"customer_id": conversation.customer_id}}, config
        )
        if "__interrupt__" in out:                       # an action turn paused at the confirmation gate
            out = await graph.ainvoke(Command(resume="CONFIRM"), config)
        finals.append(out.get("final_response"))
        if not turn.tool_calls:                          # only answer turns extend the reconstructed history
            history = history + [user, AIMessage(content=turn.content)]

    applied = backend.applied(conversation.customer_id)  # the audit log: what actually executed
    actions = tuple((a.tool, dict(a.args)) for a in applied)
    return ConversationOutcome(actions=actions, final_responses=tuple(finals))
