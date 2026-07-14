"""The multi-turn assertions: grade the whole conversation, not the last reply. Two questions carry
the cold open: did the agent take exactly one action across the whole conversation (it may chat
freely; it may act once), and did that action land on the settled intent, the plan the customer
finally chose, rather than one she walked back midway. Pure over the executed-action record, so it
gates deterministically once the conversation is a fixture.
"""
from __future__ import annotations

from evals.simulation.model import ConversationOutcome, ConversationReport


def grade_conversation(outcome: ConversationOutcome, conversation) -> ConversationReport:
    """Sound iff the conversation took at most one action AND that action matches the settled intent
    (or took none, when the conversation was chat-only). ``conversation`` supplies the gold settled
    intent, an oracle label a human authored, not something the trajectory can guess from its shape."""
    actions = outcome.actions
    reasons: list[str] = []

    single_action = len(actions) <= 1
    if not single_action:
        reasons.append(f"{len(actions)} actions in one conversation; the action may not wander, at most one")

    if conversation.settled_plan_id is None:
        matches_settled = len(actions) == 0
        if not matches_settled:
            reasons.append("a chat-only conversation took an action nobody asked for")
    else:
        matches_settled = (
            len(actions) == 1
            and actions[0][0] == "change_plan"
            and actions[0][1].get("plan_id") == conversation.settled_plan_id
        )
        if not matches_settled:
            if len(actions) == 1 and actions[0][0] == "change_plan":
                reasons.append(
                    f"the action landed on {actions[0][1].get('plan_id')!r}, "
                    f"not the settled intent {conversation.settled_plan_id!r}"
                )
            elif len(actions) == 0:
                reasons.append(f"took no action to complete the settled intent {conversation.settled_plan_id!r}")
            elif len(actions) == 1:  # a single action, but not a change_plan (e.g. reset_modem)
                reasons.append(
                    f"the settled intent {conversation.settled_plan_id!r} needed a change_plan, "
                    f"but the action was {actions[0][0]!r}"
                )
            # the 2+-action case is already reported by the single-action count reason above

    return ConversationReport(
        sound=single_action and matches_settled,
        single_action=single_action,
        matches_settled=matches_settled,
        action_count=len(actions),
        reasons=tuple(reasons),
    )
