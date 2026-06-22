"""The recorded persona conversations the hermetic gate replays (ADR-019: sims recorded once, replayed
deterministically). Each turn carries the user's message and the agent response the gateway replays
for that turn, so driving the conversation through the real graph is byte-stable, no live model.

The cold open is the mind-changer. She wants the faster plan, wavers toward a cheaper one, then
settles back on the faster plan. The settled intent is the plan she landed on; the walked-back plan is
the one she talked herself out of. Confirming the walked-back plan is the single most important failure
this suite catches, and it lives in the seam between turns, not in any one of them.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    user: str
    content: str = ""                       # the agent's replayed reply (answer turns)
    tool_calls: tuple = ()                  # the agent's replayed tool call(s) (an action turn)


@dataclass(frozen=True)
class Conversation:
    persona: str
    customer_id: str
    turns: tuple[Turn, ...]
    settled_plan_id: str | None            # gold: the plan the customer landed on (None = a chat-only conversation)
    walked_back_plan_id: str | None = None  # the option she considered and rejected


MIND_CHANGER = Conversation(
    persona="mind-changer",
    customer_id="cust_legacy_term",
    settled_plan_id="plan_current_fast",
    walked_back_plan_id="plan_legacy_value",
    turns=(
        Turn("I want to upgrade to the faster plan.",
             content="The Fiber 100 plan gives you unlimited data at a higher monthly price."),
        Turn("Actually, what about a cheaper option?",
             content="The Fiber 100 Legacy plan is cheaper but keeps a monthly data cap."),
        Turn("Let's go with the faster plan after all, switch me over.",
             tool_calls=({"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"},)),
    ),
)

# The same reversal, recorded from an agent that lost track of the settled intent: she settled on the
# faster plan, but it confirms the cheaper one she walked back (a real, offered plan, so the write
# actually lands). This is the cold-open failure as a recorded conversation; the grader must catch it
# end-to-end.
MIND_CHANGER_WRONG = Conversation(
    persona="mind-changer",
    customer_id="cust_legacy_term",
    settled_plan_id="plan_current_fast",       # she landed on Fiber 100; the agent confirmed the wrong plan
    walked_back_plan_id="plan_legacy_value",
    turns=(
        MIND_CHANGER.turns[0],
        MIND_CHANGER.turns[1],
        Turn("Let's go with the faster plan after all, switch me over.",
             tool_calls=({"name": "change_plan", "args": {"plan_id": "plan_legacy_value"}, "id": "c1"},)),
    ),
)
