"""The propose -> confirm -> execute machine as a LangGraph graph.

`interrupt()` + the checkpointer IS the confirmation gate. The side effect that binds the
idempotency key lives in `propose`, BEFORE the interrupt checkpoint, so it is never re run on
resume (the `confirm_interrupt`-side effect free footgun). Execution runs the *stored pending*
action with its bound key, so a timed out retry de duplicates.
"""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from atlas.domain.actions import ActionsBackend
from atlas.domain.confirmation import ConfirmationError, PendingAction, execute_if_confirmed


class ConfirmState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pending: Optional[dict]
    result: Optional[dict]


def build_confirm_graph(backend: ActionsBackend, ids, customer_id: str, plan_id: str, checkpointer):
    def propose(state: ConfirmState) -> dict:
        # The idempotency key is minted HERE, before the interrupt checkpoint, so it is
        # durably bound once and a resume/retry reuses it.
        return {
            "pending": {
                "tool": "change_plan",
                "args": {"plan_id": plan_id},
                "idempotency_key": ids.next(),
                "customer_id": customer_id,
            }
        }

    def confirm(state: ConfirmState) -> dict:
        typed = interrupt({"proposal": state["pending"]})  # pause; re entered with the typed value
        pending = PendingAction(**state["pending"])
        try:
            res = execute_if_confirmed(pending, typed, backend)
            return {"result": {"reference": res.reference, "applied": res.applied}}
        except ConfirmationError as exc:
            return {"result": {"error": str(exc)}}

    g = StateGraph(ConfirmState)
    g.add_node("propose", propose)
    g.add_node("confirm", confirm)
    g.add_edge(START, "propose")
    g.add_edge("propose", "confirm")
    g.add_edge("confirm", END)
    return g.compile(checkpointer=checkpointer)
