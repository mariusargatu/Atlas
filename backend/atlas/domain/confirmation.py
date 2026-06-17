"""The propose -> confirm -> execute protocol, as pure domain logic.

The LangGraph `interrupt()` + checkpointer is the orchestration realization; this is the
rule it enforces: nothing irreversible happens without an explicit typed confirmation, and
execution runs the *stored pending* action (with its bound idempotency key), never a drifted
one.
"""
from __future__ import annotations

from dataclasses import dataclass

from atlas.domain.actions import ActionResult, ActionsBackend
from atlas.domain.guard import WRITE_TOOLS


@dataclass
class PendingAction:
    tool: str
    args: dict
    idempotency_key: str
    customer_id: str


class ConfirmationError(Exception):
    pass


def execute_if_confirmed(
    pending: PendingAction,
    typed_confirmation: str,
    backend: ActionsBackend,
    *,
    expected: str = "CONFIRM",
) -> ActionResult:
    if typed_confirmation != expected:
        raise ConfirmationError("an irreversible action needs a typed confirmation, not a bare yes")
    if pending.tool not in WRITE_TOOLS:
        raise ConfirmationError(f"unknown action {pending.tool!r}")
    # Execute the *stored* pending action with its bound key. Dispatch is generic over the write
    # surface; the backend dedups on the key so a retry never double applies.
    return backend.apply(pending.tool, pending.customer_id, pending.args, pending.idempotency_key)
