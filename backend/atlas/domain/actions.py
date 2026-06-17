"""The actions backend, the write surface. Simulated, stateful, idempotency key aware.

A call that times out and is retried with the same idempotency key applies exactly once
(the distributed systems lesson, now with a customer's bill attached). The id generator is
injected (a port), so the domain never imports the harness.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionResult:
    reference: str
    applied: bool  # True if this call applied the change; False if it was a dedup'd retry


@dataclass(frozen=True)
class AppliedAction:
    customer_id: str
    tool: str
    args: tuple  # sorted args items, hashable, comparable audit record


class ActionsBackend:
    """A generic, idempotency keyed write surface. Any write tool dedups on the same key, so a
    timed out retry applies once; the audit log is queryable per customer for fidelity tests.

    With a ``writer`` injected (``apply_write`` from the accounts store), a confirmed write is also
    applied through to account state, so a later read reflects it. The writer runs only on the first
    occurrence of a key. The dedup branch returns without re applying, which is what keeps a
    timed out retry from changing the account twice. Without a writer the backend is audit only
    (the read after write tests inject one; the rest stay state free).
    """

    def __init__(self, ids, *, writer=None) -> None:
        self._ids = ids                                   # anything with .next() -> str
        self._writer = writer                             # callable(tool, customer_id, args) | None
        self._applied: dict[str, str] = {}                # idempotency_key -> reference
        self._log: list[AppliedAction] = []               # ordered audit of applied writes

    def apply(self, tool: str, customer_id: str, args: dict, idempotency_key: str) -> ActionResult:
        if idempotency_key in self._applied:
            return ActionResult(self._applied[idempotency_key], applied=False)
        reference = self._ids.next()
        self._applied[idempotency_key] = reference
        self._log.append(AppliedAction(customer_id, tool, tuple(sorted(args.items()))))
        if self._writer is not None:                      # write through, exactly once per key
            self._writer(tool, customer_id, args)
        return ActionResult(reference, applied=True)

    # ---- back compat convenience + audit queries ----
    def change_plan(self, customer_id: str, plan_id: str, idempotency_key: str) -> ActionResult:
        return self.apply("change_plan", customer_id, {"plan_id": plan_id}, idempotency_key)

    def applied(self, customer_id: str, tool: str | None = None) -> list[AppliedAction]:
        return [a for a in self._log if a.customer_id == customer_id and (tool is None or a.tool == tool)]

    def change_count(self, customer_id: str) -> int:
        return len(self.applied(customer_id, "change_plan"))
