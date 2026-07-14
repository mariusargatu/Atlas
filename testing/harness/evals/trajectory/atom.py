"""The atom: grade a single tool call, because a path is only as trustworthy as its steps (doc 08).

Four rules, and they are not a rubric, they are invariants a rule enforces, the same way at 2am as at
2pm. Three are checkable without the model and gate here: the right tool for the intent, the arguments
in bounds, and the id scoped to the session. The fourth, arguments drawn from what the customer
actually said, needs the request and is the judged part (operator lane). All are binary and
fail-closed: a call that targets the wrong account does not get nine out of ten for the rest.

Every rule REUSES the runtime's own guard/binding, so the grader can never pass a call the runtime
would refuse (the "an eval must not grade more leniently than the runtime" discipline).
"""
from __future__ import annotations

from atlas.domain import guard as guardrules
from atlas.domain.binding import is_reachable

from evals.trajectory.model import ToolCall, Verdict


def grade_tool_call(call: ToolCall, *, intent: str, session_customer_id: str) -> Verdict:
    """Binary, fail-closed. Reasons list every rule that failed (empty when the call is clean)."""
    reasons: list[str] = []
    # right tool for the intent: a tool not bound to this turn is unreachable, not merely discouraged
    if not is_reachable(intent, call.name):
        reasons.append(f"tool {call.name!r} is not reachable on a {intent} turn")
    # arguments in bounds: no internal-only plan, no bogus add-on, no invented slot (reads pass through)
    bounds = guardrules.check_value_bounds(call.name, dict(call.args))
    if not bounds.ok:
        reasons.append(bounds.reason)
    # id scoped to the session: mirror the runtime EXACTLY. pre_action_guard computes
    # check_scope(args.get("customer_id", session_id), session_id): an ABSENT key defaults to the
    # session id and passes (reads omit it, the runtime injects it), but a PRESENT key — including a
    # present-and-None one — is a model-supplied id that must match, or the eval would grade clean a
    # call the runtime fails closed (its .get returns the None value, not the default). Defaulting to
    # the session id makes the absent case a no-op, so a clean read is never flagged.
    claimed = call.args.get("customer_id", session_customer_id)
    scope = guardrules.check_scope(claimed, session_customer_id)
    if not scope.ok:
        reasons.append(scope.reason)
    return Verdict(not reasons, tuple(reasons))
