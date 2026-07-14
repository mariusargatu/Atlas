"""Decode a ``Trajectory`` from a recorded span tree.

Reuses ``tracing``'s free decoders, the ONE definition of "tools called" / "guard verdicts" / "write
applied", so a trajectory grader and the drift lane read the identical path; a renamed span or a
changed dedup rule moves in ``tracing``, never here. The bound intent is read off the turn span (what
the runtime actually bound for least agency), not re-derived from the raw utterance.
"""
from __future__ import annotations

from tracing import guard_outcomes, tool_calls, write_applied

from evals.trajectory.model import ToolCall, Trajectory


def trajectory_from_spans(spans, *, session_customer_id: str, final_response: str | None) -> Trajectory:
    # One turn per trajectory: intent is read from the turn span, but the tool/guard/write decoders run
    # over ALL spans, so more than one turn would graft one turn's intent onto another's calls. Fail
    # loudly rather than misgrade; a recorded multi-turn thread must be sliced per turn first.
    turns = [s for s in spans if s.kind == "turn"]
    if len(turns) != 1:
        raise ValueError(f"trajectory_from_spans expects exactly one turn, got {len(turns)}")
    intent = turns[0].attributes.get("intent") or "troubleshooting"
    return Trajectory(
        intent=intent,
        session_customer_id=session_customer_id,
        tool_calls=tuple(ToolCall(name, args) for name, args in tool_calls(spans)),
        guard_outcomes=tuple(guard_outcomes(spans)),
        write_applied=write_applied(spans),
        final_response=final_response,
    )
