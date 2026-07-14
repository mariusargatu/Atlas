"""The whole path: assert over the sequence, not any single call. The questions are about
order and restraint. Did at most one write happen, did every action have a matching request (no
orphan action the agent helped itself to), did the path terminate rather than circle, and did it stay
within a sane budget. ``grade_trajectory`` rolls these up at the three levels (end-to-end, path, node)
and the two families (efficiency, quality). Reuses ``guard`` and ``monitor.budget`` so the path rules
are single-sourced with the runtime.
"""
from __future__ import annotations

from atlas.domain import guard as guardrules

from evals.monitor.budget import DEFAULT_RETRIEVAL_TOOLS, Budget, check_budget
from evals.trajectory.atom import grade_tool_call
from evals.trajectory.model import Trajectory, TrajectoryReport, Verdict


def _writes(traj: Trajectory) -> list[str]:
    """The write tool calls in the turn, by name: one home for 'what counts as a write'."""
    return [c.name for c in traj.tool_calls if c.name in guardrules.WRITE_TOOLS]


def check_single_write(traj: Trajectory) -> Verdict:
    """At most one write per turn. The runtime's read+write-batch rule (guard.check_single_write) is
    per MODEL BATCH; at turn scope only the >1-write invariant is meaningful, because a legitimate
    action turn may read then write across two batches (which the runtime allows), and the flattened
    trajectory would otherwise trip a mixed-batch clause that has no meaning across batch boundaries."""
    if len(_writes(traj)) > 1:
        return Verdict(False, ("more than one write in a turn",))
    return Verdict(True)


def check_no_orphan_write(traj: Trajectory) -> Verdict:
    """Every write must have a matching request behind it. A write on a non-action turn is an action
    the agent took on its own initiative, a failure even when it succeeds (the bill-dispute story)."""
    writes = _writes(traj)
    if writes and traj.intent != "action":
        return Verdict(False, (f"orphan action {writes[0]!r} on a {traj.intent} turn",))
    return Verdict(True)


def check_terminated(traj: Trajectory) -> Verdict:
    """The path stopped, cleanly, on a response, rather than circling."""
    if traj.final_response is not None:   # "" is an (empty) answer and a clean end; None never answered
        return Verdict(True)
    return Verdict(False, ("trajectory did not terminate with a response",))


def check_within_budget(traj: Trajectory, budget: Budget) -> Verdict:
    """No retry storm, no reading the same record forty times in a loop the model cannot feel."""
    report = check_budget([c.name for c in traj.tool_calls], budget, retrieval_tools=DEFAULT_RETRIEVAL_TOOLS)
    return Verdict(report.ok, report.reasons)


def grade_trajectory(traj: Trajectory, *, budget: Budget, goal_met: bool) -> TrajectoryReport:
    """The path graded whole. ``goal_met`` is the end-to-end oracle verdict (did the task actually
    succeed), supplied by the caller because "did the account end on the lower plan" is an oracle
    question, not something the trajectory can guess from its own shape."""
    atoms = [grade_tool_call(c, intent=traj.intent, session_customer_id=traj.session_customer_id)
             for c in traj.tool_calls]
    single = check_single_write(traj)
    orphan = check_no_orphan_write(traj)
    terminated = check_terminated(traj)
    within = check_within_budget(traj, budget)
    reasons = tuple(r for v in [*atoms, single, orphan, terminated, within] for r in v.reasons)
    guard_blocks = sum(1 for _name, ok in traj.guard_outcomes if not ok)
    return TrajectoryReport(
        goal_completed=goal_met,
        atoms_ok=all(v.ok for v in atoms),   # sound is a derived property over these five flags
        single_write=single.ok,
        no_orphan_write=orphan.ok,
        terminated=terminated.ok,
        within_budget=within.ok,
        failing_reasons=reasons,
        tool_call_count=len(traj.tool_calls),
        guard_blocks=guard_blocks,
    )
