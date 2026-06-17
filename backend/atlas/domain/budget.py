"""The per-turn call budget: how many tool calls and retrieval rounds one turn may spend.

The runtime derives its graph recursion limit from DEFAULT_BUDGET and calls this exact `check_budget`
live in its read loop and before every write (so it never ships a turn this check would fail); the
monitor and trajectory lanes run the SAME check_budget over recorded tool-call sequences. One
mechanism, one arithmetic. Pure stdlib plus the domain tool sets: no model call, no clock, no
framework import.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from atlas.domain.binding import KNOWLEDGE_TOOLS


@dataclass(frozen=True)
class Budget:
    max_tool_calls: int
    max_retrieval_rounds: int


# Single-sourced so the runtime and the eval lanes cannot drift on what "within budget" means.
DEFAULT_BUDGET = Budget(max_tool_calls=6, max_retrieval_rounds=3)

# The retrieval tool set for budgeting: derived from the runtime binding (KNOWLEDGE_TOOLS) so the
# budget lanes and the graph agree on what counts as a "retrieval round".
DEFAULT_RETRIEVAL_TOOLS = frozenset(KNOWLEDGE_TOOLS)

# The graph superstep ceiling for a budget-legal turn, tied to the call budget instead of langgraph's
# implicit 25. A turn of max_tool_calls single-call reads costs 2 supersteps per read round trip
# (agent -> tools_read) plus a two-superstep answer tail (the final agent, then pre_render_guard):
# 2 * max_tool_calls + 2 supersteps. langgraph stops a run when the step counter EXCEEDS
# recursion_limit (stop = recursion_limit + 1), so the limit must sit one above that superstep count
# for the last legal turn to finish rather than raise: 2 * max_tool_calls + 3. Empirically pinned
# against langgraph 1.0.1 by testing/tests/test_budgets.py and the acceptance test in
# test_atlas_graph (max_tool_calls sequential single-call reads must COMPLETE).
RECURSION_LIMIT = 2 * DEFAULT_BUDGET.max_tool_calls + 3


@dataclass(frozen=True)
class BudgetReport:
    ok: bool
    tool_calls: int
    retrieval_rounds: int
    reasons: tuple[str, ...] = field(default_factory=tuple)


def check_budget(
    tools_called: Sequence[str],
    budget: Budget,
    *,
    retrieval_tools: Iterable[str],
) -> BudgetReport:
    """Report whether a recorded tool-call sequence stayed within budget. Deterministic; no LLM."""
    retrieval = frozenset(retrieval_tools)
    tool_calls = len(tools_called)
    retrieval_rounds = sum(1 for t in tools_called if t in retrieval)
    reasons: list[str] = []
    if tool_calls > budget.max_tool_calls:
        reasons.append(f"tool-call budget exceeded: {tool_calls} > {budget.max_tool_calls}")
    if retrieval_rounds > budget.max_retrieval_rounds:
        reasons.append(
            f"retrieval retry storm: {retrieval_rounds} rounds > {budget.max_retrieval_rounds}"
        )
    return BudgetReport(not reasons, tool_calls, retrieval_rounds, tuple(reasons))
