"""The call budget: the deterministic slice of monitoring that gates. Given a recorded tool-call
sequence, did the turn stay within a sane number of calls, and did it avoid a retrieval retry storm
(the same search fired over and over in a loop the model cannot feel itself running). Pure stdlib,
exact over recorded numbers, so it runs in the hermetic lane; the same counts trended over live
traffic are the operator lane's non-gating alarm.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from atlas.domain.binding import KNOWLEDGE_TOOLS


@dataclass(frozen=True)
class Budget:
    max_tool_calls: int
    max_retrieval_rounds: int


# The operator/sample default, single-sourced so the sample lanes (task monitor / task trajectory) and
# the trajectory tests cannot drift on what "within budget" means.
DEFAULT_BUDGET = Budget(max_tool_calls=6, max_retrieval_rounds=3)

# The retrieval tool set for budgeting, DEFAULT_BUDGET's companion: derived from the runtime binding
# (KNOWLEDGE_TOOLS) and single-sourced so the sample lanes (task monitor / task trajectory) and the
# budget tests cannot drift on what counts as a "retrieval round".
DEFAULT_RETRIEVAL_TOOLS = frozenset(KNOWLEDGE_TOOLS)


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
