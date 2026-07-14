"""The shapes the simulation driver produces and the grader reads. Frozen, so a driven conversation
is a stable fixture. The actions are what the agent actually did across the whole conversation, read
from the stateful actions backend's audit log, not the prose it ended on.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConversationOutcome:
    actions: tuple[tuple[str, Mapping], ...] = ()      # (tool, args) executed across every turn, in order
    final_responses: tuple[str | None, ...] = ()       # the agent's final reply per turn


@dataclass(frozen=True)
class ConversationReport:
    sound: bool
    single_action: bool
    matches_settled: bool
    action_count: int
    reasons: tuple[str, ...] = field(default_factory=tuple)
