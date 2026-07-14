"""The shapes a trajectory grader reads. Frozen, so a recorded trajectory is a stable fixture (a
decoded call's args are wrapped read-only, the same ``MappingProxyType`` discipline the trace core
uses for span attributes). A ``Trajectory`` is what the span tree decodes to: the ordered tool calls
(name and args),
the guard verdicts, whether a write landed, the bound intent, the session's customer id, and the
final response. Everything a path assertion needs, and nothing the model got to author.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from determinism.canonical import canonical_json


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        # wrap args read-only so a decoded call cannot be mutated in place (frozen=True only blocks
        # attribute rebinding, not in-place dict mutation) — the repo's immutability contract. `or {}`
        # tolerates a tool span that recorded null args (the wire form of "no arguments") rather than
        # crashing on dict(None).
        object.__setattr__(self, "args", MappingProxyType(dict(self.args or {})))

    def __eq__(self, other: object) -> bool:
        # Agree with __hash__. The dataclass-generated __eq__ compares args by Python value equality
        # (1 == True == 1.0), but __hash__ keys off canonical_json, which tags those apart — so two
        # equal ToolCalls could hash differently and both survive set dedup, breaking the "usable in a
        # set" contract below. Compare on the SAME canonical form so eq and hash never disagree; a tool
        # argument's type is part of the decision (int 1 is not bool True is not float 1.0).
        if not isinstance(other, ToolCall):
            return NotImplemented
        return (self.name == other.name
                and canonical_json(dict(self.args)) == canonical_json(dict(other.args)))

    def __hash__(self) -> int:
        # frozen=True advertises hashability, but a MappingProxyType field is unhashable so the
        # generated __hash__ would raise; hash a canonical serialization of the args instead — stable
        # AND safe for nested dict/list arg values (tuple(sorted(items)) raises on those), so a ToolCall
        # is genuinely usable in a set. Paired with the __eq__ above so equal calls always hash equal;
        # both an explicit __eq__ and __hash__ are left untouched by @dataclass.
        return hash((self.name, canonical_json(dict(self.args))))


@dataclass(frozen=True)
class Trajectory:
    intent: str                                   # the intent the runtime BOUND for the turn (least agency)
    session_customer_id: str                      # identity from the session, never from the model
    tool_calls: tuple[ToolCall, ...] = ()
    guard_outcomes: tuple[tuple[str, bool], ...] = ()   # (guard_name, ok) in order
    write_applied: bool = False
    final_response: str | None = None


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrajectoryReport:
    """The three levels and the two families, in one record.

    End-to-end: ``goal_completed`` (the oracle question a stakeholder asks). Across the path:
    ``sound`` (a property derived from the five path/atom flags, so it can never drift from its parts)
    and its parts. At the node: ``failing_reasons``. Efficiency family: ``tool_call_count`` /
    ``guard_blocks``. Quality family: ``sound`` / ``goal_completed``.

    What ``sound`` means, stated honestly: it is the SYSTEM's path validity. The runtime fails an
    unsafe call closed BEFORE its tool span opens, so a runtime-decoded trace never carries the bad
    call; on real traces the atom/single/orphan rules are a backstop that mainly bites hand-authored
    trajectories, and what the model ATTEMPTED and the runtime refused is reported separately as
    ``guard_blocks``. So ``sound`` reads "the system stayed on a valid path", not "the model never
    reached for anything it should not have".
    """

    goal_completed: bool
    atoms_ok: bool
    single_write: bool
    no_orphan_write: bool
    terminated: bool
    within_budget: bool
    failing_reasons: tuple[str, ...]
    tool_call_count: int
    guard_blocks: int

    @property
    def sound(self) -> bool:
        return (self.atoms_ok and self.single_write and self.no_orphan_write
                and self.terminated and self.within_budget)
