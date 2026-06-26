"""The trace core: the assertion target every later part reads from (principle 4).

A turn produces a tree of spans (turn / llm / tool / guard / node). Spans order by a monotonic
``SpanSequence``, NEVER by the frozen clock (all spans would tie). Guard verdicts are our own
domain logic, so guard nodes annotate spans **explicitly**. The LangGraph callback handler that
feeds native LLM/tool spans in dev/prod (P9) cannot see them (ADR-017). In CI the graph runs
sequentially, so span order is deterministic and the tree is byte stable.

Trajectory tests (P4), simulation (P7), attack success scoring (P8), and the production loop (P9)
all read from this. They assert the **tree** (parent/child + sequence), never a flat,
time ordered list. The default is ``NullTracer`` so runtime code never depends on being observed.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Protocol

from determinism.sources import SpanSequence


@dataclass(frozen=True)
class Span:
    seq: int
    name: str
    kind: str                       # "turn" | "llm" | "tool" | "guard" | "node"
    parent: Optional[int]
    attributes: Mapping             # read only view (MappingProxyType), frozen in value, not just field


# ---- span decoders (the one definition of "tools called" / "guard verdicts") ----
# Free functions over any span sequence, so a reader of a recorded trace (e.g. the drift lane's
# DecisionRecord) and the live InMemoryTracer share ONE definition of the trajectory: a renamed
# kind or a dedup rule changes here, never in two places that can silently disagree.
def spans_of_kind(spans, kind: str) -> list["Span"]:
    return [s for s in spans if s.kind == kind]


def tool_names(spans) -> list[str]:
    """The trajectory: tool span names in call order."""
    return [s.name for s in spans_of_kind(spans, "tool")]


def guard_outcomes(spans) -> list[tuple[str, bool]]:
    """(guard_name, ok) per guard span, in order. ok defaults False when a span is unannotated."""
    return [(s.name, bool(s.attributes.get("ok"))) for s in spans_of_kind(spans, "guard")]


def write_applied(spans) -> bool:
    """Did a write land? True iff some ``execute_action`` node span carries a truthy ``applied``.
    The one definition, structural and never from prose, shared by the eval graders and the drift
    lane so a renamed span or a changed rule moves here, not in two places that can silently diverge."""
    return any(s.name == "execute_action" and s.attributes.get("applied") for s in spans_of_kind(spans, "node"))


class Tracer(Protocol):
    """The trace port. Adapters: ``InMemoryTracer`` (CI), an OTel/Langfuse handler (dev/prod)."""

    def open(self, name: str, kind: str, parent: Optional[int] = None, **attrs) -> int: ...
    def annotate(self, seq: int, **attrs) -> None: ...


class NullTracer:
    """No op tracer. Runtime code is instrumented unconditionally, and observation is opt in."""

    def open(self, name: str, kind: str, parent: Optional[int] = None, **attrs) -> int:
        return -1

    def annotate(self, seq: int, **attrs) -> None:
        return None


class InMemoryTracer:
    """The CI adapter: an assertable, deterministic span tree ordered by ``SpanSequence``."""

    def __init__(self, spans: Optional[SpanSequence] = None) -> None:
        self._seq = spans or SpanSequence()
        self._spans: dict[int, Span] = {}
        self._order: list[int] = []

    def open(self, name: str, kind: str, parent: Optional[int] = None, **attrs) -> int:
        seq = self._seq.next()
        self._spans[seq] = Span(seq, name, kind, parent, MappingProxyType(dict(attrs)))
        self._order.append(seq)
        return seq

    def annotate(self, seq: int, **attrs) -> None:
        cur = self._spans.get(seq)
        if cur is not None:  # immutable update: replace, never mutate in place
            merged = MappingProxyType({**cur.attributes, **attrs})
            self._spans[seq] = Span(cur.seq, cur.name, cur.kind, cur.parent, merged)

    # ---- assertion helpers (what trajectory/security/production tests read) ----
    @property
    def spans(self) -> list[Span]:
        return [self._spans[s] for s in self._order]

    def of_kind(self, kind: str) -> list[Span]:
        return spans_of_kind(self.spans, kind)

    def tool_order(self) -> list[str]:
        """The trajectory: tool names in the order they were called."""
        return tool_names(self.spans)

    def guard_verdicts(self) -> list[Span]:
        return self.of_kind("guard")


__all__ = [
    "InMemoryTracer", "NullTracer", "Span", "Tracer",
    "spans_of_kind", "tool_names", "guard_outcomes", "write_applied",
]
