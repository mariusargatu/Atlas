"""The trace core: the assertion target every later part reads from.

A turn produces a tree of spans (turn / llm / tool / guard / node). Spans order by a monotonic
``SpanSequence``, never by the frozen clock (all spans would tie). Guard verdicts are our own
domain logic, so guard nodes annotate spans **explicitly**. The LangGraph callback handler that
feeds native LLM/tool spans in dev/prod cannot see them (ADR-017). In CI the graph runs
sequentially, so span order is deterministic and the tree is byte stable.

Trajectory tests, simulation, attack success scoring, and the production loop
all read from this. They assert the **tree** (parent/child + sequence), never a flat,
time ordered list. The default is ``NullTracer`` so runtime code never depends on being observed.
"""
from __future__ import annotations

import json
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


def tool_calls(spans) -> list[tuple[str, Mapping]]:
    """The trajectory WITH arguments: (tool name, its args) per tool span, in call order.

    `tool_names` keeps only the name; but for a WRITE the decision is the name AND its arguments
    (change_plan to WHICH plan), so the drift lane reads this to catch a same tool, different argument
    move that a name only trajectory is blind to. Args come straight off the tool span
    (``attributes["args"]``, recorded by the graph), empty when a tool carried none."""
    return [(s.name, s.attributes.get("args", {})) for s in spans_of_kind(spans, "tool")]


def guard_outcomes(spans) -> list[tuple[str, bool]]:
    """(guard_name, ok) per guard span, in order. ok defaults False when a span is unannotated."""
    return [(s.name, bool(s.attributes.get("ok"))) for s in spans_of_kind(spans, "guard")]


def write_applied(spans) -> bool:
    """Did a write land? True iff some ``execute_action`` node span carries a truthy ``applied``.
    The one definition, structural and never from prose, shared by the eval graders and the drift
    lane so a renamed span or a changed rule moves here, not in two places that can silently diverge."""
    return any(s.name == "execute_action" and s.attributes.get("applied") for s in spans_of_kind(spans, "node"))


def retrieved_doc_ids(spans) -> tuple[str, ...]:
    """Every chunk id the knowledge tool returned this turn, de-duplicated, first-seen order.

    Decoded from the knowledge tool span's own `result` payload, so it is exactly what retrieval
    handed the model. Two payload shapes, both handled: the happy path is a bare passages array, and
    the degradation ladder wraps the same array in `{atlas_degraded, degradation_mode, passages}`.
    A malformed or absent payload yields nothing rather than raising: a decoder over a recorded trace
    must never be the thing that fails a run.
    """
    seen: dict[str, None] = {}
    for span in spans_of_kind(spans, "tool"):
        if span.name != "search_knowledge":
            continue
        raw = span.attributes.get("result")
        if not isinstance(raw, str):
            continue
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            continue
        passages = payload.get("passages", ()) if isinstance(payload, dict) else payload
        if not isinstance(passages, list):
            continue
        for passage in passages:
            if isinstance(passage, dict) and passage.get("chunk_id"):
                seen.setdefault(str(passage["chunk_id"]), None)
    return tuple(seen)


class Tracer(Protocol):
    """The trace port. Adapters: ``InMemoryTracer`` (CI), an OTel/Langfuse handler (dev/prod).

    ``close(seq)`` (SP6 task 2): a minimal protocol extension for the five ``atlas.stage.*ms``
    durations. A no op everywhere except the real OTel adapter: ``atlas_graph.py``'s existing ~20
    ``open()`` call sites (turn/agent/guard/tool/node, owned by SP3/SP4) never call it and stay
    untouched; only the NEW embed/retrieve/rerank/assemble stage spans (and ``chat_app.py``'s ttft
    mark) pair an ``open()`` with a matching ``close(seq)``. ``InMemoryTracer`` records nothing on
    close (the hermetic lane has no wall clock to measure a duration with; the CI adapter's own
    assertion helpers read span PRESENCE and attributes, never a computed ms value), so every
    existing helper (``spans_of_kind``, ``tool_names``, ``guard_outcomes``, ...) stays byte
    compatible. Only the OTel adapter turns an open/close pair into a real monotonic duration.

    ``mark()`` / ``open(..., start_at=...)`` (SP6 task 2 fix round 2): a second, narrower protocol
    extension, for exactly one caller today (``chat_app.py``'s ttft measurement). ``mark()`` returns
    an opaque "now" reading with no span attached at all; a caller that knows its true measurement
    start but not yet the real parent to nest a span under can read one immediately, then hand it
    back to ``open(..., start_at=mark)`` once the real parent IS known, LATER. The span is created at
    that later point (so it can nest correctly, never an unparented, independently random OTel
    trace), but its reported duration is anchored to the earlier mark, not to whenever ``open()``
    itself happens to run. A no op everywhere except the OTel adapter: ``mark()`` returns ``None`` on
    both ``NullTracer`` and ``InMemoryTracer`` (neither has a wall clock story to anchor one to), and
    ``start_at=None`` (the default, every existing call site) leaves every other ``open()`` caller's
    behavior byte identical to before this addition."""

    def open(self, name: str, kind: str, parent: Optional[int] = None, *, start_at: Optional[float] = None, **attrs) -> int: ...
    def annotate(self, seq: int, **attrs) -> None: ...
    def close(self, seq: int) -> None: ...
    def mark(self) -> Optional[float]: ...


class NullTracer:
    """No op tracer. Runtime code is instrumented unconditionally, and observation is opt in."""

    def open(self, name: str, kind: str, parent: Optional[int] = None, *, start_at: Optional[float] = None, **attrs) -> int:
        return -1

    def annotate(self, seq: int, **attrs) -> None:
        return None

    def close(self, seq: int) -> None:
        return None

    def mark(self) -> Optional[float]:
        return None


class InMemoryTracer:
    """The CI adapter: an assertable, deterministic span tree ordered by ``SpanSequence``."""

    def __init__(self, spans: Optional[SpanSequence] = None) -> None:
        self._seq = spans or SpanSequence()
        self._spans: dict[int, Span] = {}
        self._order: list[int] = []

    def open(self, name: str, kind: str, parent: Optional[int] = None, *, start_at: Optional[float] = None, **attrs) -> int:
        # `start_at` is OTel adapter plumbing (fix round 2): this hermetic adapter has no wall clock
        # story to backdate against, so it is accepted (protocol compatibility) and silently ignored,
        # never folded into `attrs` (it is not a domain observable attribute).
        seq = self._seq.next()
        self._spans[seq] = Span(seq, name, kind, parent, MappingProxyType(dict(attrs)))
        self._order.append(seq)
        return seq

    def annotate(self, seq: int, **attrs) -> None:
        cur = self._spans.get(seq)
        if cur is not None:  # immutable update: replace, never mutate in place
            merged = MappingProxyType({**cur.attributes, **attrs})
            self._spans[seq] = Span(cur.seq, cur.name, cur.kind, cur.parent, merged)

    def close(self, seq: int) -> None:
        """Records nothing (SP6 task 2): the hermetic lane has no wall clock, so there is no
        duration to attach. The span opened by ``open()`` already exists and is already assertable;
        this exists only so every ``atlas_graph.py``/``chat_app.py`` call site can call
        ``tracer.close(seq)`` unconditionally, whichever adapter is wired in."""
        return None

    def mark(self) -> Optional[float]:
        """``None`` (SP6 task 2 fix round 2): the SAME "no wall clock in this hermetic lane" reason
        ``close`` already documents. Safe for ``chat_app.py`` to call unconditionally regardless of
        which adapter is wired in; the returned ``None`` flows straight into ``open(start_at=None)``,
        this adapter's own default and therefore a genuine no op end to end."""
        return None

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
    "spans_of_kind", "tool_names", "tool_calls", "guard_outcomes", "write_applied",
    "retrieved_doc_ids",
]
