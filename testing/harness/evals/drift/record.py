"""The decision record: what a turn DECIDED, separated from what it said.

The drift lane (the fourth gateway reading) re-runs the pinned agent against a new model snapshot
and asks whether behaviour moved. "Behaviour" is not the prose, it is the decisions: the intent the
turn bound, the tools it called and in what order, the guard verdicts it produced, and the terminal
outcome. The shipped text is kept too, but as a SEPARATE digest, so a reworded-but-equivalent answer
(prose drift) never masquerades as a changed decision (behavioural drift). See the drift compare.

Every decision is read from the TRACE — the structural record the runtime emits — never re-derived
or parsed out of the English. Reading prose to recover a decision is the wrong altitude: a benign
answer that happens to say "your reference is on your bill" would look like a write, the exact false
positive this lane exists to suppress.
"""
from __future__ import annotations

from dataclasses import dataclass

from determinism.canonical import digest
from tracing import guard_outcomes, tool_names

from atlas.domain.binding import classify_intent

# The decision fields — the part of a run that must not silently move, and the SINGLE source of truth
# for what counts as a decision. `decision_digest()` here and the drift `compare()` both iterate this
# tuple, so a new field is added in one place and neither the digest nor the field-by-field diff can
# silently miss it. The claim (prose) is deliberately NOT here.
DECISION_KEYS = ("intent", "tools", "guards", "outcome")


@dataclass(frozen=True)
class DecisionRecord:
    intent: str                                  # the intent the runtime BOUND (read from the trace)
    tools: tuple[str, ...]                        # tool names called, in order (the trajectory)
    guards: tuple[tuple[str, bool], ...]          # (guard_name, ok) verdicts, in order
    outcome: str                                 # "answer" | "handoff" | "write-applied"
    claim_digest: str                            # digest of the shipped text: prose, kept apart

    def decision_digest(self) -> str:
        """A content-addressed digest of the DECISIONS only (never the prose). Two runs whose only
        difference is reworded prose share this digest; a changed tool/guard/outcome does not.
        `canonical` normalizes the tuples (even nested) to lists, so the field types need no
        pre-conversion here."""
        return digest({key: getattr(self, key) for key in DECISION_KEYS})


def _intent_from_trace(trace):
    """The intent the runtime actually BOUND for the turn, read from the turn span (atlas_graph
    records it there). None when the trace carries no turn span (e.g. a synthetic record)."""
    for span in trace:
        if span.kind == "turn":
            bound = span.attributes.get("intent")
            if bound is not None:
                return bound
    return None


def _outcome_of(trace) -> str:
    """Recover the TERMINAL outcome from the span tree, never from the prose. The graph records the
    decision structurally: an `execute_action` node span (applied=True -> a write landed; applied=
    False -> the confirmation was refused), and the guard spans (any ok=False -> a fail-closed
    handoff). Reading prose to recover this would let a benign answer that merely says "your
    reference is ..." masquerade as a write — the exact false positive the drift lane must not make.
    """
    executes = [s for s in trace if s.kind == "node" and s.name == "execute_action"]
    if any(bool(s.attributes.get("applied")) for s in executes):
        return "write-applied"
    if executes:  # an execute_action that did not apply == a refused confirmation
        return "handoff"
    if any(not ok for _, ok in guard_outcomes(trace)):
        return "handoff"
    return "answer"


def extract(utterance: str, trace, final_response: str) -> DecisionRecord:
    """Build a `DecisionRecord` from a driven turn: the spans it emitted and what it shipped.

    Every decision is read from the TRACE (the structural record), via the shared `tracing` helpers:
    the bound intent from the turn span, the tool order and guard verdicts from their spans, the
    outcome from the action/guard spans. `utterance` is only a fallback for the intent when the trace
    carries no turn span (a synthetic record). The prose is digested, never compared field-by-field.
    """
    return DecisionRecord(
        intent=_intent_from_trace(trace) or classify_intent(utterance),
        tools=tuple(tool_names(trace)),
        guards=tuple(guard_outcomes(trace)),
        outcome=_outcome_of(trace),
        claim_digest=digest(final_response or ""),
    )


__all__ = ["DECISION_KEYS", "DecisionRecord", "extract"]
