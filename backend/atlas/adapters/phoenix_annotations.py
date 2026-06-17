"""Phoenix annotation mirror (SP8 Task 4 remainder, D30): label + score + explanation attached to
an already exported span via Phoenix's own annotation API. `atlas.adapters.label_store.LabelStore`
(the append only JSONL) is the system of record; this module is a VIEW ONLY mirror, matching D30's
own wording verbatim ("Phoenix is a view"). If a live Phoenix client were ever unreachable, the
label itself is unaffected: `backend/atlas/label_routes.py`'s own `post_label` calls `mirror_label`
only AFTER `LabelStore.append` has already durably written the line.

The span_id gap `atlas.adapters.label_store`'s own docstring names (the chat response envelope
carries only a turn level `trace_id`, never a real `span_id`) is resolved here the SAME way:
`mirror_label` keys its annotation by `trace_id`, the identity that actually exists.

Hermetic by construction: `PhoenixAnnotationClient` is a small `Protocol`, and
`NullPhoenixAnnotationClient` (the default `build_label_router` constructs) is a documented no op,
so `task test` never makes a live Phoenix call -- there is no HTTP client anywhere in this module,
only the seam a real one plugs into. Wiring an actual Phoenix client (an HTTP call to Phoenix's own
annotation REST endpoint) is an operator/live concern, not built by this task, mirroring `judge/
live_provisional.py`'s own "the mechanism lands here, live wiring is a separate, documented step"
precedent.
"""
from __future__ import annotations

from typing import Protocol

# `atlas.adapters.label_store.LabelRecord`'s own verdict vocabulary ("pass"/"fail", a HUMAN
# adjudication label, never the judge's wire vocabulary "grounded"/"ungrounded") -- this module
# mirrors labels, not judge verdicts, so it translates from THIS vocabulary, never `judge.llm_judge`'s.
_VERDICT_SCORE: dict[str, float] = {"pass": 1.0, "fail": 0.0}


class PhoenixAnnotationClient(Protocol):
    """The one seam a live Phoenix client fills. `annotate` mirrors Phoenix's own span annotation
    shape (label/score/explanation), keyed by `trace_id` (the span_id gap resolution above)."""

    def annotate(self, *, trace_id: str, label: str, score: float, explanation: str) -> None: ...


class NullPhoenixAnnotationClient:
    """The hermetic default: no live Phoenix endpoint is configured, so mirroring is a documented
    no op, never a silently swallowed failure mistaken for success (`metrics.py`'s own
    `_corpus_staleness` "absence is not a fault" discipline, applied to a client instead of a
    gauge)."""

    def annotate(self, *, trace_id: str, label: str, score: float, explanation: str) -> None:
        return None


def mirror_label(client: PhoenixAnnotationClient, *, trace_id: str, verdict: str, critique: str) -> None:
    """Thin call: translates one stored label's own fields (`verdict`/`critique`) into Phoenix's
    label/score/explanation triple and hands off to `client`. This function does no I/O of its own
    -- `client` (a live Phoenix client, or `NullPhoenixAnnotationClient`) is the only thing that ever
    performs one. An unrecognized `verdict` scores `0.0` rather than raising: by the time this
    function runs, `LabelStore.append` has already validated `verdict` against its own known set, so
    this mirror is never the place that first rejects a malformed label."""
    client.annotate(
        trace_id=trace_id,
        label=verdict,
        score=_VERDICT_SCORE.get(verdict, 0.0),
        explanation=critique,
    )


__all__ = ["NullPhoenixAnnotationClient", "PhoenixAnnotationClient", "mirror_label"]
