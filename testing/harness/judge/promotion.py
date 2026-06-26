"""The promotion loop (SP8 Task 5.1, D34): turns a judge's fail spans and end user thumbs down
feedback into taxonomy labeled candidate dataset cases, `origin: promoted`, the seam SP7's own
`dataset_tools.generator.validate_case` already accepts (that function's own docstring: "Accepts
`origin: promoted` exactly like any other origin value: promotion is a later, human gated
activation... never a schema level restriction").

Two failure SOURCES, read independently, never blended:

  - `judge_fail_trace_ids` walks a burst's own span sequence (`tracing.Span`, the same decoder shape
    `tracing.spans_of_kind` already defines) for a `kind="judge"` span whose informal `verdict`
    attribute is the wire fail value (`judge.llm_judge.VERDICT_UNGROUNDED`, "ungrounded";
    `judge.emission.emit_verdict` is the one writer of that attribute). The trace id is the span's
    own `parent`, `str()`-ed: `judge.emission`'s own docstring names the parent as "typically the
    graded turn's own root span", the SAME identity `labeling.generate_label_set.
    generate_label_items` already stamps as `trace_id` (`str(result.get("trace_root"))`) -- so a
    judge fail span's parent and a label item's `trace_id` name the same turn.
  - `thumbs_down_trace_ids` walks a list of `atlas.adapters.label_store.LabelRecord` (the append only
    label JSONL, SP8 Task 4, `fc4d65d`) for `role="end_user"` and `verdict="fail"` (that store's own
    thumbs down vocabulary), keyed by the SAME `trace_id`.

Neither source carries the turn's own content (a judge span has no question/answer text; a
`LabelRecord` has no question/answer text either, only `trace_id`/`role`/`verdict`/`critique`).
`candidates_from_trace_ids` is the join: it resolves each trace id against `items_by_trace_id`, a
mapping in the SAME shape `generate_label_items` already produces
(`{"trace_id", "question", "answer", "retrieved_chunks", "registry_facts"}`, e.g. the HITL page's own
label item set, or a burst's equivalent). A trace id with no matching item is skipped, never
fabricated -- mirrors `generate_label_items`'s own "never fabricates a label item" rule.

`promote` is the taxonomy gate (D34: "promotion requires a taxonomy label"): a candidate with no
`failure_class`, or one not a code `taxonomy.Taxonomy` recognizes, is rejected. A promoted case is
built with `answerable: True` by construction -- every promoted failure originates from a turn the
agent actually attempted to answer (an ungrounded verdict or a thumbs down), never one it correctly
declined, so unanswerable is never the right default here; a future task curating promoted cases
further may revise a specific one by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from dataset_tools.generator import validate_case
from dataset_tools.taxonomy import Taxonomy

from tracing import spans_of_kind

from judge.llm_judge import VERDICT_UNGROUNDED

_END_USER_ROLE = "end_user"
_END_USER_FAIL_VERDICT = "fail"

_ORIGIN_PROMOTED = "promoted"
_DEFAULT_INTENT = "troubleshooting"
_DEFAULT_SPLIT = "dev"


class PromotionError(ValueError):
    """A candidate cannot be promoted: no taxonomy label was given at all. A label naming an UNKNOWN
    code raises `dataset_tools.taxonomy.TaxonomyError` instead (that module's own concern); both are
    `ValueError` subclasses, so a caller catching broadly still sees every rejection."""


@dataclass(frozen=True)
class PromotionCandidate:
    """One turn a failure source flagged, joined with its own question/answer/retrieved content.
    `failure_source` names where the flag came from (`"judge_fail"` / `"end_user_thumbs_down"`,
    though any caller supplied string is accepted -- this module never enumerates the set, the same
    free string convention `contracts/dataset/schema.json`'s own `candidate_source` field already
    uses); a promoted case's own `candidate_source` field records it verbatim."""

    trace_id: str
    question: str
    answer: str
    failure_source: str
    retrieved_chunks: tuple[dict, ...] = ()
    registry_facts: tuple[dict, ...] = ()


def judge_fail_trace_ids(spans) -> list[str]:
    """Trace ids for every judge span whose verdict is the wire fail value (`"ungrounded"`), in span
    order, deduplicated (a re judged turn counts once). A judge span with no parent (never opened
    under a real turn root) is skipped: there is no trace id to key a candidate on."""
    ids: list[str] = []
    for span in spans_of_kind(spans, "judge"):
        if span.attributes.get("verdict") != VERDICT_UNGROUNDED:
            continue
        if span.parent is None:
            continue
        trace_id = str(span.parent)
        if trace_id not in ids:
            ids.append(trace_id)
    return ids


def thumbs_down_trace_ids(records) -> list[str]:
    """Trace ids for every end user thumbs down (`role="end_user"`, `verdict="fail"`) label record,
    in the order given, deduplicated (a customer thumbing the same turn down twice promotes once)."""
    ids: list[str] = []
    for record in records:
        if record.role != _END_USER_ROLE or record.verdict != _END_USER_FAIL_VERDICT:
            continue
        if record.trace_id not in ids:
            ids.append(record.trace_id)
    return ids


def candidates_from_trace_ids(
    trace_ids, items_by_trace_id: dict, *, source: str
) -> tuple[PromotionCandidate, ...]:
    """Joins `trace_ids` (from either failure source above) against `items_by_trace_id`
    (`trace_id -> {"question", "answer", "retrieved_chunks", "registry_facts", ...}`, the same shape
    `labeling.generate_label_set.generate_label_items` already produces). A trace id absent from
    `items_by_trace_id` is silently skipped -- an unresolved candidate is an incomplete one, never a
    partially fabricated one."""
    candidates: list[PromotionCandidate] = []
    for trace_id in trace_ids:
        item = items_by_trace_id.get(trace_id)
        if item is None:
            continue
        candidates.append(
            PromotionCandidate(
                trace_id=trace_id,
                question=item["question"],
                answer=item["answer"],
                failure_source=source,
                retrieved_chunks=tuple(item.get("retrieved_chunks") or ()),
                registry_facts=tuple(item.get("registry_facts") or ()),
            )
        )
    return tuple(candidates)


def promote(
    candidate: PromotionCandidate,
    *,
    failure_class: Optional[str],
    taxonomy: Taxonomy,
    case_id: Optional[str] = None,
) -> dict:
    """Builds and validates one `origin: promoted` dataset case from `candidate`. D34: "promotion
    requires a taxonomy label" -- a missing or empty `failure_class` raises `PromotionError`; a
    non empty one not among `taxonomy`'s known codes raises `dataset_tools.taxonomy.TaxonomyError`
    (`Taxonomy.validate_failure_class`). Only once both checks clear is the case built and run
    through `dataset_tools.generator.validate_case`, the SAME jsonschema validation every other
    dataset case passes -- a promoted case is never exempt from the shared contract."""
    if not failure_class:
        raise PromotionError(
            f"{candidate.trace_id}: promotion requires a taxonomy label, none given"
        )
    taxonomy.validate_failure_class(failure_class)
    case = {
        "case_id": case_id or f"promoted-{candidate.trace_id}",
        "split": _DEFAULT_SPLIT,
        "origin": _ORIGIN_PROMOTED,
        "candidate_source": candidate.failure_source,
        "source_trace_id": candidate.trace_id,
        "intent": _DEFAULT_INTENT,
        "failure_class": failure_class,
        "answerable": True,
        "turns": [{"user": candidate.question}],
    }
    validate_case(case)
    return case


__all__ = [
    "PromotionCandidate",
    "PromotionError",
    "candidates_from_trace_ids",
    "judge_fail_trace_ids",
    "promote",
    "thumbs_down_trace_ids",
]
