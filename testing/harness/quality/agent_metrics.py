"""Deterministic generation side metrics (SP7 Task 3): tool call selection precision/recall and
argument exact match against `expected_tool_calls`, citation precision/recall via entity_id
overlap, an intent confusion matrix (observed classified intent vs `case.intent`), missed and
false refusal rates (`answerable` vs the observed refusal outcome), and reference based answer
correctness dereferencing `expected_facts` against tool results and response text. Also exposes
the counterfactual equivalence check (matched `expected_facts` set plus `refusal_class` equality)
SP7 Task 5's fairness pairs and multi turn runner reuse.

NO reference free anything lives here. Every metric dereferences a case's own declared ground
truth (`expected_tool_calls`, `expected_facts`, `case.intent`, `answerable`, `refusal_class`);
none infers correctness from the response alone. A reference free faithfulness proxy, a judge, or
a rubric belongs entirely to SP8's one calibrated judge (D15), the 04/05 grader boundary this
repo's CLAUDE.md names. This module never reads or emits `atlas.judge.*`.

Where a metric's denominator is empty (no expected tool calls, no citations, no expected facts, no
cases in an answerable/unanswerable bucket) it returns 0.0, never a `ZeroDivisionError`. That is
the SAME guarded convention `quality.ir_metrics.recall_at_k`/`average_precision_at_k` use for an
empty relevant set, applied uniformly here rather than invented per function. A caller that wants
to average one of these over a golden set should curate its input the same way a retrieval golden
slice always labels at least one relevant chunk: skip a case whose contract entry omits the field
this metric reads, rather than pass an empty list through and read the 0.0 guard as a real failure.

`is_fact_grounded`/`answer_correctness_rate` dereference an `expected_facts` value against
response text (substring containment: a rendered value is a literal token or clause in prose) or a
tool result (exact equality over every flattened leaf), both sides cast through `str()` first, the
SAME typed coercion `corpus_tools.verify._registry_value` applies (`str(reg.entity(entity_id
).fields[field])`) so an int or a bool value compares the same way here as it does at corpus build
time, never a second, silently different stringification rule. No span or clause level grading is
attempted (D14 does not ask for it); a fact whose only rendered form is a prose branch not
literally containing the value (`contract_months: 0` rendering as "No contract. Cancel any time.",
the same branch `corpus_tools.render`/`verify` document) is an accepted false negative here, the
same way it is not something `quality.ir_metrics` retrieval grading resolves either.

Pure stdlib, no wall clock, no randomness, no unordered iteration: the same hermetic discipline as
`quality.stats` and `quality.ir_metrics`, so this module runs under the same `task test`.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import NamedTuple

__all__ = [
    "CitationMetrics",
    "ConfusionMatrix",
    "RefusalRates",
    "ToolCallMetrics",
    "answer_correctness_rate",
    "build_confusion_matrix",
    "citation_precision_recall",
    "confusion_matrix_accuracy",
    "counterfactual_equivalent",
    "expected_entity_ids",
    "is_fact_grounded",
    "refusal_rates",
    "tool_call_metrics",
]


# ---- tool call selection precision/recall + argument exact match ----


class ToolCallMetrics(NamedTuple):
    """Three distinct failure modes, kept separate rather than folded into one score: `precision`
    (did the agent call tools it should not have), `recall` (did it call every tool it needed),
    `argument_exact_match` (of the expected calls, how many got the tool AND its args exactly
    right, not merely the right tool name)."""

    precision: float
    recall: float
    argument_exact_match: float


def _tool_name(call: Mapping[str, object]) -> str:
    return str(call["tool"])


def _canonical_args(call: Mapping[str, object]) -> str:
    # sort_keys so {"a": 1, "b": 2} and {"b": 2, "a": 1} compare equal; a missing "args" key and an
    # explicit empty dict both canonicalize to "{}", the schema's own optional-field convention.
    return json.dumps(call.get("args") or {}, sort_keys=True, separators=(",", ":"))


def tool_call_metrics(
    expected: Sequence[Mapping[str, object]], observed: Sequence[Mapping[str, object]]
) -> ToolCallMetrics:
    """Multiset match on tool name for selection precision/recall (a tool called twice when once
    was expected only credits one hit); multiset match on (tool, canonical args) for the exact
    match rate. Every ratio is guarded per the module docstring's empty denominator convention."""
    expected_names = Counter(_tool_name(c) for c in expected)
    observed_names = Counter(_tool_name(c) for c in observed)
    name_hits = sum(min(count, observed_names[name]) for name, count in expected_names.items())
    precision = name_hits / len(observed) if observed else 0.0
    recall = name_hits / len(expected) if expected else 0.0

    expected_full = Counter((_tool_name(c), _canonical_args(c)) for c in expected)
    observed_full = Counter((_tool_name(c), _canonical_args(c)) for c in observed)
    full_hits = sum(min(count, observed_full[key]) for key, count in expected_full.items())
    argument_exact_match = full_hits / len(expected) if expected else 0.0

    return ToolCallMetrics(precision=precision, recall=recall, argument_exact_match=argument_exact_match)


# ---- citation precision/recall via entity_id overlap ----


class CitationMetrics(NamedTuple):
    precision: float
    recall: float


def citation_precision_recall(
    cited_entity_ids: Sequence[str], expected_entity_ids_: frozenset[str] | set[str]
) -> CitationMetrics:
    """Set based overlap between the entity_ids the response actually cited and the registry
    ground truth entity_ids for the case. No `@k` ranking notion: a response either cites the
    grounding entity somewhere or it does not, unlike a ranked retrieval list."""
    cited = set(cited_entity_ids)
    expected = set(expected_entity_ids_)
    hits = len(cited & expected)
    precision = hits / len(cited) if cited else 0.0
    recall = hits / len(expected) if expected else 0.0
    return CitationMetrics(precision=precision, recall=recall)


def expected_entity_ids(expected_facts: Sequence[Mapping[str, object]]) -> frozenset[str]:
    """The registry ground truth entity_id set for a case's `expected_facts`: the entity_id half
    of each fact's `entity_id:field` fact_id, the same `fact_ref` convention `corpus_tools.
    registry`/`corpus_tools.verify`/`dataset_tools.generator` all share (mirrored here, never a
    third variant). A fact_id with no colon (a hand curated case predating this convention, e.g.
    the committed `contracts/dataset/examples` fixtures) degrades to the whole fact_id string
    rather than raising: a citation grader for that case simply cannot resolve a real entity id
    out of an opaque label."""
    return frozenset(str(fact["fact_id"]).partition(":")[0] for fact in expected_facts)


# ---- intent confusion matrix ----


class ConfusionMatrix(NamedTuple):
    """`labels` is every distinct intent seen on either side, sorted for determinism. `counts`
    maps `(expected_intent, observed_intent) -> count`; the diagonal (`expected == observed`) is
    the correctly classified mass `confusion_matrix_accuracy` sums."""

    labels: tuple[str, ...]
    counts: Mapping[tuple[str, str], int]


def build_confusion_matrix(pairs: Sequence[tuple[str, str]]) -> ConfusionMatrix:
    """`pairs` is `(case.intent, observed_intent)` per case, walked in the given order (no set or
    dict reordering of the pairs themselves; only the final label list is sorted, for a
    reproducible, not insertion order dependent, `labels` tuple)."""
    counts: dict[tuple[str, str], int] = {}
    labels: set[str] = set()
    for expected_intent, observed_intent in pairs:
        key = (expected_intent, observed_intent)
        counts[key] = counts.get(key, 0) + 1
        labels.add(expected_intent)
        labels.add(observed_intent)
    return ConfusionMatrix(labels=tuple(sorted(labels)), counts=counts)


def confusion_matrix_accuracy(matrix: ConfusionMatrix) -> float:
    """Fraction of cases where the observed intent equals the case's declared intent (the matrix
    diagonal). Guarded: a matrix built from zero pairs returns 0.0, not a ZeroDivisionError."""
    total = sum(matrix.counts.values())
    if total == 0:
        return 0.0
    correct = sum(
        count
        for (expected_intent, observed_intent), count in matrix.counts.items()
        if expected_intent == observed_intent
    )
    return correct / total


# ---- missed / false refusal rates ----


class RefusalRates(NamedTuple):
    missed_refusal_rate: float
    false_refusal_rate: float


def refusal_rates(outcomes: Sequence[tuple[bool, bool]]) -> RefusalRates:
    """`outcomes` is `(case.answerable, observed_refused)` per case. A MISSED refusal is the
    hallucination shaped failure: the case is unanswerable but the agent answered anyway (did not
    refuse). A FALSE refusal is the over cautious failure: the case is answerable but the agent
    refused anyway. Each rate is computed over its own bucket (unanswerable cases for missed,
    answerable cases for false); an empty bucket is guarded to 0.0."""
    unanswerable_refusals = [refused for answerable, refused in outcomes if not answerable]
    answerable_refusals = [refused for answerable, refused in outcomes if answerable]
    missed = sum(1 for refused in unanswerable_refusals if not refused)
    false_refusals = sum(1 for refused in answerable_refusals if refused)
    missed_rate = missed / len(unanswerable_refusals) if unanswerable_refusals else 0.0
    false_rate = false_refusals / len(answerable_refusals) if answerable_refusals else 0.0
    return RefusalRates(missed_refusal_rate=missed_rate, false_refusal_rate=false_rate)


# ---- reference based answer correctness ----


def _flatten_values(obj: object):
    if isinstance(obj, Mapping):
        for value in obj.values():
            yield from _flatten_values(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _flatten_values(item)
    else:
        yield obj


def is_fact_grounded(
    fact: Mapping[str, object], response_text: str, tool_results: Sequence[object] = ()
) -> bool:
    """One `expected_facts` entry (`{"fact_id": ..., "value": ...}`) dereferenced against the two
    places a correct answer's value can legitimately come from: the response text itself
    (substring containment) or a tool result the agent actually received (exact equality over
    every flattened leaf value). See the module docstring for the shared `str()` typed coercion
    and the known prose branch false negative limitation."""
    expected = str(fact["value"])
    if expected in response_text:
        return True
    return any(expected == str(leaf) for leaf in _flatten_values(tool_results))


def answer_correctness_rate(
    expected_facts: Sequence[Mapping[str, object]],
    response_text: str,
    tool_results: Sequence[object] = (),
) -> float:
    """Fraction of a case's `expected_facts` successfully dereferenced. Guarded: no expected facts
    returns 0.0 (nothing to check is a defined value, not a claimed perfect score)."""
    if not expected_facts:
        return 0.0
    grounded = sum(1 for fact in expected_facts if is_fact_grounded(fact, response_text, tool_results))
    return grounded / len(expected_facts)


# ---- counterfactual equivalence (D33, reused by SP7 Task 5's fairness pairs) ----


def _fact_value_set(case: Mapping[str, object]) -> frozenset[tuple[str, str]]:
    facts = case.get("expected_facts") or ()
    return frozenset((str(fact["fact_id"]), str(fact["value"])) for fact in facts)


def counterfactual_equivalent(case_a: Mapping[str, object], case_b: Mapping[str, object]) -> bool:
    """D33's exact equivalence check: two persona paired cases are equivalent when their
    `expected_facts` sets match exactly (fact_id and value, order independent) and their
    `refusal_class` is equal. Registry anchored and deterministic, no embedding or fuzzy measure;
    persona/style/region are the only fields an equivalent pair is allowed to differ on, and
    neither is read here. Symmetric by construction (set equality and `==` are both symmetric),
    the property SP7 Task 5's pair generator tests against."""
    return _fact_value_set(case_a) == _fact_value_set(case_b) and case_a.get("refusal_class") == case_b.get(
        "refusal_class"
    )
