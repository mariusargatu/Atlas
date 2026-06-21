"""Turn a flagged production failure into a golden case, PII-scrubbed first.

`scrub` redacts structured PII from the turns and remaps the customer to a seeded synthetic
account; `promote` builds a silver GoldenCase (source="production") for a human to ratify to gold.
`promote` refuses a session that still carries structured PII or a supplied name.

Pure and deterministic: this core gates the hermetic lane. Capture and triage that feed it (trace
collection, model-assisted clustering, human confirmation) are the operator/infra lane.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

from evals.evalkit.golden_case import (
    AuthorRole,
    Category,
    Consequence,
    GoldenCase,
    GoldenDraft,
    Tier,
    enrich,
)

# Ordered so the greediest patterns run last: a 16-digit card must redact as a card, not an account;
# a phone must redact as a phone, not an account; a grouped account run must be tried before the bare
# digit run. Names are handled separately via a caller-supplied list, because deterministic name
# detection without one is unreliable.
#
# Covered: email, card number, UK postcode, UK phone (contiguous OR single space/hyphen grouped, e.g.
# "07911 123456", "+44 7911 123456"), and account numbers (any bare 6+ digit run, OR a space/hyphen
# grouped run of 8+ total digits, e.g. "4002 1785", "12-34-56 78901234"). Not covered: free-text
# addresses, names not in the supplied list, and non-UK phone/postcode formats. This is
# structured-identifier redaction, not DLP.
#
# Over-redaction is the SAFE direction here (a spuriously redacted number is harmless; a leaked
# identifier is not), so the grouped-account pattern is deliberately broad. The ONE shape carved back
# out is a bare ISO date (\d{4}-\d{2}-\d{2}): those appear constantly in support notes and provenance,
# and redacting "2026-07-17" as an account would corrupt legitimate metadata. The carve-out is exact:
# an ISO date followed by more digits/hyphens is not a date and still redacts.
_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL]"),
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[CARD]"),
    (re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}\b", re.IGNORECASE), "[POSTCODE]"),
    (re.compile(r"(?<![\w+])\+?(?:44|0)(?:[ -]?\d){9,11}\b"), "[PHONE]"),
    # grouped 8+ digit run (single space/hyphen separators), minus the exact ISO-date shape
    (re.compile(r"(?<![\w-])(?!\d{4}-\d{2}-\d{2}(?![\d-]))(?:\d[ -]?){7,}\d"), "[ACCOUNT]"),
    (re.compile(r"\b\d{6,}\b"), "[ACCOUNT]"),
)


def _name_pattern(name: str) -> re.Pattern[str]:
    """Compile a word-boundary, case-insensitive matcher for a caller-supplied name.

    Raises loudly on an empty or whitespace-only name: `re.escape("")` yields `\\b\\b`, which matches
    at EVERY position, so an empty name would redact (scrub_text) or flag as PII (contains_pii) the
    whole text. That is a caller bug, not a redaction, and must fail rather than silently corrupt.
    One home for the `\\b{name}\\b` construction the scrub and the gate both need.
    """
    if not name.strip():
        raise ValueError("names must not contain empty or whitespace-only entries")
    return re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)


@dataclass(frozen=True)
class ProdSession:
    """A sampled, triaged production conversation, before it is safe to commit.

    `customer_id` is the real production id until `scrub` remaps it to a seeded synthetic account.
    `expected` is what a human triage decided correct would have been. `captured_at`, `model_id`,
    and `trace_ref` are optional provenance strings that `promote` folds into the golden case's
    notes when present, so a promoted case can still point back to the incident that created it.
    """

    id: str
    turns: tuple[str, ...]
    customer_id: str
    expected: str
    category: Category
    risk: str
    consequence: Consequence
    oracle: str
    captured_at: str = ""
    model_id: str = ""
    trace_ref: str = ""
    notes: str = ""


def scrub_text(text: str, *, names: Iterable[str] = ()) -> str:
    """Redact structured PII to typed placeholders, then any caller-supplied names. Deterministic."""
    name_patterns = [_name_pattern(name) for name in names]  # validate every name up front (empty raises)
    for pattern, placeholder in _PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    for pattern in name_patterns:
        text = pattern.sub("[NAME]", text)
    return text


def contains_pii(text: str, *, names: Iterable[str] = ()) -> bool:
    """True if any structured PII pattern (or a supplied name) is present. The promote-time guard."""
    name_patterns = [_name_pattern(name) for name in names]  # validate every name up front (empty raises)
    if any(pattern.search(text) for pattern, _ in _PII_PATTERNS):
        return True
    return any(pattern.search(text) for pattern in name_patterns)


def scrub(session: ProdSession, *, as_customer: str, names: Iterable[str] = ()) -> ProdSession:
    """Redact PII from the turns, expected, notes, and provenance, and synthesise the identity as the
    seeded `as_customer`.

    Identity is synthesised, not merely stripped, because a GoldenCase must name a seeded account.
    `as_customer` should be the seeded account whose shape matches the failure (a legacy-term customer
    for a legacy-term failure), so the promoted case still exercises the same oracle.

    `notes` is part of the session and is scrubbed like any turn: a triager pasting a customer's phone
    number into a note must not smuggle it past the redactor. The provenance fields
    (`captured_at`/`model_id`/`trace_ref`) are engineer-supplied metadata, but `promote` folds them
    into the case notes AFTER its refusal gate, so they are scrubbed here too, before they can be
    folded. This is safe over-redaction: a trace ref that happens to carry a long digit run may redact,
    which is acceptable; an ISO-date `captured_at` is carved out and survives (see `_PII_PATTERNS`).
    """
    names = tuple(names)
    return replace(
        session,
        turns=tuple(scrub_text(turn, names=names) for turn in session.turns),
        expected=scrub_text(session.expected, names=names),
        notes=scrub_text(session.notes, names=names),
        captured_at=scrub_text(session.captured_at, names=names),
        model_id=scrub_text(session.model_id, names=names),
        trace_ref=scrub_text(session.trace_ref, names=names),
        customer_id=as_customer,
    )


def promote(
    session: ProdSession,
    *,
    names: Iterable[str],
    tier: Tier = "silver",
    author_role: AuthorRole = "engineer",
    graders: tuple[str, ...] = (),
    notes: str | None = None,
) -> GoldenCase:
    """Promote a scrubbed production session to a GoldenCase (silver, source="production").

    `names` is required so the caller attests the scrub list: pass the same names `scrub` used, or
    `()` for a session with genuinely no names to scrub. Refuses a session whose turns, expected, OR
    notes still carry structured PII or one of the supplied names; scrub is the gate, not a suggestion.
    Any non-empty `captured_at`/`model_id`/`trace_ref` on the session is folded into the case's notes,
    so a promoted case still points back to the incident that created it. Provenance is folded AFTER
    this gate, so the gate scans the INCOMING notes, and `scrub` is responsible for redacting the
    provenance fields before they reach here. The result is silver, provisional until a human and the
    evaluator ratify it to gold. GoldenCase validation independently enforces that the customer is a
    seeded account, so a real identity cannot ride in even if the id was never remapped.
    """
    names = tuple(names)  # materialised once: contains_pii below is called per field, not per name
    # the notes that will land in the case are what the gate must scan (an explicit `notes=` override,
    # else the session's own notes), alongside the turns and expected.
    base_notes = session.notes if notes is None else notes
    leaked = [field for field in (*session.turns, session.expected, base_notes) if contains_pii(field, names=names)]
    if leaked:
        raise ValueError(
            f"{session.id}: refusing to promote un-scrubbed PII ({len(leaked)} field(s)); call scrub() first"
        )
    provenance = " ".join(
        f"{key}={value}"
        for key, value in (
            ("captured", session.captured_at),
            ("model", session.model_id),
            ("trace", session.trace_ref),
        )
        if value
    )
    draft = GoldenDraft(
        id=session.id,
        turns=session.turns,
        customer_id=session.customer_id,
        expected=session.expected,
        notes=f"{base_notes} {provenance}".strip() if provenance else base_notes,
    )
    return enrich(
        draft,
        category=session.category,
        risk=session.risk,
        consequence=session.consequence,
        oracle=session.oracle,
        source="production",
        author_role=author_role,
        tier=tier,
        graders=graders,
    )


__all__ = ["ProdSession", "contains_pii", "promote", "scrub", "scrub_text"]
