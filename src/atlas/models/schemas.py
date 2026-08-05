"""The two reply shapes both providers are constrained to return. Data only, no client
or network access, so `ProviderReply` and `evals.judge.Verdict` are real types rather
than hopes about what a model returned."""

from __future__ import annotations

from typing import Any

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "The answer shown to the customer."},
        "cited": {
            "type": "array", "items": {"type": "string"},
            "description": (
                "Chunk ids only. Each passage in the prompt is introduced by its id in "
                "square brackets; write the id exactly as it appears there, without the "
                "brackets themselves."
            ),
        },
        "outcome": {
            "type": "string",
            "enum": ["answered", "refused", "unknown", "not_applicable"],
        },
        "reason": {"type": "string", "description": "Why, when refusing. Empty otherwise."},
    },
    "required": ["text", "cited", "outcome", "reason"],
    "additionalProperties": False,
}

# Separate from ANSWER_SCHEMA, not a superset: a judge able to return `cited` would be
# inventing the evidence it's grading.
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "reason": {
            "type": "string",
            "description": "Which rule in the scoring guide decided this verdict.",
        },
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}
