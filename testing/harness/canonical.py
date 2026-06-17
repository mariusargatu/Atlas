"""Canonicalization + the run digest contract (ADR-007 / 03-test-architecture.md).

The cassette key and the run digest are only as deterministic as this module. Rules:
sorted keys, money as a value normalized Decimal (never a float), dates as ISO 8601
strings (from the frozen clock), and an explicit allow list for the model request digest
so unrelated kwargs can never shift the key.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

# The allow list for the model request digest (ADR-007). Anything not here is ignored, so framework
# kwargs (run_manager, callbacks, ...) never perturb the cassette key. This is the SINGLE source of
# truth for what shapes the key: `cassette.build_request` derives the request fields it copies from
# this same tuple, so the two halves of the key contract cannot drift apart.
REQUEST_ALLOW = (
    "model_id",
    "system",
    "messages",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
)

#: Structural request fields the gateway sets explicitly (not forwarded from caller kwargs).
REQUEST_STRUCTURAL = ("model_id", "system", "messages")


def canonical(value: Any) -> Any:
    """Normalize a value into a JSON serializable, canonical form.

    dicts -> sorted keys; Decimal -> exact quantized string; float -> stable repr;
    datetime/date -> ISO 8601. Bool is preserved (and handled before int fall-through).
    """
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, Decimal):
        # Normalize so value equal Decimals of different scale hash identically (Decimal("35.00")
        # and Decimal("35") are the same money). A content addressed key must turn on value, not
        # representation. `:f` keeps plain decimal notation (never scientific) after normalize.
        return f"D:{value.normalize():f}"
    if isinstance(value, float):
        return f"F:{value!r}"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def canonical_json(value: Any) -> str:
    """Deterministic JSON bytes: sorted keys, no whitespace drift, canonical scalars."""
    return json.dumps(
        canonical(value),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def digest(value: Any) -> str:
    """sha256 of the canonical JSON, the cassette key / run digest."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def serialize_tool_result(payload: Any) -> str:
    """Canonical serialization of a tool result payload before it enters `messages`.

    Money is a quantized Decimal, dates are frozen clock ISO strings, keys are sorted,
    so the same logical result always produces the same bytes (and thus the same next
    cassette key). Applied at the MCP adapter boundary.
    """
    return canonical_json(payload)


def request_digest(request: dict[str, Any]) -> str:
    """Hash only the allow listed request fields (ADR-007)."""
    return digest({k: request.get(k) for k in REQUEST_ALLOW})


__all__ = [
    "REQUEST_ALLOW",
    "REQUEST_STRUCTURAL",
    "canonical",
    "canonical_json",
    "digest",
    "request_digest",
    "serialize_tool_result",
]
