"""Spotlighting: the prompt-assembly control that marks retrieved content as DATA, not instructions.

Microsoft's spotlighting combines delimiting (wrap the passage in unambiguous boundaries) with
datamarking (interleave a marker token between words so an injected instruction loses the contiguous
shape a model reads as a command). Pure string transforms, so the CONTROL is unit-testable here even
though its EFFECTIVENESS on a live model is only observable in the operator lane (~95%+ on current
models, never 100%).

Deliberately NOT wired into the runtime knowledge serializer yet: doing so changes the bytes of the
tool result the model sees, which would invalidate the recorded cassettes the replay lane depends on.
Wiring it (and re-recording) is a scoped follow-up; this module defines and tests the transform.
"""
from __future__ import annotations

DATAMARK = "ˆ"  # a combining-free marker interleaved between tokens (Microsoft datamarking)
_BEGIN = "<<BEGIN UNTRUSTED DATA doc={doc_id}>>"
_END = "<<END UNTRUSTED DATA>>"


def datamark(text: str, marker: str = DATAMARK) -> str:
    """Interleave ``marker`` between whitespace-separated tokens. Token-reversible via ``undatamark``:
    the token sequence round-trips, but runs of whitespace normalize to single spaces (``str.split``
    collapses them and drops leading/trailing), so this is not a byte-exact codec. Marking the tokens,
    not preserving whitespace, is the control's job."""
    return marker.join(text.split())


def undatamark(marked: str, marker: str = DATAMARK) -> str:
    """Recover the token stream. Datamarking is token-reversible, not byte-lossless: ``datamark``
    already normalized whitespace runs to single spaces, so multi-space/tab/newline text does not
    round-trip to its exact original (single-spaced text does)."""
    return marked.replace(marker, " ")


def spotlight(text: str, doc_id: str) -> str:
    """Wrap a retrieved passage in the delimited, datamarked envelope that tells the model the content
    is untrusted data to be summarised or quoted, never instructions to follow."""
    return f"{_BEGIN.format(doc_id=doc_id)}\n{datamark(text)}\n{_END}"
