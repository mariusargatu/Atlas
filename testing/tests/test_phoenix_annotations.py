"""`atlas.adapters.phoenix_annotations` (SP8 Task 4 remainder, D30): the Phoenix annotation mirror.
Hermetic by construction -- `NullPhoenixAnnotationClient` is the only client this suite ever
constructs, so no test here makes a network call. `mirror_label`'s translation (verdict -> label/
score) and its one call to whatever client it is given are exercised against a stub that records
calls, never a real Phoenix client (there is none in this repo; wiring one is an operator/live
concern, documented, not built by this task).
"""
from __future__ import annotations

from atlas.adapters.phoenix_annotations import (
    NullPhoenixAnnotationClient,
    mirror_label,
)


class _RecordingClient:
    """A stub `PhoenixAnnotationClient`: records every call instead of doing any I/O."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def annotate(self, *, trace_id: str, label: str, score: float, explanation: str) -> None:
        self.calls.append(
            {"trace_id": trace_id, "label": label, "score": score, "explanation": explanation}
        )


def test_null_client_annotate_is_a_no_op_and_returns_none():
    client = NullPhoenixAnnotationClient()
    result = client.annotate(trace_id="t1", label="pass", score=1.0, explanation="Grounded.")
    assert result is None


def test_mirror_label_never_raises_against_the_null_client():
    mirror_label(NullPhoenixAnnotationClient(), trace_id="t1", verdict="fail", critique="Unsupported claim.")


def test_mirror_label_forwards_trace_id_and_critique_as_explanation():
    client = _RecordingClient()
    mirror_label(client, trace_id="t42", verdict="pass", critique="Matches the cited page.")
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["trace_id"] == "t42"
    assert call["explanation"] == "Matches the cited page."


def test_mirror_label_scores_a_pass_verdict_as_one():
    client = _RecordingClient()
    mirror_label(client, trace_id="t1", verdict="pass", critique="Grounded.")
    assert client.calls[0]["label"] == "pass"
    assert client.calls[0]["score"] == 1.0


def test_mirror_label_scores_a_fail_verdict_as_zero():
    client = _RecordingClient()
    mirror_label(client, trace_id="t1", verdict="fail", critique="Unsupported claim.")
    assert client.calls[0]["label"] == "fail"
    assert client.calls[0]["score"] == 0.0


def test_mirror_label_calls_annotate_exactly_once_per_label():
    client = _RecordingClient()
    mirror_label(client, trace_id="t1", verdict="pass", critique="First.")
    mirror_label(client, trace_id="t2", verdict="fail", critique="Second.")
    assert len(client.calls) == 2
