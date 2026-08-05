"""What may and may not be uploaded to a dashboard.

Checked against a recording client rather than a server: nothing here opens a socket.
"""

from __future__ import annotations

from typing import Any

from atlas.contracts import DocName, Question, QuestionName
from evals.langfuse_dataset import DATASET_NAME, item_evaluations, sync_items


class RecordingClient:
    """Every call, kept."""

    def __init__(self) -> None:
        self.datasets: list[dict[str, Any]] = []
        self.items: list[dict[str, Any]] = []

    def create_dataset(self, **kwargs: Any) -> None:
        self.datasets.append(kwargs)

    def create_dataset_item(self, **kwargs: Any) -> None:
        self.items.append(kwargs)


def _question(name: str, required: tuple[str, ...], primary: tuple[str, ...] = ()) -> Question:
    return Question(
        name=QuestionName(name), text=f"the text of {name}", kind="across_documents",
        required=tuple(DocName(d) for d in required),
        primary=tuple(DocName(d) for d in primary),
    )


def test_the_uploaded_answer_key_is_documents_and_never_chunks() -> None:
    """An answer key at chunk level is correct for exactly one ChunkSettings, so uploading
    one would publish something authoritative-looking that outlives the settings that
    produced it and cannot be recomputed.
    """
    client = RecordingClient()
    sync_items(client, [_question("task_001", ("doc_a", "doc_b"), ("doc_a",))])

    item = client.items[0]
    assert item["expected_output"] == {"required_documents": ["doc_a", "doc_b"]}
    uploaded = repr(item["expected_output"]) + repr(item["metadata"])
    assert "chunk" not in uploaded.lower(), (
        f"something chunk-shaped reached the dataset's answer key: {uploaded}"
    )


def test_syncing_is_keyed_on_the_question_so_a_second_run_does_not_double_the_dataset() -> None:
    # Without an explicit id the API mints one per call, so every run would append 97
    # more items and the dataset would grow without ever being wrong in a visible way.
    client = RecordingClient()
    questions = [_question("task_001", ("doc_a",)), _question("task_002", ("doc_b",))]
    sync_items(client, questions)
    sync_items(client, questions)

    assert [i["id"] for i in client.items] == ["task_001", "task_002"] * 2
    assert {d["name"] for d in client.datasets} == {DATASET_NAME}


def test_a_metric_that_cannot_speak_for_a_question_is_left_out_rather_than_sent_as_nan() -> None:
    """graded nDCG is not a number for the 75 tasks tau2 names no primary documents for.

    A printed row can drop those questions; a score has no such column, so a NaN posted
    as a number would average into every aggregate downstream while arriving labelled as
    though it meant something.
    """
    output = {
        "question": "task_001",
        "rankings": {"reranked": ["chunk_1", "chunk_2"]},
        "correct": ["chunk_1"],
        "primary": [],  # tau2 names none, so the graded metric cannot be computed
        "generated": False,
    }
    names = {e.name for e in item_evaluations(output, k=10)}
    assert "recall@k" in names and "nDCG@k" in names
    assert "graded nDCG@k" not in names, (
        "a not-a-number was published as a score, which reads as a measurement"
    )

    # Otherwise the assertion above passes for any reason at all, including somebody
    # deleting the graded metric outright.
    named = {e.name for e in item_evaluations({**output, "primary": ["chunk_1"]}, k=10)}
    assert "graded nDCG@k" in named, (
        "the graded metric is missing even when the question names primary documents, so "
        "the check above no longer tells us anything about not-a-number"
    )
