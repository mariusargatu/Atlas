"""The graded gold set: which required documents actually carry the answer.

tau2 names the documents an agent must consult and does not say which one holds the
answer. For 22 of the 97 tasks it is derived from the arguments of the tool calls a
correct agent would make, rather than annotated.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from atlas.contracts import Chunk, ChunkId, DocName, Question, QuestionName, Range
from atlas.corpus.gold import resolve
from atlas.corpus.tau2 import Tau2Source
from evals.ir_metrics import graded_ndcg_at_k, ndcg_at_k


def _chunk(name: str, doc: str, text: str = "x" * 40) -> Chunk:
    return Chunk(name=ChunkId(name), document=DocName(doc), ordinal=0,
                    span=Range(0, len(text)), text=text)


@pytest.fixture
def graded_question():
    """A question requiring three documents, one of which carries the answer."""
    return Question(
        name=QuestionName("q_graded"), text="which card pays the most",
        kind="across_documents",
        required=(DocName("doc.gold"), DocName("doc.silver"), DocName("doc.bronze")),
        primary=(DocName("doc.gold"),),
    )


@pytest.fixture
def graded_setup(graded_question):
    chunks = tuple(_chunk(f"{d}#0000", d) for d in ("doc.gold", "doc.silver", "doc.bronze"))
    return graded_question, chunks


def test_primary_chunks_are_a_subset_of_the_correct_ones(graded_setup) -> None:
    # A primary chunk outside the correct set would be a gain a ranking could earn
    # for retrieving something the gold set says is wrong.
    question, chunks = graded_setup
    correct = resolve(question, chunks)
    assert correct.primary <= correct.chunks
    assert correct.primary == frozenset({ChunkId("doc.gold#0000")})


def test_ranking_the_answer_first_beats_ranking_it_last(graded_setup) -> None:
    # Both orderings retrieve all three required documents, so every flat measurement
    # scores them identically; only the graded one can tell them apart.
    question, chunks = graded_setup
    correct = resolve(question, chunks)
    answer_first = [ChunkId("doc.gold#0000"), ChunkId("doc.silver#0000"),
                    ChunkId("doc.bronze#0000")]
    answer_last = list(reversed(answer_first))

    assert ndcg_at_k(answer_first, correct.chunks, 3) == pytest.approx(
        ndcg_at_k(answer_last, correct.chunks, 3)
    ), "the ungraded metric is supposed to be blind to this, or the grade buys nothing"
    assert graded_ndcg_at_k(answer_first, correct, 3) > graded_ndcg_at_k(
        answer_last, correct, 3
    )


def test_a_perfect_graded_ranking_scores_one(graded_setup) -> None:
    question, chunks = graded_setup
    correct = resolve(question, chunks)
    perfect = [ChunkId("doc.gold#0000"), ChunkId("doc.silver#0000"),
               ChunkId("doc.bronze#0000")]
    assert graded_ndcg_at_k(perfect, correct, 3) == pytest.approx(1.0)


def test_a_question_with_no_derivable_primary_scores_not_a_number(graded_setup) -> None:
    # Never a silent fallback to the flat metric: three quarters of the tasks have no
    # derivable primary document, so a graded figure would be mostly ungraded ones.
    question, chunks = graded_setup
    ungraded = replace(question, primary=())
    correct = resolve(ungraded, chunks)
    assert correct.primary == frozenset()
    assert math.isnan(graded_ndcg_at_k([f.name for f in chunks], correct, 3))


def test_the_derivation_covers_the_tasks_it_claims_and_stays_inside_the_gold_set() -> None:
    # The count is pinned because a matching change that stopped matching would leave the
    # graded slice empty, turning every graded number into not a number rather than red.
    source = Tau2Source()
    questions = source.questions()
    graded = [q for q in questions if q.primary]
    assert len(graded) == 22, f"{len(graded)} tasks derive a primary document, expected 22"
    for question in graded:
        assert set(question.primary) <= set(question.required)
        assert len(question.primary) < len(question.required) or len(question.required) == 1, (
            f"{question.name} marks every required document primary, which grades nothing"
        )
