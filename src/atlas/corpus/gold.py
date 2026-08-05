from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import get_args

from atlas.contracts import Chunk, ChunkId, DocName, Question, QuestionKind, QuestionName

# Derived from the Literal rather than restated, so a kind missing here can't silently
# skip the empty-correct-set guard below.
ANSWERABLE_KINDS = frozenset(get_args(QuestionKind))

_EMPTY = frozenset[ChunkId]()


class EmptyCorrectSetError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Correct:
    """What should have been retrieved, recomputed against whatever chunk set is under
    test. Never stored: a chunk name is correct only for one way of cutting, so a saved
    mapping would be quietly wrong for any other configuration."""

    question: QuestionName
    chunks: frozenset[ChunkId]
    # Subset of `chunks` that carries the answer, as opposed to chunks merely consulted
    # on the way to it. Empty when the source can't tell which is which.
    primary: frozenset[ChunkId] = _EMPTY


@dataclass(frozen=True, slots=True)
class GoldIndex:
    """The chunk set grouped by the document each chunk was cut from, so resolving a
    question is a lookup per required document rather than a scan of the whole corpus.
    Correctness is granular to the document, because that's all the source knows. See
    docs/where-the-answers-come-from.md before adding finer-grained truth.
    """

    by_document: Mapping[DocName, frozenset[ChunkId]]

    @classmethod
    def build(cls, chunks: Sequence[Chunk]) -> GoldIndex:
        grouped: dict[DocName, set[ChunkId]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.document, set()).add(chunk.name)
        return cls(by_document={doc: frozenset(names) for doc, names in grouped.items()})

    def correct(self, question: Question) -> Correct:
        """Which chunks this question's required documents were cut into."""
        cut_from: dict[DocName, frozenset[ChunkId]] = {}
        for document in question.required:
            names = self.by_document.get(document, _EMPTY)
            # Recall over an empty set is undefined and the obvious implementation
            # returns one, so a required document with no chunks must raise rather than
            # silently score as perfect.
            if not names and question.kind in ANSWERABLE_KINDS:
                raise EmptyCorrectSetError(f"{document} was cut into no chunk")
            if names:
                cut_from[document] = names

        # Built from cut_from, not resolved separately, so a chunk can only be primary
        # if it was already correct.
        return Correct(
            question=question.name,
            chunks=_EMPTY.union(*cut_from.values()) if cut_from else _EMPTY,
            primary=_EMPTY.union(
                *(cut_from.get(document, _EMPTY) for document in question.primary)
            ) if question.primary else _EMPTY,
        )


def resolve(question: Question, chunks: Sequence[Chunk]) -> Correct:
    """One question's correct set, against a chunk set this call groups from scratch.
    A caller resolving a whole question set should build one `GoldIndex` instead, or
    pay for the grouping once per question."""
    return GoldIndex.build(chunks).correct(question)
