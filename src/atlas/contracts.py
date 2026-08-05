"""The types every other module agrees on, and the protocols that keep them apart.

Every protocol here declares its members read-only: each is satisfied by frozen
dataclasses, and a frozen dataclass cannot satisfy a protocol declaring a settable
attribute.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NewType, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

DocName = NewType("DocName", str)  # "doc_bank_accounts_bank_accounts_(general)_031"
ChunkId = NewType("ChunkId", str)  # "doc_bank_accounts_bank_accounts_(general)_031#0000"
QuestionName = NewType("QuestionName", str)  # "task_004"

Arm = Literal["vector", "keyword"]
QuestionKind = Literal["lookup", "across_documents"]
Outcome = Literal["answered", "refused", "unknown", "not_applicable"]


@dataclass(frozen=True, slots=True, order=True)
class Range:
    """A character range inside a document. Carried by Chunk, so a chunk's text can be
    checked against the slice of the document it claims to be."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if not (0 <= self.start <= self.end):
            raise ValueError(f"invalid range {self.start}:{self.end}")


@dataclass(frozen=True, slots=True)
class Document:
    name: DocName
    kind: str  # "tau2_knowledge", ...
    text: str


@dataclass(frozen=True, slots=True)
class Collection:
    """The documents a source produced. What `CollectionSource` returns."""

    documents: tuple[Document, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    name: ChunkId
    document: DocName
    ordinal: int
    span: Range
    text: str


@dataclass(frozen=True, slots=True)
class Question:
    name: QuestionName
    text: str
    kind: QuestionKind
    # The documents a correct agent must consult; tau2's own `required_documents`, a
    # third-party gold set. Empty only for a question typed at the command line.
    required: tuple[DocName, ...]
    # Subset of `required` that carries the answer, as opposed to documents merely
    # consulted on the way to it. Empty means the source cannot tell (not "none of
    # them"); a consumer must branch on emptiness, which is why
    # `evals.ir_metrics.graded_ndcg_at_k` returns not-a-number when it is empty.
    primary: tuple[DocName, ...] = ()


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: ChunkId
    score: float
    rank: int  # STARTS AT ONE: reciprocal rank divides by it, and RRF by constant + it


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: str
    arm: Arm
    hits: tuple[Hit, ...]


@dataclass(frozen=True, slots=True)
class FusedResult:
    query: str
    hits: tuple[Hit, ...]
    arm_ranks: Mapping[ChunkId, Mapping[Arm, int]]  # which search found what, and where


@dataclass(frozen=True, slots=True)
class RerankResult:
    query: str
    hits: tuple[Hit, ...]
    input_order: tuple[ChunkId, ...]  # exists so a test can prove this was only a reordering


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cost_usd: float


ZERO_USAGE = Usage(input_tokens=0, output_tokens=0, cost_usd=0.0)


@dataclass(frozen=True, slots=True)
class Answer:
    question: QuestionName
    text: str
    cited: tuple[ChunkId, ...]
    outcome: Outcome
    shown: tuple[ChunkId, ...]  # exactly what went into the prompt
    model: str
    usage: Usage
    # Recorded, never raised, so a cited-but-unshown passage travels as data on the
    # answer rather than as an exception. See docs/checking-answers.md.
    violations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StageTiming:
    stage: str
    wall_ms: float


class JudgeRecord(Protocol):
    """The shape the cost summary needs off a judge verdict. Named here rather than
    imported so atlas never imports the measuring package that defines
    `evals.judge.JudgeVerdict`, which satisfies this structurally."""

    @property
    def usage(self) -> Usage: ...


class Grader(Protocol):
    """Scores one written answer against the passages it was shown. `texts`, not the
    chunks themselves, since a judge has no use for spans or ordinals."""

    def __call__(
        self, question: Question, answer: Answer, texts: Mapping[ChunkId, str]
    ) -> JudgeRecord: ...


@dataclass(frozen=True, slots=True)
class Record:
    """One question through the whole pipeline. No schema_version: no Record is ever
    serialised or read back; `evals.labels` keeps the version the store actually needs."""

    run_id: str
    question: QuestionName
    vector: SearchResult
    keyword: SearchResult
    fused: FusedResult
    reranked: RerankResult
    answer: Answer | None
    timings: tuple[StageTiming, ...]
    judge: JudgeRecord | None = None
    # None when tracing is off (no Langfuse credentials). Joining a measurement to its
    # trace is what turns "12% of answers failed" into "here are the twelve".
    trace_id: str | None = None


@runtime_checkable
class CollectionSource(Protocol):
    """What produces a collection. Takes a seed and nothing else: a source reading real
    documents has no registry to be handed and no say in cut settings."""

    @property
    def name(self) -> str: ...

    def __call__(self, seed: int) -> Collection: ...


class Chunker(Protocol):
    @property
    def name(self) -> str: ...

    def __call__(self, doc: Document) -> tuple[Chunk, ...]: ...


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...
    @property
    def model_version(self) -> str: ...
    @property
    def size(self) -> int: ...
    @property
    def normalised(self) -> bool: ...

    # Tokens this embedder has actually sent (not what a cache-served vector cost).
    @property
    def input_tokens(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...  # shape (n, size), float32


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def __call__(self, query: str, chunks: Sequence[Chunk]) -> RerankResult: ...


class AnswerWriter(Protocol):
    @property
    def model(self) -> str: ...

    def __call__(self, q: Question, shown: Sequence[Chunk]) -> Answer: ...
