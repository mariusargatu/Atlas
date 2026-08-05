from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from atlas.config import Settings
from atlas.contracts import (
    Answer,
    AnswerWriter,
    Chunk,
    ChunkId,
    Embedder,
    Grader,
    JudgeRecord,
    Question,
    Record,
    Reranker,
    StageTiming,
)
from atlas.models.embed import get_embedder
from atlas.models.generate import get_answer_writer
from atlas.retrieval.dense import VectorIndex
from atlas.retrieval.fuse import rrf_fuse
from atlas.retrieval.rerank import get_reranker
from atlas.retrieval.sparse import KeywordIndex, build_keyword_index
from atlas.trace import SpanKind, current_trace_id, observe, stage_timer

# The canonical stage names. Kept here, not derived, so renaming a stage fails a check
# rather than moving silently: tests/pipeline/test_contracts.py keeps its own copy.
# "judge" is the only optional one; a run that doesn't judge just contributes no rows
# for it. "embed_query" is timed separately from "vector" since they're different costs:
# a network round trip versus a dot product against an in-memory matrix.
STAGES = ("embed_query", "vector", "keyword", "fuse", "rerank", "answer", "judge")


@dataclass(frozen=True, slots=True)
class PreparedCorpus:
    """Everything a question needs that does not depend on the question. Embedding the
    chunks, building both indexes, and loading the cross encoder are corpus-level costs;
    paying them per question would re-bill the whole corpus for every question asked."""

    settings: Settings
    chunks: tuple[Chunk, ...]
    by_name: Mapping[ChunkId, Chunk]
    embedder: Embedder
    vector_index: VectorIndex
    keyword_index: KeywordIndex
    reranker: Reranker
    writer: AnswerWriter

    @classmethod
    def build(
        cls,
        settings: Settings,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
        writer: AnswerWriter | None = None,
    ) -> PreparedCorpus:
        """Pays the costs that belong to the corpus once. `embedder` and `writer` default
        to the real clients but stay arguments so a test can substitute a fake one. Both
        are constructed before a single chunk is embedded, so a missing or rejected key
        fails here rather than after the corpus has been paid for."""
        frozen = tuple(chunks)
        chosen = embedder if embedder is not None else get_embedder(settings.embedding)
        chosen_writer = writer if writer is not None else get_answer_writer(settings.answer)
        matrix = observe(chosen.encode, kind="embedding")([f.text for f in frozen])
        return cls(
            settings=settings,
            chunks=frozen,
            by_name={f.name: f for f in frozen},
            embedder=chosen,
            vector_index=VectorIndex.build(frozen, matrix),
            keyword_index=build_keyword_index(frozen, settings.sparse),
            reranker=get_reranker(settings.rerank),
            writer=chosen_writer,
        )

    def ask(
        self, question: Question, generate: bool = True, grader: Grader | None = None
    ) -> Record:
        """One question through every stage, inside one trace. This wrapper provides the
        root span: without it, every `observe`d stage in `_ask` starts its own trace
        instead of sharing one. Named for the question, so the trace list reads as the
        question set rather than as identical rows called "_ask"."""
        return observe(self._ask, kind="chain", name=str(question.name))(
            question, generate, grader
        )

    def _ask(
        self, question: Question, generate: bool = True, grader: Grader | None = None
    ) -> Record:
        """Runs one question through every stage against this prepared corpus, timing and
        tracing each, and returns one frozen Record.

        `generate=False` stops after reranking, returning a Record with `answer=None`,
        for inspecting what retrieval found without paying to have it written up.
        `grader` scores the answer onto `Record.judge` and defaults to None since a
        verdict is a second billed call per question.
        """
        settings = self.settings
        timings: list[StageTiming] = []

        def timed[T](stage: str, kind: SpanKind, call: Callable[..., T], *args: Any) -> T:
            """One stage: timed, traced, and named for the stage rather than for whatever
            is being called (the tracer would otherwise fall back to reading `__name__`,
            which a reranker or writer instance does not carry)."""
            with stage_timer(stage, timings):
                return observe(call, kind=kind, name=stage)(*args)

        query_vector = timed("embed_query", "embedding", self.embedder.encode, [question.text])[0]
        vector = timed(
            "vector", "retriever", self.vector_index.search,
            question.text, query_vector, settings.search.vector_candidates,
        )
        keyword = timed(
            "keyword", "retriever", self.keyword_index.search,
            question.text, settings.search.keyword_candidates,
        )
        fused = timed("fuse", "retriever", rrf_fuse, vector, keyword, settings.fusion)
        candidates = [self.by_name[hit.chunk] for hit in fused.hits]
        reranked = timed("rerank", "retriever", self.reranker, question.text, candidates)

        answer: Answer | None = None
        verdict: JudgeRecord | None = None
        if generate:
            shown = [self.by_name[hit.chunk] for hit in reranked.hits][: settings.answer.max_shown]
            answer = timed("answer", "generation", self.writer, question, shown)
            if grader is not None:
                # Read back off the answer rather than rebuilt from `shown`, so the judge
                # always grades against what the writer actually saw.
                texts = {name: self.by_name[name].text for name in answer.shown}
                verdict = timed("judge", "evaluator", grader, question, answer, texts)

        return Record(
            run_id=settings.run_id, question=question.name, vector=vector,
            keyword=keyword, fused=fused, reranked=reranked, answer=answer,
            timings=tuple(timings), judge=verdict,
            # Read while the spans this question produced are still the current trace.
            trace_id=current_trace_id(),
        )


def run_question(
    settings: Settings,
    question: Question,
    chunks: Sequence[Chunk],
    generate: bool = True,
) -> Record:
    """One question against a corpus prepared for it alone. Wasteful for many questions:
    preparing the corpus is the expensive, billed part, so a caller with a question set
    should build one `PreparedCorpus` and call `ask` on it in a loop instead."""
    return PreparedCorpus.build(settings, chunks).ask(question, generate=generate)
