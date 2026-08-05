"""The deliberately broken systems themselves, and the healthy control.

This module builds; `evals.adversaries` decides what fires and holds the table wiring the
two together. Nothing here mutates a System: `build_broken` copies with `replace` rather
than assigning through.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from atlas.config import RrfSettings
from atlas.contracts import (
    ZERO_USAGE,
    Answer,
    Chunk,
    ChunkId,
    DocName,
    Outcome,
    Question,
    QuestionName,
    Range,
)
from atlas.corpus.gold import GoldIndex
from atlas.models.generate import record_violations
from atlas.retrieval.fuse import rrf_fuse
from evals.injection import POISON_PAYLOAD
from evals.validity import Benchmark, build_benchmark

# The depth the cached default benchmark is built at, and nothing else: the depth a
# system is *scored* at is `benchmark.limit`, read separately everywhere below.
_LIMIT = 10


@dataclass(frozen=True, slots=True)
class System:
    name: str
    benchmark: Benchmark
    retrieved: dict[QuestionName, tuple[ChunkId, ...]]
    answers: dict[QuestionName, Answer]


def _retrieve(benchmark: Benchmark, question: Question) -> tuple[ChunkId, ...]:
    query_vector = benchmark.embedder.encode([question.text])[0]
    vector_result = benchmark.vector_index.search(question.text, query_vector, 50)
    keyword_result = benchmark.keyword_index.search(question.text, 50)
    fused = rrf_fuse(vector_result, keyword_result, RrfSettings())
    return tuple(hit.chunk for hit in fused.hits)


def _honest_answer(question: Question, shown: tuple[ChunkId, ...], correct: frozenset[ChunkId]) -> Answer:
    """Never states a wrong fact: answers from the correct chunk when it was shown,
    refuses otherwise. Used for the healthy control and for every broken system whose
    defect is not in the answer writer, so a detector reading answers never mistakes bad
    retrieval for a dishonest writer.
    """
    hit = next((name for name in shown if name in correct), None)
    cited: tuple[ChunkId, ...]
    outcome: Outcome
    if hit is not None:
        cited = (hit,)
        outcome = "answered"
        text = f"The shown fact, drawn from {hit}."
    else:
        cited = ()
        outcome = "refused"
        text = "The shown passages do not contain the answer."
    violations = record_violations(cited, shown, outcome, citation_required=True)
    return Answer(
        question=question.name, text=text, cited=cited, outcome=outcome, shown=shown,
        model="stub-honest", usage=ZERO_USAGE, violations=violations,
    )


def _build_healthy(benchmark: Benchmark) -> System:
    """The healthy control over any already built benchmark, not only the cached default
    one. Split out of `build_healthy` so a test can prove the same invariant against a
    configuration other than the default.

    Cut to `benchmark.limit` rather than the module's `_LIMIT`: those coincide at ten
    today, but a control cut to the constant would be measured over slots it was never
    given the moment a caller asked for a different depth.
    """
    retrieved: dict[QuestionName, tuple[ChunkId, ...]] = {}
    answers: dict[QuestionName, Answer] = {}
    for question in benchmark.questions:
        shown = _retrieve(benchmark, question)[:benchmark.limit]
        retrieved[question.name] = shown
        correct = benchmark.gold.correct(question).chunks
        answers[question.name] = _honest_answer(question, shown, correct)
    return System(name="healthy", benchmark=benchmark, retrieved=retrieved, answers=answers)


@lru_cache(maxsize=1)
def build_healthy() -> System:
    """Cached: a pure function of the vendored corpus, and a walk over all broken systems
    would otherwise embed the corpus and retrieve all 97 tasks repeatedly.
    """
    return _build_healthy(build_benchmark(seed=0, limit=_LIMIT))


def build_broken(name: str) -> System:
    healthy = build_healthy()
    benchmark = healthy.benchmark

    if name == "empty_answer_writer":
        answers = {}
        for question in benchmark.questions:
            shown = healthy.retrieved[question.name]
            answers[question.name] = Answer(
                question=question.name, text="", cited=(), outcome="answered", shown=shown,
                model="stub-empty", usage=ZERO_USAGE,
                violations=record_violations((), shown, "answered", True),
            )
        return replace(healthy, name=name, answers=answers)

    if name == "search_returns_everything":
        all_names = tuple(sorted(f.name for f in benchmark.chunks))
        retrieved = {q.name: all_names for q in benchmark.questions}
        answers = {
            q.name: _honest_answer(
                q, all_names[:benchmark.limit], benchmark.gold.correct(q).chunks
            )
            for q in benchmark.questions
        }
        return System(name=name, benchmark=benchmark, retrieved=retrieved, answers=answers)

    if name == "ignores_account_records":
        answers = dict(healthy.answers)
        for question in benchmark.questions:
            if any(str(doc).startswith("doc_bank_accounts") for doc in question.required):
                shown = healthy.retrieved[question.name]
                answers[question.name] = Answer(
                    question=question.name, text="This account does not appear in the records.",
                    cited=(), outcome="unknown", shown=shown, model="stub-ignores-accounts",
                    usage=ZERO_USAGE, violations=(),
                )
        return replace(healthy, name=name, answers=answers)

    if name == "same_ten_passages":
        fixed = benchmark.fixed_list
        retrieved = {q.name: fixed for q in benchmark.questions}
        answers = {
            q.name: _honest_answer(q, fixed, benchmark.gold.correct(q).chunks)
            for q in benchmark.questions
        }
        return System(name=name, benchmark=benchmark, retrieved=retrieved, answers=answers)

    if name == "retrieves_nothing":
        retrieved = {q.name: () for q in benchmark.questions}
        answers = {
            q.name: _honest_answer(q, (), benchmark.gold.correct(q).chunks)
            for q in benchmark.questions
        }
        return System(name=name, benchmark=benchmark, retrieved=retrieved, answers=answers)

    if name == "poisoned_corpus":
        # Payload sits inside otherwise plausible policy text: a chunk that is nothing but
        # an instruction is a strawman, real poisoning hides in a document somebody had a
        # reason to ingest.
        payload = POISON_PAYLOAD
        poisoned = ChunkId("poisoned.0000")
        chunks = (
            *benchmark.chunks,
            Chunk(name=poisoned, document=DocName("gen.poisoned"), ordinal=0,
                  span=Range(0, len(payload)), text=payload),
        )
        # Gold index rebuilt from the same chunks or detectors would resolve against the
        # untainted corpus.
        tainted = replace(benchmark, chunks=chunks, gold=GoldIndex.build(chunks))
        # Retrieved first for every question: a payload nothing retrieves is inert.
        retrieved = {q.name: (poisoned, *healthy.retrieved[q.name][:benchmark.limit - 1])
                     for q in benchmark.questions}
        answers = {
            q.name: _honest_answer(q, retrieved[q.name], benchmark.gold.correct(q).chunks)
            for q in benchmark.questions
        }
        return System(name=name, benchmark=tainted, retrieved=retrieved, answers=answers)

    if name == "fluent_wrong_citations":
        decoy = sorted(f.name for f in benchmark.chunks)[0]
        answers = {}
        for question in benchmark.questions:
            shown = healthy.retrieved[question.name]
            cited = (decoy,)
            answers[question.name] = Answer(
                question=question.name, text="Within sixty days, as shown.", cited=cited,
                outcome="answered", shown=shown, model="stub-fluent", usage=ZERO_USAGE,
                violations=record_violations(cited, shown, "answered", True),
            )
        return replace(healthy, name=name, answers=answers)

    raise ValueError(f"no broken system named {name!r}")
