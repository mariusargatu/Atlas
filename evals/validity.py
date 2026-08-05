from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from atlas.config import ChunkSettings, EmbeddingSettings, RrfSettings, SparseSettings
from atlas.contracts import Chunk, ChunkId, Embedder, Question, QuestionName
from atlas.corpus.chunk import cut
from atlas.corpus.gold import ANSWERABLE_KINDS, EmptyCorrectSetError, GoldIndex
from atlas.corpus.registry import get_source
from atlas.models.embed import CachedEmbedder, get_embedder
from atlas.retrieval.dense import VectorIndex
from atlas.retrieval.fuse import rrf_fuse
from atlas.retrieval.sparse import KeywordIndex, build_keyword_index
from evals.baselines import best_constant_ranking
from evals.ir_metrics import Bounded, recall_at_k
from evals.stats import paired_comparison, resample_interval


@dataclass(frozen=True, slots=True)
class ValidityResult:
    passed: bool
    detail: str


def correct_answers_exist(
    questions: tuple[Question, ...], chunks: tuple[Chunk, ...]
) -> ValidityResult:
    gold = GoldIndex.build(chunks)
    failures: list[str] = []
    for question in questions:
        if question.kind not in ANSWERABLE_KINDS:
            continue
        try:
            gold.correct(question)
        except EmptyCorrectSetError as error:
            failures.append(f"{question.name}: {error}")
    if failures:
        return ValidityResult(passed=False, detail="; ".join(failures))
    return ValidityResult(passed=True, detail="every question resolves to a correct chunk")


# A fraction of what the question set actually allows, not of one: recall@10's mean
# maximum on this corpus is 0.604, so a raw-recall threshold could never fire.
_HEADROOM_ATTAINED = 0.98


@dataclass(frozen=True, slots=True)
class Benchmark:
    """The chunk set, both indexes, the question set and the two scorers a
    validity check needs. real_scorer and fixed_list_scorer are fields, not
    methods, so a test can swap either with dataclasses.replace to prove a check
    goes red without touching the collection that backs it."""

    chunks: tuple[Chunk, ...]
    # Kept beside `chunks` rather than resolved on demand. Anything that replaces
    # `chunks` must replace this too: see `evals.systems.build_broken`.
    gold: GoldIndex
    vector_index: VectorIndex
    keyword_index: KeywordIndex
    embedder: Embedder
    questions: tuple[Question, ...]
    limit: int
    # Computed once at build time: a property of the collection, not of any question.
    fixed_list: tuple[ChunkId, ...]
    real_scorer: Callable[[Question, Benchmark], float]
    fixed_list_scorer: Callable[[Question, Benchmark], float]


def _real_score(question: Question, benchmark: Benchmark) -> float:
    correct = benchmark.gold.correct(question).chunks
    if not correct:
        return math.nan
    query_vector = benchmark.embedder.encode([question.text])[0]
    vector_result = benchmark.vector_index.search(question.text, query_vector, 50)
    keyword_result = benchmark.keyword_index.search(question.text, 50)
    fused = rrf_fuse(vector_result, keyword_result, RrfSettings())
    ranked = tuple(hit.chunk for hit in fused.hits)
    return recall_at_k(ranked, correct, benchmark.limit) if ranked else 0.0


def _fixed_list_score(question: Question, benchmark: Benchmark) -> float:
    correct = benchmark.gold.correct(question).chunks
    if not correct:
        return math.nan
    return recall_at_k(benchmark.fixed_list, correct, benchmark.limit)


@dataclass(frozen=True, slots=True)
class _EmbeddedCorpus:
    chunks: tuple[Chunk, ...]
    gold: GoldIndex
    vector_index: VectorIndex
    keyword_index: KeywordIndex
    embedder: Embedder


_EMBEDDED: dict[tuple[str, str, int, bool, str | None], _EmbeddedCorpus] = {}


def _embedded_corpus(settings: EmbeddingSettings, cache_dir: str | None) -> _EmbeddedCorpus:
    """The corpus, embedded once per process per model identity.

    `build_benchmark` is called once per seed, and a seed resamples the question set
    without touching a single document, so embedding belongs here rather than there.
    Keyed on the full model identity so a matrix made by one model is never served to a
    run that asked for another.
    """
    key = (
        settings.model_name, settings.model_version, settings.size,
        settings.normalise, cache_dir,
    )
    if key not in _EMBEDDED:
        source = get_source()
        chunks = cut(source(0).documents, ChunkSettings())
        # With no cache directory this is memory only, which is all a check needs.
        embedder = CachedEmbedder(get_embedder(settings), cache_dir)
        matrix = embedder.encode([f.text for f in chunks])
        _EMBEDDED[key] = _EmbeddedCorpus(
            chunks=chunks,
            gold=GoldIndex.build(chunks),
            vector_index=VectorIndex.build(chunks, matrix),
            keyword_index=build_keyword_index(chunks, SparseSettings()),
            embedder=embedder,
        )
    return _EMBEDDED[key]


def build_benchmark(
    seed: int,
    cache_dir: str | None = None,
    limit: int = 10,
    embedding: EmbeddingSettings = EmbeddingSettings(),
) -> Benchmark:
    """Wires up everything a validity check needs to score the corpus.

    The corpus is a fixed third-party artefact, so `seed` does not regenerate it; it
    draws a bootstrap resample of the question set instead, measuring the sampling
    variability of the headline number directly.

    Documents don't vary with the seed, so neither do their vectors; `_embedded_corpus`
    is what stops each seed paying for them again.
    """
    source = get_source()
    every_question = source.questions()
    if seed == 0:
        questions = every_question
    else:
        rng = random.Random(f"question-resample:{seed}")
        questions = tuple(rng.choices(every_question, k=len(every_question)))

    corpus = _embedded_corpus(embedding, cache_dir)
    return Benchmark(
        chunks=corpus.chunks, gold=corpus.gold, vector_index=corpus.vector_index,
        keyword_index=corpus.keyword_index, embedder=corpus.embedder,
        questions=questions, limit=limit,
        # Built at `limit`, not at `best_constant_ranking`'s default of ten: both scorers
        # must compare arms scored at the same depth, or the question-ignoring arm would
        # be handicapped whenever a caller asks for a depth other than ten.
        fixed_list=best_constant_ranking(corpus.chunks, every_question, size=limit),
        real_scorer=_real_score, fixed_list_scorer=_fixed_list_score,
    )


def headline_recall(benchmark: Benchmark) -> float:
    """recall_at_k at the collection's own limit, averaged over every question
    `benchmark.real_scorer` can score."""
    scores = [benchmark.real_scorer(q, benchmark) for q in benchmark.questions]
    finite = [s for s in scores if not math.isnan(s)]
    return statistics.fmean(finite) if finite else math.nan


def headline_bounded_recall(benchmark: Benchmark) -> Bounded:
    """The headline recall with the maximum it was measured against.

    The ceiling is computed from the gold sets alone rather than by re-running retrieval:
    `_real_score` always fuses at least `limit` candidates, so the maximum is
    `min(limit, |correct|) / |correct|` per question, and re-retrieving to obtain it would
    embed the question set a second time inside a check that blocks a merge.
    """
    values = [benchmark.real_scorer(q, benchmark) for q in benchmark.questions]
    ceilings = []
    for question in benchmark.questions:
        correct = benchmark.gold.correct(question).chunks
        ceilings.append(min(benchmark.limit, len(correct)) / len(correct) if correct else math.nan)
    usable = [(v, c) for v, c in zip(values, ceilings, strict=True)
              if not math.isnan(v) and not math.isnan(c)]
    if not usable:
        return Bounded(math.nan, math.nan)
    return Bounded(statistics.fmean(v for v, _ in usable),
                   statistics.fmean(c for _, c in usable))


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    passed: bool
    message: str
    real_score: float
    fixed_list_score: float
    # The paired interval on the gap, where there is one. `headroom_check` compares a
    # figure against its own ceiling, so it has no second arm to pair with and leaves
    # these NaN.
    low: float = math.nan
    high: float = math.nan


def fixed_list_check(benchmark: Benchmark, seed: int = 0) -> CheckOutcome:
    """Does the real system beat the ranking that never reads the question, by a margin a
    paired bootstrap can tell from zero?

    Uses `evals.stats.paired_comparison` rather than an unpaired variance estimate: the
    two arms are scored on the same questions, so the variance they share cancels, and
    the right bar is the spread of the per-question *difference*. The gate is that the
    interval clears zero; there is no threshold constant to tune.
    """
    real: dict[QuestionName, float] = {}
    fixed: dict[QuestionName, float] = {}
    for question in benchmark.questions:
        real_value = benchmark.real_scorer(question, benchmark)
        fixed_value = benchmark.fixed_list_scorer(question, benchmark)
        # Dropped from both or from neither: keeping a question on one side alone would
        # compare two different sets.
        if math.isnan(real_value) or math.isnan(fixed_value):
            continue
        real[question.name] = real_value
        fixed[question.name] = fixed_value

    if len(real) < 2:
        return CheckOutcome(
            passed=False,
            message=(f"only {len(real)} question(s) could be scored on both arms, which is "
                     "too few to pair a comparison over"),
            real_score=math.nan, fixed_list_score=math.nan,
        )

    real_score = statistics.fmean(real.values())
    fixed_list_score = statistics.fmean(fixed.values())
    gap = real_score - fixed_list_score
    comparison = paired_comparison(before=fixed, after=real, seed=seed)
    # Strictly greater, so an arm that merely ties produces [0.000, 0.000] and fails.
    passed = comparison.low > 0.0
    verdict = "clears" if passed else "does not clear"
    message = (
        f"real recall {real_score:.3f} against the same ten passages fixed list "
        f"{fixed_list_score:.3f}, a paired gap of {gap:.3f} with interval "
        f"[{comparison.low:+.4f}, {comparison.high:+.4f}] over {comparison.questions} "
        f"questions, which {verdict} zero"
    )
    return CheckOutcome(
        passed=passed, message=message, real_score=real_score,
        fixed_list_score=fixed_list_score, low=comparison.low, high=comparison.high,
    )


def headroom_check(benchmark: Benchmark) -> CheckOutcome:
    """Whether anything downstream still has room to show an improvement.

    Against the ceiling this question set allows, never against one; see
    `_HEADROOM_ATTAINED`.
    """
    bounded = headline_bounded_recall(benchmark)
    real_score = bounded.value
    attained = bounded.attained
    passed = math.isnan(attained) or attained < _HEADROOM_ATTAINED
    if passed:
        message = (f"recall sits at {real_score:.3f} of an achievable {bounded.ceiling:.3f} "
                   f"({attained:.0%} attained), with room to detect a change")
    else:
        message = (
            f"recall sits at {real_score:.3f} of an achievable {bounded.ceiling:.3f} "
            f"({attained:.0%} attained), which is the ceiling: no comparison in this benchmark "
            "has room to show improvement. Report recall at one and three chunks alongside the "
            "full limit, and add filler documents until real headroom returns."
        )
    return CheckOutcome(passed=passed, message=message, real_score=real_score, fixed_list_score=math.nan)


@dataclass(frozen=True, slots=True)
class ChanceOutcome:
    low: float
    high: float


def chance_check(benchmark: Benchmark, seed: int) -> ChanceOutcome:
    """A random ranking's recall, reported with an interval.

    This blocks a merge: `tests/measurement/test_validity.py` asserts the real system's
    headline recall clears `chance.high`, and that test carries no `reporting` marker.
    """
    rng = random.Random(seed)
    names = tuple(f.name for f in benchmark.chunks)
    scores: list[float] = []
    for question in benchmark.questions:
        correct = benchmark.gold.correct(question).chunks
        if not correct:
            continue
        shuffled = rng.sample(names, len(names))
        scores.append(recall_at_k(shuffled, correct, benchmark.limit))
    if not scores:
        return ChanceOutcome(low=math.nan, high=math.nan)
    low, high = resample_interval(scores, seed)
    return ChanceOutcome(low=low, high=high)
