from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from dotenv import load_dotenv
from hypothesis import settings as property_settings

from atlas.config import ChunkSettings, EmbeddingSettings, Settings
from atlas.contracts import (
    Answer,
    Chunk,
    ChunkId,
    DocName,
    FusedResult,
    Hit,
    Outcome,
    Question,
    QuestionName,
    Range,
    Record,
    RerankResult,
    SearchResult,
    StageTiming,
    Usage,
)
from atlas.corpus.chunk import cut
from atlas.corpus.tau2 import Tau2Source
from atlas.models.embed import CachedEmbedder, get_embedder
from evals.judge import JudgeVerdict, Verdict
from evals.validity import build_benchmark

pytest_plugins = ("pytester", "tests.no_skip_guard")

# atlas.trace decides at import time whether tracing is on, by reading env vars. Anything
# imported above load_dotenv() below must not reach it (transitively) or it'll decide
# before .env is read. evals.report reaches it via atlas.pipeline, so the one fixture
# needing evals.report imports it locally instead of at module scope.

property_settings.register_profile("checks", max_examples=200, deadline=None)
property_settings.register_profile("development", max_examples=25)
property_settings.register_profile("thorough", max_examples=2000)

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

REQUIRED_KEY = "OPENAI_API_KEY"


def pytest_configure(config: pytest.Config) -> None:
    """Fails fast before collection: every test embeds against a real endpoint, so a
    missing key would otherwise fail the whole suite one SDK error at a time."""
    if not os.environ.get(REQUIRED_KEY):
        raise pytest.UsageError(
            f"{REQUIRED_KEY} is not set, and this suite runs real models. Put it in "
            f"{ROOT / '.env'} or export it. Every check embeds against a real "
            f"endpoint, so there is no subset of this suite that runs without it."
        )


@pytest.fixture(scope="session")
def source():
    return Tau2Source()


@pytest.fixture(scope="session")
def documents(source):
    return source.documents()


@pytest.fixture(scope="session")
def collection(source):
    return source(0)


@pytest.fixture(scope="session")
def large(source):
    return source.questions()


@pytest.fixture
def small(large):
    return large[:40]


@pytest.fixture(scope="session")
def chunks(collection):
    return cut(collection.documents, ChunkSettings())


@pytest.fixture
def lookup_question(large):
    return min(large, key=lambda q: len(q.required))


@pytest.fixture
def settings():
    return Settings()


class _TieHeavy:
    def __init__(self, chunks, matrix, query_text, query_vector):
        self.chunks = chunks
        self.matrix = matrix
        self.query_text = query_text
        self.query_vector = query_vector


@pytest.fixture(scope="session")
def tie_heavy():
    # Session scoped: a function scoped fixture inside a property check trips hypothesis's
    # own health check. Two chunks share identical text so their vectors tie exactly.
    texts = [
        "alpha document text", "duplicate shared text", "duplicate shared text",
        "beta document text", "gamma document text", "delta document text",
    ]
    chunks = tuple(
        Chunk(
            name=ChunkId(f"doc.{i}#0000"), document=DocName(f"doc.{i}"), ordinal=0,
            span=Range(0, len(text)), text=text,
        )
        for i, text in enumerate(texts)
    )
    embedder = CachedEmbedder(get_embedder(EmbeddingSettings()))
    matrix = embedder.encode(texts)
    query_text = "duplicate shared text"
    query_vector = embedder.encode([query_text])[0]
    return _TieHeavy(chunks=chunks, matrix=matrix, query_text=query_text, query_vector=query_vector)


@pytest.fixture
def one_vector_hit():
    return SearchResult(query="q", arm="vector",
                         hits=(Hit(chunk=ChunkId("frag.0000"), score=0.9, rank=1),))


@pytest.fixture
def empty_keyword():
    return SearchResult(query="q", arm="keyword", hits=())


@pytest.fixture
def tied_searches():
    # Each chunk found by exactly one arm at rank one; at rrf_fuse's default weights
    # the two rows land on the same score.
    vector = SearchResult(query="q", arm="vector",
                           hits=(Hit(chunk=ChunkId("b"), score=0.9, rank=1),))
    keyword = SearchResult(query="q", arm="keyword",
                            hits=(Hit(chunk=ChunkId("a"), score=5.0, rank=1),))
    return vector, keyword


@pytest.fixture
def second_ranked_in_keyword_only():
    # Rank blending narrowly favours "top" at even weights and flips to favour
    # "second" once the keyword arm is weighted heavily.
    vector = SearchResult(query="q", arm="vector",
                           hits=(Hit(chunk=ChunkId("top"), score=0.9, rank=1),))
    keyword = SearchResult(query="q", arm="keyword",
                            hits=(Hit(chunk=ChunkId("second"), score=5.0, rank=2),))
    return vector, keyword


@pytest.fixture
def deep_and_shallow():
    # More chunks than any reranker's depth_in, with a real spread of relevance to
    # "what does the plan cost", so the cross encoder has genuine text to reorder.
    texts = [
        "a card dispute must be filed within 60 days of the statement date",
        "installation appointments run monday through saturday, 8am to 6pm",
        "our support macros cover password resets and outage reporting",
        "a provisional credit is issued within 10 business days of filing",
        "regional coverage maps are published every quarter for each area",
        "early termination before the contract ends adds a one time fee",
        "router firmware updates are pushed automatically overnight",
        "the copper200 plan costs 35 dollars per month for legacy lines",
        "customer accounts list an entitled data cap and a billing cycle",
        "static ip addresses are available as a paid add on to any plan",
        "the weather this week has been unusually mild for the season",
        "our office plants are watered every Tuesday by the facilities team",
    ]
    return tuple(
        Chunk(
            name=ChunkId(f"deep.{i:04d}"), document=DocName("gen.deep"), ordinal=i,
            span=Range(0, len(text)), text=text,
        )
        for i, text in enumerate(texts)
    )


class _SmallRun:
    def __init__(self, settings, question, chunks):
        self.settings = settings
        self.question = question
        self.chunks = chunks


@pytest.fixture
def small_run(settings, small, chunks):
    question = min(small, key=lambda q: len(q.required))
    return _SmallRun(settings=settings, question=question, chunks=chunks)


@pytest.fixture(scope="module")
def benchmark():
    return build_benchmark(seed=0)


@pytest.fixture
def one_question():
    return Question(
        name=QuestionName("q_judge_0001"),
        text="How do I dispute a card transaction?",
        kind="lookup",
        required=(DocName("doc_bank_accounts_bank_accounts_(general)_031"),),
    )


@pytest.fixture
def one_answer(one_question):
    return Answer(
        question=one_question.name, text="Within sixty days.",
        cited=(ChunkId("card_dispute_policy#0000"),), outcome="answered",
        shown=(ChunkId("card_dispute_policy#0000"),), model="stub",
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
    )


@pytest.fixture
def shown_texts(one_answer):
    return {
        name: "A card dispute must be filed within 60 days of the statement date."
        for name in one_answer.shown
    }


@pytest.fixture
def stub_judge():
    @dataclass(frozen=True, slots=True)
    class _JudgeReply:
        verdict: Verdict
        reason: str
        usage: Usage

    class _StubJudge:
        def complete(self, prompt: str) -> _JudgeReply:
            return _JudgeReply(
                verdict="pass", reason="cites the shown passage and states the right amount",
                usage=Usage(input_tokens=80, output_tokens=15, cost_usd=0.002),
            )

    return _StubJudge()


@pytest.fixture
def verdicts():
    return (
        JudgeVerdict(
            question=QuestionName("q0042"), verdict="fail", reason="price disagrees with the record",
            rubric_version="1.0.0", prompt_version="1.0.0", model="stub",
            usage=Usage(input_tokens=80, output_tokens=15, cost_usd=0.002),
        ),
    )


@pytest.fixture
def recorded_runs():
    base_ms = {"vector": 5.0, "keyword": 2.0, "fuse": 0.5, "rerank": 30.0, "answer": 200.0}
    empty_vector = SearchResult(query="q", arm="vector", hits=())
    empty_keyword = SearchResult(query="q", arm="keyword", hits=())
    empty_blend = FusedResult(query="q", hits=(), arm_ranks={})
    empty_rerank = RerankResult(query="q", hits=(), input_order=())

    records = []
    for i in range(20):
        name = QuestionName(f"q{i:04d}")
        timings = tuple(
            StageTiming(stage=stage, wall_ms=base_ms[stage] + (i % 3))
            for stage in ("vector", "keyword", "fuse", "rerank", "answer")
        )
        answer = Answer(
            question=name, text="Within sixty days.", cited=(), outcome="answered",
            shown=(), model="stub", usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.001 * (i + 1)),
        )
        judge = JudgeVerdict(
            question=name, verdict="pass", reason="cites the shown passage",
            rubric_version="1.0.0", prompt_version="1.0.0", model="stub-judge",
            usage=Usage(input_tokens=80, output_tokens=15, cost_usd=0.002 * (i + 1)),
        ) if i % 2 == 0 else None
        records.append(Record(
            run_id="deadbeefcafe", question=name, vector=empty_vector, keyword=empty_keyword,
            fused=empty_blend, reranked=empty_rerank, answer=answer, timings=timings, judge=judge,
        ))
    return tuple(records)


@pytest.fixture
def runs_with_one_slow_question(recorded_runs):
    records = list(recorded_runs)
    slow = records[-1]
    louder_timings = tuple(
        replace(t, wall_ms=t.wall_ms * 10) if t.stage == "answer" else t for t in slow.timings
    )
    records[-1] = replace(slow, timings=louder_timings)
    return tuple(records)


@pytest.fixture
def verdict_runs():
    # Factory, not a fixed pair: both sides are scored on the same 200 questions so the
    # interval is taken over per-question differences rather than two independent samples.
    # Imported locally, same reason as the module docstring above (atlas.trace / load_dotenv).
    from evals.report import RunSample

    def build(change: float, spread: float, tail_change: float):
        names = tuple(QuestionName(f"q{i:04d}") for i in range(200))
        before_quality = {q: 0.20 + 0.60 * ((i * 37) % 200) / 199 for i, q in enumerate(names)}
        after_quality = {
            q: before_quality[q] + change + spread * (2 * (2 * ((i * 13) % 97) / 96 - 1))
            for i, q in enumerate(names)
        }
        before = RunSample(quality=before_quality, slowest_tenth_ms=100.0, cost_usd=0.05)
        after = RunSample(quality=after_quality, slowest_tenth_ms=100.0 + tail_change, cost_usd=0.05 + 0.01)
        return before, after

    return build


@pytest.fixture
def two_chunks():
    # Named to match the SHOWN constant test_contracts.py cites against, so a fake
    # model can legitimately cite one of them.
    names = ("card_dispute_policy#0001", "card_dispute_policy#0002")
    texts = ["a card dispute must be filed within 60 days", "provisional credit is issued within ten days"]
    return tuple(
        Chunk(
            name=ChunkId(name), document=DocName("card_dispute_policy"), ordinal=i,
            span=Range(0, len(text)), text=text,
        )
        for i, (name, text) in enumerate(zip(names, texts, strict=True))
    )


@pytest.fixture
def fake_model():
    @dataclass(frozen=True, slots=True)
    class _FakeReply:
        text: str
        cited: tuple[ChunkId, ...]
        outcome: Outcome
        reason: str
        usage: Usage

    class _FakeModel:
        def __init__(self, text: str, cited: list, outcome: Outcome, reason: str) -> None:
            self._reply = _FakeReply(
                text=text, cited=tuple(cited), outcome=outcome, reason=reason,
                usage=Usage(input_tokens=50, output_tokens=10, cost_usd=0.001),
            )

        def complete(self, prompt: str) -> _FakeReply:
            return self._reply

    return _FakeModel


