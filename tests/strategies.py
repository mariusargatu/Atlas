from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from hypothesis import strategies as st

from atlas.config import ChunkSettings, RrfSettings, SparseSettings
from atlas.contracts import (
    Chunk,
    ChunkId,
    DocName,
    Document,
    Hit,
    Range,
    RerankResult,
    SearchResult,
)
from atlas.retrieval.dense import VectorIndex
from atlas.retrieval.fuse import rrf_fuse
from atlas.retrieval.ranking import ranked_hits
from atlas.retrieval.sparse import build_keyword_index

_RERANK_WORD = re.compile(r"[a-zA-Z0-9]+")


class StubReranker:
    """Test double, not a production backend. Scores by word overlap between query
    and chunk text, so a test can construct a strictly worse candidate by controlling
    which words go into a chunk."""

    name = "stub"

    def __init__(self, depth_in: int, depth_out: int) -> None:
        self.depth_in = depth_in
        self.depth_out = depth_out

    def __call__(self, query: str, chunks: Sequence[Chunk]) -> RerankResult:
        query_words = {w.lower() for w in _RERANK_WORD.findall(query)}
        prefix = tuple(chunks)[: self.depth_in]
        scores = [
            float(len(query_words & {w.lower() for w in _RERANK_WORD.findall(f.text)}))
            for f in prefix
        ]
        return RerankResult(
            query=query,
            hits=ranked_hits(zip((f.name for f in prefix), scores, strict=True), self.depth_out),
            input_order=tuple(f.name for f in prefix),
        )


@dataclass(frozen=True, slots=True)
class Written:
    """A generated document plus the character ranges of the values written into it.

    Document itself carries no spans (tau2's gold set names documents, never offsets),
    but a chunker property needs to know where a short value landed to assert no
    boundary split it."""

    document: Document
    spans: tuple[Range, ...]


class _Builder:
    def __init__(self, name: DocName) -> None:
        self._name = name
        self._pieces: list[str] = []
        self._length = 0
        self._spans: list[Range] = []

    def write(self, text: str) -> None:
        self._pieces.append(text)
        self._length += len(text)

    def emit(self, value: str) -> None:
        start = self._length
        self._pieces.append(value)
        self._length += len(value)
        self._spans.append(Range(start, self._length))

    def finish(self) -> Written:
        return Written(
            document=Document(name=self._name, kind="generated", text="".join(self._pieces)),
            spans=tuple(self._spans),
        )

_WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=3, max_size=10)

# Fixed query every reranker property sends: "what does the plan cost". Fillers never
# draw these words, so overlap comes only from keywords deliberately written in.
_RERANK_QUERY_WORDS = frozenset({"what", "does", "the", "plan", "cost"})
_RERANK_KEYWORDS = ("plan", "cost")
_FILLER_WORD = _WORD.filter(lambda w: w not in _RERANK_QUERY_WORDS)


def _sentence(draw) -> str:
    words = draw(st.lists(_WORD, min_size=3, max_size=10))
    return " ".join(words) + ". "


@st.composite
def documents(draw) -> Document:
    """A document with one to four written values, at generated sentence boundaries."""
    name = DocName(f"gen.doc.{draw(st.integers(min_value=0, max_value=10**9))}")
    b = _Builder(name)
    num_facts = draw(st.integers(min_value=1, max_value=4))
    for i in range(num_facts):
        b.write(_sentence(draw))
        value = draw(_WORD)
        b.emit(value)
        b.write(". ")
    b.write(_sentence(draw))
    return b.finish().document


@st.composite
def cut_settings(draw) -> ChunkSettings:
    strategy = draw(st.sampled_from(["fixed", "sentence", "recursive"]))
    max_tokens = draw(st.integers(min_value=8, max_value=64))
    # ChunkSettings rejects a nonzero overlap for strategies that would ignore it.
    overlap_tokens = (
        draw(st.integers(min_value=0, max_value=max_tokens - 1)) if strategy == "fixed" else 0
    )
    return ChunkSettings(strategy=strategy, max_tokens=max_tokens, overlap_tokens=overlap_tokens)


@st.composite
def chunk_lists(draw, min_size: int = 2, max_size: int = 20) -> tuple[Chunk, ...]:
    """Chunks that each carry one of the fixed rerank query's real words, so a
    reranker scoring by word overlap always ranks every member of this list above
    anything worse_chunk_lists produces, whatever the rest of their text says."""
    count = draw(st.integers(min_value=min_size, max_value=max_size))
    names = draw(st.lists(_FILLER_WORD, min_size=count, max_size=count, unique=True))
    chunks = []
    for i, name in enumerate(names):
        keyword = draw(st.sampled_from(_RERANK_KEYWORDS))
        filler = draw(st.lists(_FILLER_WORD, min_size=1, max_size=6))
        text = " ".join([keyword, *filler])
        chunks.append(Chunk(
            name=ChunkId(f"frag.{name}"), document=DocName("gen.candidates"),
            ordinal=i, span=Range(0, len(text)), text=text,
        ))
    return tuple(chunks)


@st.composite
def worse_chunk_lists(draw, max_size: int = 8) -> tuple[Chunk, ...]:
    """Chunks that never contain a word the fixed rerank query holds, so a
    reranker scoring by word overlap always ranks them at the very bottom, strictly
    below any chunk chunk_lists produced."""
    count = draw(st.integers(min_value=0, max_value=max_size))
    names = draw(st.lists(_FILLER_WORD, min_size=count, max_size=count, unique=True))
    chunks = []
    for i, name in enumerate(names):
        filler = draw(st.lists(_FILLER_WORD, min_size=2, max_size=6))
        text = " ".join(filler)
        chunks.append(Chunk(
            name=ChunkId(f"pad.{name}"), document=DocName("gen.padding"),
            ordinal=i, span=Range(0, len(text)), text=text,
        ))
    return tuple(chunks)


@st.composite
def documents_with_a_short_value(draw) -> Written:
    """Long enough, at the settings this generator is paired with, that a fixed size
    chunker really produces more than one chunk, with one written value short enough
    that it could in principle be held whole by a single chunk."""
    name = DocName(f"gen.short.{draw(st.integers(min_value=0, max_value=10**9))}")
    b = _Builder(name)
    for _ in range(draw(st.integers(min_value=15, max_value=25))):
        b.write(_sentence(draw))
    value = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6))
    b.emit(value)
    b.write(". ")
    for _ in range(draw(st.integers(min_value=15, max_value=25))):
        b.write(_sentence(draw))
    return b.finish()


@st.composite
def ranked_lists(draw) -> SearchResult:
    """A search result the way a real producer would return one: unique names, ranks
    starting at one, hits already ordered by descending score then chunk name.
    Scores stay well inside the float range so the scale invariance property can
    multiply one of them by up to a thousand without approaching infinity."""
    names = draw(st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz.", min_size=3, max_size=12),
        min_size=1, max_size=10, unique=True,
    ))
    scores = draw(st.lists(
        st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=len(names), max_size=len(names),
    ))
    ordered = sorted(zip(names, scores), key=lambda pair: (-pair[1], pair[0]))
    hits = tuple(
        Hit(chunk=ChunkId(name), score=score, rank=rank)
        for rank, (name, score) in enumerate(ordered, start=1)
    )
    arm = draw(st.sampled_from(["vector", "keyword"]))
    query = draw(st.text(min_size=1, max_size=10))
    return SearchResult(query=query, arm=arm, hits=hits)


def build_producer(kind: str, chunks, matrix):
    """Hides each search producer's own signature behind one shape the property calls
    the same way regardless of which producer it is given. One branch today; later
    pieces in this part add to this same adapter rather than starting a second one."""
    if kind == "vector":
        index = VectorIndex.build(chunks, matrix)

        class _VectorProducer:
            def search(self, query_text, query_vector, candidates):
                return index.search(query_text, query_vector, candidates)

        return _VectorProducer()
    if kind == "keyword":
        keyword_index = build_keyword_index(chunks, SparseSettings())

        class _KeywordProducer:
            def search(self, query_text, query_vector, candidates):
                return keyword_index.search(query_text, candidates)

        return _KeywordProducer()
    if kind == "rerank":
        class _RerankProducer:
            def search(self, query_text, query_vector, candidates):
                reranker = StubReranker(depth_in=len(chunks), depth_out=candidates)
                return reranker(query_text, chunks)

        return _RerankProducer()
    if kind == "fuse":
        vector_index = VectorIndex.build(chunks, matrix)
        keyword_index = build_keyword_index(chunks, SparseSettings())

        class _BlendProducer:
            def search(self, query_text, query_vector, candidates):
                vector_result = vector_index.search(query_text, query_vector, candidates)
                keyword_result = keyword_index.search(query_text, candidates)
                return rrf_fuse(vector_result, keyword_result, RrfSettings())

        return _BlendProducer()
    raise ValueError(f"no producer registered for {kind!r}")
