"""The keyword arm: BM25, and the `SetOverlapIndex` baseline it replaced. Both stay
because a runnable baseline is the only kind that stays honest. `SparseSettings.stopwords`
and `.query_term_frequency` are measured, non-default-sensitive levers here; see
docs/the-keyword-search.md for the ablation numbers."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from atlas.config import SparseSettings
from atlas.contracts import Chunk, ChunkId, SearchResult
from atlas.retrieval.ranking import name_order, ranked_hits

# parents[1] is the atlas package root: the word list is package data, not owned by
# the retrieval subpackage, so this must resolve independent of the working directory.
STOPWORDS_PATH = Path(__file__).resolve().parents[1] / "data" / "stopwords.txt"
STOPWORDS_DIGEST = "7901f79acfc830b6"

# Keeps a product code, version string, or postal code in one token: letters/digits,
# optionally continued across a hyphen or dot.
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-.][a-zA-Z0-9]+)*")


_NO_STOPWORDS = frozenset[str]()


@lru_cache(maxsize=1)
def _load_stopwords() -> frozenset[str]:
    """Read once per process; `tokenise` runs for every chunk and every query."""
    return frozenset(STOPWORDS_PATH.read_text(encoding="utf-8").split())


def stopword_set(setting: str = "default") -> frozenset[str]:
    """The word list a `SparseSettings.stopwords` value names. Takes a named setting
    rather than a path so the word list's contents, not just a filename, are covered by
    `run_id`; `STOPWORDS_DIGEST` pins what "default" means."""
    if setting == "none":
        return _NO_STOPWORDS
    if setting == "default":
        return _load_stopwords()
    raise ValueError(f"no stopword list named {setting!r}; known: 'default', 'none'")


def tokenise(text: str, stopwords: frozenset[str] | None = None) -> list[str]:
    """Lowercased alphanumeric runs, minus the stopword list. Takes the resolved set
    rather than a setting name so `search` isn't re-resolving a word list per query."""
    drop = _load_stopwords() if stopwords is None else stopwords
    return [t for t in (m.lower() for m in _TOKEN_PATTERN.findall(text)) if t not in drop]


@dataclass(frozen=True, slots=True)
class SetOverlapIndex:
    """How many of the query's distinct words a chunk contains, and nothing else: no
    term frequency, no inverse document frequency, no length normalisation."""

    chunks: tuple[ChunkId, ...]
    token_sets: tuple[frozenset[str], ...]
    # Carried rather than re-resolved per search, so a query is tokenised by the same
    # rule the chunks were.
    stopwords: frozenset[str] = _NO_STOPWORDS

    @classmethod
    def build(cls, chunks: Sequence[Chunk], settings: SparseSettings | None = None) -> SetOverlapIndex:
        stopwords = stopword_set((settings or SparseSettings()).stopwords)
        order = name_order(chunks)
        return cls(
            chunks=tuple(chunks[i].name for i in order),
            token_sets=tuple(frozenset(tokenise(chunks[i].text, stopwords)) for i in order),
            stopwords=stopwords,
        )

    def search(self, query_text: str, candidates: int) -> SearchResult:
        query_tokens = frozenset(tokenise(query_text, self.stopwords))
        scores = (float(len(query_tokens & tokens)) for tokens in self.token_sets)
        return SearchResult(
            query=query_text, arm="keyword",
            hits=ranked_hits(zip(self.chunks, scores, strict=True), candidates),
        )


@dataclass(frozen=True, slots=True)
class Bm25Index:
    """Okapi BM25, written out rather than imported: inverse document frequency down-
    weights common terms, `k1` saturates repeated term frequency, `b` normalises for
    chunk length against the collection average. idf reads the whole collection, so a
    chunk's score depends on what else is in the corpus. See docs/the-keyword-search.md.
    """

    chunks: tuple[ChunkId, ...]
    frequencies: tuple[Counter[str], ...]
    # k1 * (1 - b + b * length / average_length), per chunk. Depends on nothing a query
    # supplies, so it's paid once at build rather than per search.
    length_norms: tuple[float, ...]
    idf: dict[str, float]
    k1: float
    stopwords: frozenset[str] = _NO_STOPWORDS
    query_term_frequency: str = "linear"

    @classmethod
    def build(cls, chunks: Sequence[Chunk], settings: SparseSettings) -> Bm25Index:
        stopwords = stopword_set(settings.stopwords)
        order = name_order(chunks)
        tokenised = [tokenise(chunks[i].text, stopwords) for i in order]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenised:
            document_frequency.update(set(tokens))

        total = len(tokenised) or 1
        lengths = [len(tokens) for tokens in tokenised]
        average_length = (sum(lengths) / total) or 1.0
        k1, b = settings.k1, settings.b
        return cls(
            chunks=tuple(chunks[i].name for i in order),
            frequencies=tuple(Counter(tokens) for tokens in tokenised),
            length_norms=tuple(k1 * (1 - b + b * length / average_length) for length in lengths),
            idf={
                term: math.log(1 + (total - count + 0.5) / (count + 0.5))
                for term, count in document_frequency.items()
            },
            k1=k1,
            stopwords=stopwords,
            query_term_frequency=settings.query_term_frequency,
        )

    def search(self, query_text: str, candidates: int) -> SearchResult:
        # `linear` (default) sums over every query term occurrence, unsaturated, rather
        # than Okapi's canonical *distinct*-terms-only or k3-saturated variants. This
        # matters here because tau2 queries are long and repeat terms; see
        # docs/the-keyword-search.md for the measured effect of switching to `distinct`.
        query_tokens = tokenise(query_text, self.stopwords)
        if self.query_term_frequency == "distinct":
            query_tokens = list(dict.fromkeys(query_tokens))
        scores = [
            sum(
                self.idf[term] * count * (self.k1 + 1) / (count + norm)
                for term in query_tokens
                if (count := frequency.get(term, 0))
            )
            for frequency, norm in zip(self.frequencies, self.length_norms, strict=True)
        ]
        return SearchResult(
            query=query_text, arm="keyword",
            hits=ranked_hits(zip(self.chunks, scores, strict=True), candidates),
        )


KeywordIndex = Bm25Index | SetOverlapIndex


def build_keyword_index(chunks: Sequence[Chunk], settings: SparseSettings) -> KeywordIndex:
    """`scorer="overlap"` selects the baseline BM25 is measured against."""
    if settings.scorer == "overlap":
        return SetOverlapIndex.build(chunks, settings)
    return Bm25Index.build(chunks, settings)
