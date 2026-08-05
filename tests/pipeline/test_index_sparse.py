from __future__ import annotations

import hashlib
import math

import pytest

from atlas.config import SparseSettings
from atlas.contracts import Chunk, ChunkId, DocName, Range
from atlas.retrieval.sparse import (
    STOPWORDS_DIGEST,
    STOPWORDS_PATH,
    Bm25Index,
    build_keyword_index,
    stopword_set,
    tokenise,
)


def _chunk(name: str, doc: str, text: str) -> Chunk:
    return Chunk(name=ChunkId(name), document=DocName(doc), ordinal=0, span=Range(0, len(text)), text=text)


def test_bm25_scores_match_a_hand_computed_okapi_formula():
    # The only numeric pin on a BM25 score in the suite: a mutant dropping the 0.5
    # smoothing epsilon from the idf numerator survived everything else. The formula
    # below is written from the Okapi definition, not from atlas.retrieval.sparse.
    chunks = (_chunk("a", "docA", "alpha beta"), _chunk("b", "docB", "alpha alpha gamma"))
    settings = SparseSettings(scorer="bm25", k1=1.5, b=0.75)
    index = build_keyword_index(chunks, settings)
    scores = {hit.chunk: hit.score for hit in index.search("alpha beta", candidates=10).hits}

    total, avg_length = 2, (2 + 3) / 2
    k1, b = settings.k1, settings.b
    idf_alpha = math.log(1 + (total - 2 + 0.5) / (2 + 0.5))  # in both chunks: df=2
    idf_beta = math.log(1 + (total - 1 + 0.5) / (1 + 0.5))   # in one chunk: df=1
    norm_a = k1 * (1 - b + b * 2 / avg_length)
    norm_b = k1 * (1 - b + b * 3 / avg_length)
    expected_a = idf_alpha * 1 * (k1 + 1) / (1 + norm_a) + idf_beta * 1 * (k1 + 1) / (1 + norm_a)
    expected_b = idf_alpha * 2 * (k1 + 1) / (2 + norm_b)  # beta never appears in docB

    assert scores[ChunkId("a")] == pytest.approx(expected_a, rel=1e-9)
    assert scores[ChunkId("b")] == pytest.approx(expected_b, rel=1e-9)
    assert scores[ChunkId("a")] > scores[ChunkId("b")], "docA holds both query terms and should rank first"


@pytest.mark.parametrize("token", ["FIB-500-X", "v2.10.1", "SE1-7TP"])
def test_a_code_a_version_string_and_a_postal_code_survive_as_single_terms(token):
    assert tokenise(f"plan {token} applies") == ["plan", token.lower(), "applies"]


def test_the_keyword_index_uses_the_snapshotted_stopword_list(tie_heavy):
    # The digest test below stays green for an index that ignores the file and takes the
    # library's bundled list, which is the drift the snapshot exists to prevent.
    listed = frozenset(STOPWORDS_PATH.read_text(encoding="utf-8").split())
    assert listed, "the snapshotted list parsed to nothing"
    assert frozenset(tokenise(" ".join(sorted(listed)))) == frozenset()


def test_the_stopword_list_matches_its_recorded_digest():
    # Bytes, not decoded text. Hashing text read in text mode makes the digest depend
    # on how the platform translates line endings, and the check then goes red on a
    # contributor's machine for a reason that has nothing to do with the word list.
    digest = hashlib.blake2s(STOPWORDS_PATH.read_bytes(), digest_size=8).hexdigest()
    assert digest == STOPWORDS_DIGEST


def test_the_stopword_setting_reaches_both_the_index_and_its_queries() -> None:
    # Dropping the word list costs 0.1562 nDCG@10 on the keyword arm. A setting that
    # reached `build` but not `search` would be worse than none, because the index and
    # the query would disagree about what a word is.
    stops = stopword_set("default")
    assert "the" in stops, "the shipped list is not the shipped list"

    chunks = [_chunk("a#0", "a", "the dispute is the thing"), _chunk("b#0", "b", "dispute")]
    kept = Bm25Index.build(chunks, SparseSettings(stopwords="none"))
    dropped = Bm25Index.build(chunks, SparseSettings(stopwords="default"))
    assert "the" in kept.idf, "stopwords='none' still dropped a stopword at build"
    assert "the" not in dropped.idf, "the default list did not reach build"
    assert kept.search("the the the", 2).hits[0].score > 0.0
    assert dropped.search("the the the", 2).hits[0].score == 0.0


def test_the_query_term_frequency_setting_changes_the_score_and_not_the_index() -> None:
    # `linear` counts a query term once per occurrence, which is not what Okapi specifies
    # and is what every recorded number was measured under. It is a property of the query
    # loop alone, so the two settings must build identical indexes and score differently.
    chunks = [_chunk("a#0", "a", "dispute dispute dispute"), _chunk("b#0", "b", "dispute charge")]
    linear = Bm25Index.build(chunks, SparseSettings(query_term_frequency="linear"))
    distinct = Bm25Index.build(chunks, SparseSettings(query_term_frequency="distinct"))
    assert linear.idf == distinct.idf and linear.length_norms == distinct.length_norms

    repeated = "dispute dispute dispute dispute"
    assert linear.search(repeated, 2).hits[0].score > distinct.search(repeated, 2).hits[0].score
    # The two definitions agree on a query with no repeats.
    once = "dispute"
    assert linear.search(once, 2).hits[0].score == distinct.search(once, 2).hits[0].score
