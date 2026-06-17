"""Retrieval domain (D8): pure, no framework/client imports. `RetrievalConfig` is the one typed
knob-set every retrieval call carries; `rrf_fuse` is the ~30-line reciprocal rank fusion math the
HLD names as app code, not a database extension, so it is testable with synthetic rank lists alone.
Both are the load-bearing shapes behind the `search_chunks(query, k, config)` port boundary
(`atlas/ports/knowledge.py`); this module knows nothing about Postgres, TEI, or the in memory
adapter, only the score arithmetic and the config shape.

`l2_normalize` and `vector_literal` live here for the same reason: pure arithmetic and pure string
formatting that the index builder (`rag_tools.ingest`) and both hybrid search adapters
(`adapters/pgvector_retriever`, `matrix.live_search`) all need. They used to be three byte identical
private copies, each justified by a docstring citing a "backend must never import harness code"
rule that the import lint did not actually enforce. The lint now enforces it, and the legal
direction (harness importing backend) carries these instead.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


# The RRF rank constant (Cormack et al. 2009). Declared once, here, as `rrf_fuse`'s own default:
# it used to be re declared as a `RetrievalConfig` field threaded to exactly one call site that
# always passed it, plus a second copy in a duplicate fusion implementation. A study of a different
# k is an explicit `rrf_fuse(rankings, k=...)` at the call that wants it, not a config knob every
# caller carries and none varies.
RRF_K = 60

# The deployed retrieval widths, declared ONCE here because four modules need them and none of them
# may own them: `mcp_servers/knowledge_server.py` (whose `DEPLOYED_K` is now `K_FINAL`, not a second
# literal), and the three RAG variants `orchestration/agentic_rag.py`, `orchestration/graph_rag.py`
# and `matrix/variants.py`. Each of those three used to declare its own `_K_FUSED = 20` / `_K_FINAL
# = 3` pair under a comment asserting it matched the others by hand; nothing checked the assertion,
# and `matrix/embedders.py` had already drifted (calling k=5 "the production DEPLOYED_K"). A pure
# domain constant is importable from every one of them without dragging an MCP or LangGraph module
# into a peer, which was the stated reason for not importing in the first place.
#
# `K_FUSED` is the wide candidate pool retrieval asks for before reranking; `K_FINAL` is what
# survives the reranker's truncation and what the naive path returns.
K_FUSED = 20
K_FINAL = 3


@dataclass(frozen=True)
class RetrievalConfig:
    """Typed retrieval knobs threaded through the one `search_chunks` boundary. `k_fused` is how
    many candidates each backend ranking contributes before fusion; `k_final` is what survives
    after (optional) reranking. `exact_scan` is the recall ground-truth mode (HNSW bypassed);
    `ef_search` is the HNSW query-time knob, a no-op for adapters (like the in memory one) that
    have no HNSW index. `lexical_only` (SP4 task 4, the degradation ladder's embedding down rung)
    drops the vector arm, AND the TEI embed call that would have produced its query vector,
    entirely, keeping only the tsvector lexical arm: an adapter with no embedding service left to
    call can still answer from lexical search alone rather than failing the whole turn."""

    k_fused: int = 50
    k_final: int = 5
    rerank_enabled: bool = True
    exact_scan: bool = False
    ef_search: int = 40
    lexical_only: bool = False


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion over one or more ranked id lists (D8): `score(d) = sum over rankings
    of 1 / (k + rank_d)`, rank is 1-based; a ranking an id is absent from contributes nothing to
    its score. Pure score arithmetic, no I/O, no randomness.

    Deterministic even under a tie: sorted by score descending, then by id ascending, so the
    result never depends on dict iteration order or which ranking listed an id first. The score
    set depends only on each id's (ranking, rank) memberships, so permuting the order of the
    `rankings` sequence itself never changes the result (summation over rankings commutes).

    `k` must be positive: `k = 0` makes the top ranked document's contribution 1.0 (RRF's whole
    point is damping it), and a negative `k` divides by zero at rank `-k`. Rejected loudly rather
    than silently producing a ranking nobody can reason about.
    """
    if k < 1:
        raise ValueError(f"rrf_fuse k must be >= 1, got {k}")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Normalize one vector to unit length. A zero vector (degenerate, should not occur for a real
    embedding) is returned unchanged rather than dividing by zero.

    Pure arithmetic over stdlib `math`, so it belongs beside `rrf_fuse` rather than being restated
    per caller: the index builder normalizes document vectors at ingest, and both hybrid search
    adapters normalize the query vector at search time. All three used to carry byte identical
    private copies, justified by an import rule that did not actually exist (see
    `testing/tests/test_import_lint.py`, now derived); the legal direction, harness importing
    backend, was always open."""
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return list(vector)
    return [component / norm for component in vector]


def vector_literal(values: Sequence[float]) -> str:
    """A pgvector text literal (`[0.1,0.2,...]`), bound as an ordinary string `%s` parameter and
    cast with `::vector` in the SQL text. Avoids an extra `pgvector`-python adapter dependency:
    Postgres itself parses the literal against the column's `vector` type. Pure string formatting,
    no client import, so it sits here rather than in each of the three modules that need it."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"
