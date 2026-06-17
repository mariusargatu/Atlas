"""Graph retrieval domain math (SP9 task 2): pure, no framework/client import, alongside
`domain/retrieval.py`'s `rrf_fuse`. This module knows nothing about Postgres, Neo4j, or the
`KnowledgeGraph` port itself; it holds only the two steps of the graph-RAG pipeline that are plain
data transforms once the I/O (entity resolution, hop expansion) has already happened elsewhere
(`orchestration/graph_rag.py`, which calls the `KnowledgeGraph` adapter's `resolve`/`neighbors`):

  `extract_candidate_mentions`  the query's own candidate surface forms to try resolving against
                                the graph (pure tokenisation, no lookup: a candidate that resolves to
                                nothing is simply not an entity, never an error).
  `collect_chunks_by_entities`  the join from traversal to chunks: which of an already retrieved
                                candidate pool actually attaches to the entity ids traversal reached,
                                via `Chunk.entity_ids` overlap. The SAME set arithmetic
                                `testing.harness.quality.agent_metrics.citation_precision_recall`
                                already applies to a whole response's citations, applied here to a
                                candidate chunk pool instead (this module cannot import that harness
                                module: `testing/tests/test_import_lint.py`'s product/harness
                                boundary is one way, harness may import backend, never the reverse).

Both functions are the pure, downstream half of an I/O call, exactly the role `rrf_fuse` plays for
the two SQL ranking arms in `pgvector_retriever.py`: the adapter/orchestration layer does the network
or database round trip, this module does the deterministic arithmetic on what came back.
"""
from __future__ import annotations

from collections.abc import Sequence

from atlas.ports.knowledge import Chunk

# A word must survive this stripping to count as part of a candidate mention; matches the
# `agentic_rag._STRIP_CHARS` convention (trailing/leading punctuation never blocks a real match).
_STRIP_CHARS = ".,!?;:\"'()"

# The longest candidate mention (in words) this module will try to resolve. Registry entity names on
# this corpus top out at three words ("Equipment Rental Fee", "Early Termination Fee"), so four is a
# safety margin, not a guess; a real deployment with longer entity names would raise this constant,
# never change the algorithm.
_MAX_MENTION_WORDS = 4


def extract_candidate_mentions(query: str, max_words: int = _MAX_MENTION_WORDS) -> tuple[str, ...]:
    """Every contiguous word span of the query, length 1 up to `max_words`, punctuation stripped,
    in a stable order (shortest spans first, left to right within a length), deduplicated. This is
    deliberately overly inclusive: a candidate that is not a real entity surface form simply fails
    `KnowledgeGraph.resolve` (returns `None`) at the caller, never raises here. Pure tokenisation, no
    lookup, so the same query always yields the same candidate tuple regardless of what graph it is
    later tried against."""
    words = [w.strip(_STRIP_CHARS) for w in query.split()]
    words = [w for w in words if w]
    seen: dict[str, None] = {}
    for n in range(1, max_words + 1):
        for start in range(len(words) - n + 1):
            phrase = " ".join(words[start : start + n])
            seen.setdefault(phrase, None)
    return tuple(seen)


def collect_chunks_by_entities(chunks: Sequence[Chunk], entity_ids: frozenset[str]) -> list[Chunk]:
    """The chunks (from `chunks`, in their given order -- never reordered here, that is `rerank`'s
    job) whose `entity_ids` overlap `entity_ids` at all. `entity_ids` empty (no entity resolved, or
    traversal reached nothing) returns `[]` rather than raising: an empty join is a defined, guarded
    result, the same convention `quality.agent_metrics`'s own module docstring documents for every
    metric here ("where a metric's denominator is empty ... it returns 0.0, never a
    ZeroDivisionError"). The caller (`orchestration/graph_rag.py`) decides what an empty join means
    for the turn (there, falling back to the unjoined candidate pool rather than answering from
    nothing); this function only reports the join itself."""
    if not entity_ids:
        return []
    return [c for c in chunks if frozenset(c.entity_ids) & entity_ids]
