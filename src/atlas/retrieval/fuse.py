from __future__ import annotations

from atlas.config import RrfSettings
from atlas.contracts import Arm, ChunkId, FusedResult, SearchResult
from atlas.retrieval.ranking import ranked_hits


def rrf_fuse(vector: SearchResult, keyword: SearchResult, settings: RrfSettings) -> FusedResult:
    """Reciprocal rank fusion: score each chunk by the sum of `weight / (k + rank)` over
    the arms that found it, and rank by that. Fuses on rank rather than raw score because
    a cosine similarity and an integer overlap count aren't on the same scale."""
    weights: dict[Arm, float] = {
        "vector": settings.vector_weight, "keyword": settings.keyword_weight,
    }
    ranks_by_arm: dict[Arm, dict[ChunkId, int]] = {
        "vector": {hit.chunk: hit.rank for hit in vector.hits},
        "keyword": {hit.chunk: hit.rank for hit in keyword.hits},
    }

    arm_ranks: dict[ChunkId, dict[Arm, int]] = {}
    scores: dict[ChunkId, float] = {}
    # dict.fromkeys instead of set union: a set's iteration order depends on the
    # interpreter's string hash seed, which would make this walk nondeterministic
    # across processes.
    for name in dict.fromkeys((*ranks_by_arm["vector"], *ranks_by_arm["keyword"])):
        found = {
            arm: rank
            for arm, ranks in ranks_by_arm.items()
            if (rank := ranks.get(name)) is not None
        }
        arm_ranks[name] = found
        scores[name] = sum(weights[arm] / (settings.constant + rank) for arm, rank in found.items())

    return FusedResult(query=vector.query, hits=ranked_hits(scores.items()), arm_ranks=arm_ranks)
