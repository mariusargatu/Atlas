"""Every modelling choice a run makes, in one tree that hashes to its run_id.

A field belongs here only if changing it changes what the run measures (a model name).
Operational values (keys, endpoints, timeouts) are read from the environment instead,
since everything in this file lands in run_id and prints next to every figure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


@dataclass(frozen=True, slots=True)
class ChunkSettings:
    strategy: Literal["fixed", "sentence", "recursive"] = "recursive"
    max_tokens: int = 256
    overlap_tokens: int = 0
    encoding: str = "cl100k_base"  # pinned; changing it moves every boundary

    def __post_init__(self) -> None:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be less than max_tokens "
                f"({self.max_tokens}), or a fixed size chunker never advances"
            )
        if self.overlap_tokens and self.strategy != "fixed":
            raise ValueError(
                f"overlap_tokens is only applied by the fixed size chunker; strategy "
                f"{self.strategy!r} packs whole leaf units and would ignore "
                f"{self.overlap_tokens}, while still moving run_id"
            )


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    """Which embedding model produces the vectors, and at what width. `size` is
    requested of the endpoint as `dimensions` rather than assumed, so the matrix width
    matches this number by construction."""

    model_name: str = "text-embedding-3-small"
    model_version: str = "2024-01"
    size: int = 256
    # `VectorIndex.search` divides by both norms (cosine similarity), so this cannot
    # change a ranking, but it stays in run_id and the cache key. Becomes load-bearing
    # if the dense arm ever moves to a dot-product index.
    normalise: bool = True


@dataclass(frozen=True, slots=True)
class SparseSettings:
    """Which keyword scorer, and the two constants BM25 is defined by.

    `scorer`, `stopwords`, and `query_term_frequency` are all measured, ablatable
    levers rather than implementation details, in that order of discovery rather than
    of size: `stopwords` and `query_term_frequency` moved nDCG@10 more than the
    scorer choice did, and were only made settings after that was measured. See
    docs/the-keyword-search.md for the numbers.
    """

    scorer: Literal["bm25", "overlap"] = "bm25"
    k1: float = 1.5  # how fast term frequency saturates; the literature's value
    b: float = 0.75  # how much chunk length is normalised away; the literature's value
    # Both default to what the recorded runs were measured under, so `Settings()` still
    # hashes the same; see `Settings._hashed_tree`.
    stopwords: Literal["default", "none"] = "default"
    # "linear" counts a query term once per occurrence (what `Bm25Index.search` has
    # always done); "distinct" is the textbook Okapi form. `data/settings/*.json` runs
    # each ablation.
    query_term_frequency: Literal["linear", "distinct"] = "linear"

    def __post_init__(self) -> None:
        # b above one can drive the length norm negative for a chunk shorter than the
        # collection average, producing a nonsensical BM25 score.
        if self.k1 <= 0.0:
            raise ValueError(
                f"k1 must be positive, got {self.k1}: term frequency saturation is undefined "
                "at or below zero"
            )
        if not (0.0 <= self.b <= 1.0):
            raise ValueError(
                f"b must be within [0, 1], got {self.b}: above one, length normalisation can "
                "drive a matching chunk's score negative"
            )


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """How many candidates each arm hands to fusion, before it cuts to the reranker's
    depth_in.

    Widening either number is not free of surprises: fused recall@10 is not monotonic
    in candidate depth. A chunk excluded by one arm's shallow cutoff can resurface once
    it widens and leapfrog a chunk that was correctly retrieved at a strong rank by a
    single arm. "More candidates can only help" is right for one chunk's own RRF score,
    wrong for the resulting ranking. See docs/the-reranker.md.
    """

    vector_candidates: int = 50
    keyword_candidates: int = 50

    def __post_init__(self) -> None:
        # A negative depth turns every `[:n]` in the pipeline from "keep n" into "drop
        # the last n"; zero hands the answer writer nothing while still billing a
        # generation.
        if self.vector_candidates < 1:
            raise ValueError(f"vector_candidates must be at least 1, not {self.vector_candidates}")
        if self.keyword_candidates < 1:
            raise ValueError(
                f"keyword_candidates must be at least 1, not {self.keyword_candidates}"
            )


@dataclass(frozen=True, slots=True)
class RrfSettings:
    """Reciprocal rank fusion. See `atlas.retrieval.fuse.rrf_fuse`."""

    constant: int = 60  # the k in RRF, from Cormack et al. 2009; 60 is the paper's value
    vector_weight: float = 1.0
    keyword_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class RerankSettings:
    # `backend="passthrough"` is how reranking is turned off; no separate enabled flag.
    # Passthrough is the default because every cross encoder measured on this corpus
    # ranked worse than not reranking. See docs/the-reranker.md.
    backend: Literal["onnx", "passthrough"] = "passthrough"
    model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    depth_in: int = 50
    depth_out: int = 10

    def __post_init__(self) -> None:
        if self.depth_out < 1 or self.depth_in < 1:
            raise ValueError(
                f"rerank depths must be at least 1, not depth_in={self.depth_in} "
                f"depth_out={self.depth_out}: a depth of zero reranks to an empty list and "
                "the pipeline still pays a model to answer from it"
            )
        if self.depth_out > self.depth_in:
            raise ValueError(
                f"depth_out={self.depth_out} exceeds depth_in={self.depth_in}, so the "
                "reranker is asked for more chunks than it was given"
            )


@dataclass(frozen=True, slots=True)
class AnswerSettings:
    model: str = "gpt-5.6-luna"
    # Inert on the default model (`gpt-5.6-luna` rejects `temperature`; see
    # `OpenAIStructuredClient`), but still honoured on models that accept it.
    # `OpenAIStructuredClient.randomness_applied` reports which happened.
    randomness: float = 0.0
    max_shown: int = 10
    citation_required: bool = True
    prompt_version: str = "4.0.0"  # must agree with data/prompts/answer.md.j2's front matter

    def __post_init__(self) -> None:
        # max_shown=0 bills a generation against zero passages, and the resulting refusal
        # is indistinguishable in the Record from an honest one.
        if self.max_shown < 1:
            raise ValueError(
                f"max_shown must be at least 1, not {self.max_shown}: the writer would be "
                "paid to answer from no passages, and a refusal under no evidence records "
                "identically to an honest one"
            )


@dataclass(frozen=True, slots=True)
class JudgeSettings:
    model: str = "gpt-5.6-luna"
    rubric_path: str = "data/rubric.md"
    rubric_version: str = "1.0.0"  # must agree with data/rubric.md's front matter
    prompt_path: str = "data/prompts/judge.md.j2"
    # Must agree with prompt_path's front matter; checked against a hash of the body
    # rather than trusting a manual bump.
    prompt_version: str = "3.0.0"
    # How many times `evals.calibration.noise_floor` re-judges the same answer. In run_id
    # deliberately: a floor measured over five repeats isn't comparable to one over one.
    repeats: int = 5


@dataclass(frozen=True, slots=True)
class Settings:
    """Every modelling choice one run makes, and nothing else. No `profile` field (see
    docs/why-every-run-uses-real-models.md) and no `collection_seed`: `Tau2Source.__call__`
    takes a seed to satisfy the protocol and ignores it, since the corpus is a fixed
    third-party artefact rather than something regenerated per run."""

    chunk: ChunkSettings = ChunkSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    search: SearchSettings = SearchSettings()
    sparse: SparseSettings = SparseSettings()
    fusion: RrfSettings = RrfSettings()
    rerank: RerankSettings = RerankSettings()
    answer: AnswerSettings = AnswerSettings()
    judge: JudgeSettings = JudgeSettings()

    def _hashed_tree(self) -> dict[str, Any]:
        """The settings tree as `run_id` hashes it, which is not quite the whole tree: a
        field named in `_ADDED_AFTER_THE_STORE` is dropped while it holds its default, so
        a tree not using it hashes exactly as before the field existed. This keeps the
        guarantee that matters (two runs differing in any setting never share an id)
        without forcing every new field to rename every already-recorded run.
        """
        tree: dict[str, Any] = asdict(self)
        for group, names in _ADDED_AFTER_THE_STORE.items():
            defaults = type(getattr(self, group))()
            for name in names:
                if tree[group][name] == getattr(defaults, name):
                    del tree[group][name]
        return tree

    @property
    def run_id(self) -> str:
        payload = json.dumps(self._hashed_tree(), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2s(payload.encode(), digest_size=6).hexdigest()


# Fields that arrived after data/results/runs.jsonl was first written. See
# `Settings._hashed_tree`. Adding an existing field here would silently merge runs that
# really did differ.
_ADDED_AFTER_THE_STORE: dict[str, frozenset[str]] = {
    "sparse": frozenset({"stopwords", "query_term_frequency"}),
}


# The groups Settings nests, by the key each occupies in a settings file.
_GROUPS: dict[str, type] = {
    "chunk": ChunkSettings,
    "embedding": EmbeddingSettings,
    "search": SearchSettings,
    "sparse": SparseSettings,
    "fusion": RrfSettings,
    "rerank": RerankSettings,
    "answer": AnswerSettings,
    "judge": JudgeSettings,
}


def _build[T: DataclassInstance](cls: type[T], payload: dict[str, Any]) -> T:
    """Constructs one settings group, rejecting keys the dataclass has no field for, so a
    typo in a settings file raises instead of silently applying the default."""
    known = {f.name for f in fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"{cls.__name__} has no field(s) {sorted(unknown)}")
    return cls(**payload)


def load_settings(path: str | Path) -> Settings:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    groups: dict[str, Any] = {
        key: _build(cls, payload[key]) for key, cls in _GROUPS.items() if key in payload
    }
    scalars = {k: v for k, v in payload.items() if k not in _GROUPS}
    return _build(Settings, {**scalars, **groups})
