"""Stage 1: embedders on retrieval only metrics (recall@k, nDCG@k, no LLM anywhere). The two real
embedder axes SP9 task 3 pinned in `models.lock` (`bge-m3` local, `text-embedding-3-small` openai --
the documented narrowness: no Voyage key, so the embedder matrix collapses to exactly these two cells,
per the digest's own instruction to name a narrowness rather than pad it) PLUS the two named baseline
rows D8/research 14 both call for: BM25 (lexical, no reranker) and `exact_scan` (the recall ground
truth row, `atlas.domain.retrieval.RetrievalConfig.exact_scan` already built).

Every component is a `search(case) -> Sequence[Chunk]` callable, best result first. Hermetically these are
seeded fixture callables (a small, committed table standing in for what a live embedder + retriever
call would return -- this task's own "fully hermetic vs seeded REPLAY fixtures" property); a live
caller (deferred to the batched live capture session) wires the SAME shape to
`atlas.adapters.pgvector_retriever.PgvectorRetriever.search_chunks` bound to a real `EmbeddingClient`
per axis instead, with NO change to this module's own contract.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from quality.retrieval_report import CaseRetrieval, RetrievalReport, evaluate

from atlas.ports.knowledge import Chunk

from matrix.cache import MatrixCache, cell_key
from matrix.cases import MatrixCase
from matrix.chunks import deserialize_chunk, serialize_chunk

#: The two named baseline rows every table carries (D8/research 14), never omitted from a stage 1
#: run: BM25 plus no reranker (the lexical floor) and exact_scan (the recall ground truth row, HNSW
#: bypassed). Named here as constants so a caller/test can assert their presence without hand typing
#: the string twice.
BM25_COMPONENT_ID = "bm25-no-reranker"
EXACT_SCAN_COMPONENT_ID = "exact-scan"
BASELINE_COMPONENT_IDS = frozenset({BM25_COMPONENT_ID, EXACT_SCAN_COMPONENT_ID})


@dataclass(frozen=True)
class EmbedderComponent:
    """`embedding_model` is `{"id":..., "revision":...}` (the manifest contract's own shape) for a
    component with a real pinned embedder, or `None` for one that has none (the BM25 lexical
    baseline: nothing to embed, a tsvector rank is not a vector search at all). `index_build_id` is
    the caller's own real, content addressed build id (`rag_tools.fingerprint.index_build_id`) when
    a real index backs this component; `None` in every hermetic test (no index is ever actually
    built here), honestly rendered as `matrix.lineage.NOT_APPLICABLE` rather than a fabricated id."""

    component_id: str
    search: Callable[[MatrixCase], Sequence[Chunk]]
    embedding_model: dict | None
    is_baseline: bool = False
    index_build_id: str | None = None


@dataclass(frozen=True)
class EmbedderCell:
    component_id: str
    embedding_model: dict | None
    is_baseline: bool
    candidates: dict[str, tuple[dict, ...]]  # case_id -> ranked, serialized Chunk dicts (top k)
    report: RetrievalReport
    index_build_id: str | None = None


def run_embedder_stage(
    cases: Sequence[MatrixCase],
    embedders: Sequence[EmbedderComponent],
    *,
    k: int,
    seed: int,
    cache: MatrixCache,
    corpus_version: str,
    dataset_version: str,
    pool_size: int | None = None,
) -> dict[str, EmbedderCell]:
    """One `EmbedderCell` per component, in the SAME order `embedders` was given (a dict preserves
    insertion order; iteration is always over the caller's own explicit sequence, never a set).

    `k` is the retrieval only METRIC truncation (this stage's own `recall@k`/`nDCG@k`); `pool_size`
    (defaults to `k`, unchanged behaviour for a caller with no stage 2 downstream) is how many
    candidates are actually CACHED per case. The two are deliberately separate: stage 2's own depth
    sweep (`{20, 50, 100}`) needs a candidate pool at least as wide as its widest depth to sweep
    over at all, while stage 1's own reported metric stays at whatever `k` the caller actually wants
    scored (e.g. the production width, `atlas.domain.retrieval.K_FINAL` -- single sourced there
    rather than restated as a literal here, the exact drift this docstring used to cause when it
    said "k=5" while the real deployed width was 3). A caller with no stage 2 (or a hermetic test
    with no need for headroom) simply omits `pool_size` and gets the prior, single-`k` behaviour.

    Content hash cached per component at `{"stage": "embedder", "k": k, "pool_size": pool_size}`: a
    rerun over an unchanged `(corpus_version, dataset_version, component_id, k, pool_size)` never
    calls `search` again.
    """
    pool = pool_size if pool_size is not None else k
    cells: dict[str, EmbedderCell] = {}
    for embedder in embedders:
        key = cell_key(
            corpus_version=corpus_version,
            dataset_version=dataset_version,
            component_id=embedder.component_id,
            params={"stage": "embedder", "k": k, "pool_size": pool},
        )

        def _compute() -> dict:
            return {case.case_id: [serialize_chunk(c) for c in embedder.search(case)[:pool]] for case in cases}

        cached = cache.get_or_compute(key, _compute)
        candidates = {case_id: tuple(chunks) for case_id, chunks in cached.items()}
        case_retrievals = [
            CaseRetrieval(
                case.case_id,
                tuple(c["doc_id"] for c in candidates.get(case.case_id, ())[:k]),
                case.relevant_doc_ids,
            )
            for case in cases
        ]
        report = evaluate(case_retrievals, k=k, seed=seed)
        cells[embedder.component_id] = EmbedderCell(
            component_id=embedder.component_id,
            embedding_model=embedder.embedding_model,
            is_baseline=embedder.is_baseline,
            candidates=candidates,
            report=report,
            index_build_id=embedder.index_build_id,
        )
    return cells


def candidate_chunks(cell: EmbedderCell, case_id: str) -> list[Chunk]:
    """The stage's own cached candidates for one case, rehydrated back into `Chunk` objects (the
    shape stage 2's reranker needs). Missing case_id -> `[]`, never a `KeyError`: a case a live
    corpus genuinely returned nothing for is a defined (empty) result, not a crash."""
    return [deserialize_chunk(d) for d in cell.candidates.get(case_id, ())]


__all__ = [
    "BASELINE_COMPONENT_IDS",
    "BM25_COMPONENT_ID",
    "EXACT_SCAN_COMPONENT_ID",
    "EmbedderCell",
    "EmbedderComponent",
    "candidate_chunks",
    "run_embedder_stage",
]
