"""Stage 2: rerankers over STAGE 1's cached candidate lists, at depths `{20, 50, 100}` (research 14's
own "Drowning in Documents" citation: reranker quality can degrade past a depth, so depth is a swept
variable, never a fixed constant). Today's axis is thin, `{BGE reranker v2 m3, none}` -- no Voyage
`rerank-2.5-lite` key, the documented narrowness, never padded to look wider than it is.

Still no LLM anywhere in this stage: `atlas.ports.reranker.Reranker.rerank` is a deterministic
cross encoder score sort (or, for the "none" axis, an identity operation with no effect), never a
model call. This stage crosses embedder x reranker x depth freely (all three are cheap and
deterministic); the expensive, LLM backed axis (stage 3's generators) is the one this whole staged
design keeps OFF a full cross product, per D17.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from quality.retrieval_report import CaseRetrieval, RetrievalReport, evaluate

from atlas.ports.knowledge import Chunk
from atlas.ports.reranker import Reranker

from matrix.cache import MatrixCache, cell_key
from matrix.cases import MatrixCase
from matrix.chunks import deserialize_chunk, serialize_chunk
from matrix.embedders import EmbedderCell

#: Research 14's own swept depths: how many of stage 1's fused candidates the reranker gets to work
#: with before truncating to `k_final`. Kept as a module constant (never hand typed per call site)
#: so a stat over "every depth this run swept" reads it once, here.
DEPTHS: tuple[int, ...] = (20, 50, 100)

#: The "none" axis's own component id (the reranker matrix's other half, alongside BGE): an identity
#: pass through, never omitted from a table the way a silently skipped axis would be.
NONE_RERANKER_ID = "none"


@dataclass(frozen=True)
class RerankerComponent:
    """`reranker` is `None` for the `NONE_RERANKER_ID` axis (identity pass through: the candidates
    already in fused order survive, truncated to `k_final`, exactly as if no rerank step ran)."""

    component_id: str
    reranker: Reranker | None


@dataclass(frozen=True)
class RerankerCell:
    embedder_component_id: str
    reranker_component_id: str
    depth: int
    candidates: dict[str, tuple[dict, ...]]
    report: RetrievalReport


def config_id(embedder_component_id: str, reranker_component_id: str, depth: int) -> str:
    """The one retrieval config identifier stage 3 (and the manifest) names a stage 2 cell by, built
    once here so it can never drift between the writer and a reader."""
    return f"{embedder_component_id}::{reranker_component_id}@{depth}"


def run_reranker_stage(
    cases: Sequence[MatrixCase],
    embedder_cells: dict[str, EmbedderCell],
    rerankers: Sequence[RerankerComponent],
    *,
    k_final: int,
    seed: int,
    cache: MatrixCache,
    corpus_version: str,
    dataset_version: str,
    depths: tuple[int, ...] = DEPTHS,
) -> dict[str, RerankerCell]:
    """One `RerankerCell` per (embedder, reranker, depth) triple, keyed by `config_id(...)`. Every
    loop is over an explicit caller supplied sequence (`embedder_cells.items()` walks in the
    dict's own insertion order, which stage 1 built from ITS caller's own explicit `embedders`
    sequence; `rerankers`/`depths` are walked in the caller's/module's own fixed order), never
    resorted through an unordered set."""
    cases_by_id = {c.case_id: c for c in cases}
    cells: dict[str, RerankerCell] = {}
    for embedder_id, embedder_cell in embedder_cells.items():
        for reranker in rerankers:
            for depth in depths:
                cid = config_id(embedder_id, reranker.component_id, depth)
                key = cell_key(
                    corpus_version=corpus_version,
                    dataset_version=dataset_version,
                    component_id=cid,
                    params={"stage": "reranker", "depth": depth, "k_final": k_final},
                )

                def _compute(embedder_cell=embedder_cell, reranker=reranker, depth=depth) -> dict:
                    out: dict[str, list[dict]] = {}
                    for case_id, serialized in embedder_cell.candidates.items():
                        chunks: list[Chunk] = [deserialize_chunk(d) for d in serialized][:depth]
                        query = cases_by_id[case_id].query
                        reranked = chunks if reranker.reranker is None else reranker.reranker.rerank(query, chunks)
                        out[case_id] = [serialize_chunk(c) for c in reranked[:k_final]]
                    return out

                cached = cache.get_or_compute(key, _compute)
                candidates = {case_id: tuple(chunks) for case_id, chunks in cached.items()}
                case_retrievals = [
                    CaseRetrieval(
                        case.case_id,
                        tuple(c["doc_id"] for c in candidates.get(case.case_id, ())),
                        case.relevant_doc_ids,
                    )
                    for case in cases
                ]
                report = evaluate(case_retrievals, k=k_final, seed=seed)
                cells[cid] = RerankerCell(
                    embedder_component_id=embedder_id,
                    reranker_component_id=reranker.component_id,
                    depth=depth,
                    candidates=candidates,
                    report=report,
                )
    return cells


__all__ = [
    "DEPTHS",
    "NONE_RERANKER_ID",
    "RerankerCell",
    "RerankerComponent",
    "config_id",
    "run_reranker_stage",
]
