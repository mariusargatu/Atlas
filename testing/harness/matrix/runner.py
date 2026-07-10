"""The staged benchmark matrix runner (SP9 task 4): embedders (stage 1) -> rerankers (stage 2) ->
generators (stage 3), staged per D17, never a cross product on the expensive (LLM backed) axis.
Assembles one run manifest (every cell's own D26 lineage row, `matrix.lineage.build_manifest_row`)
plus HELM style per query result files, wired to `quality.stats` (via `matrix.compare`, holm
corrected paired deltas with 95% CIs) and `quality.gate` (`gate_on_lower_bound` promotes the top
retrieval config into stage 3, the "any matrix regression check routes through this gate" the
planning digest names). `judge.panel.panel_vote` runs for real in stage 3 (`matrix.generators`),
its first caller anywhere in this repo.

Fully hermetic by construction: every embedder/reranker component is a deterministic, seeded fixture
callable and every generator/judge call is a `replay.gateway.GatewayChatModel` pinned to REPLAY mode
against committed cassettes (keyless, zero egress). Determinism end to end: given the same inputs
(cases, components, cache, config), two calls to `run_matrix` produce a byte identical
`manifest.json` -- every score comes from a pure function of its inputs plus a stamped seed, and the
manifest is serialized with sorted keys over sorted (never in dict order) lists.

SP9 task 5: an optional `spend_gate` (`matrix.spend_gate.SpendGate`) checks every generator cell's
own `estimated_usd` BEFORE it runs at all (never after); a cell that would exceed its provider's
remaining budget is SKIPPED entirely and logged into the manifest's own `dropped_cells` list
(`matrix.spend_gate.dropped_cell_for`), never a silent drop. `spend_gate` defaults to `None`
(no gating at all, every generator runs), so every hermetic caller from before this task -- REPLAY
mode, no live spend to gate -- keeps its own manifest shape, plus one new, always present, empty
`dropped_cells` key.

SP9 final review (finding I1): an optional `variants` (`matrix.variants.VariantsConfig`) runs the
naive vs agentic vs graph comparison stage (`matrix.variants.run_variant_comparison`) over the SAME
`cases` this call already threads through stages 1-3, adding one `variant_comparison` manifest key
(a sorted list of one row per variant, `None` when `variants` is omitted -- the same
"absent config, absent effect" shape `spend_gate=None` already established). This is the stage that
actually MEASURES `orchestration/agentic_rag.py` and `orchestration/graph_rag.py`, closing the gap
the review named: both subgraphs were real and unit tested, but nothing in this package ever invoked
either one. `run_variant_comparison` is async (both variants are compiled LangGraph subgraphs); this
still synchronous `run_matrix` bridges that with `asyncio.run`, paid only when a caller actually supplies
`variants`.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from quality import ir_metrics
from quality.gate import GateDecision, gate_on_lower_bound
from quality.retrieval_report import RetrievalReport

from matrix.cache import MatrixCache
from matrix.cases import MatrixCase
from matrix.compare import compare_components, delta_to_dict
from matrix.embedders import BASELINE_COMPONENT_IDS, EmbedderCell, EmbedderComponent, run_embedder_stage
from matrix.generators import (
    GenerationCell,
    GeneratorComponent,
    OffDiagonalCheck,
    off_diagonal_validation,
    run_generation_cell,
)
from matrix.spend_gate import DroppedCell, SpendGate, check_spend, dropped_cell_for, record_spend
from matrix.lineage import NOT_APPLICABLE, build_manifest_row
from matrix.rerankers import DEPTHS, RerankerCell, RerankerComponent, run_reranker_stage
from matrix.select import all_retrieval_configs, select_top_configs
from matrix.variants import VariantsConfig, run_variant_comparison, variant_comparison_rows


@dataclass(frozen=True)
class MatrixRunConfig:
    """Every field a `run_matrix` invocation needs beyond the components themselves. `seed` feeds
    every bootstrap/permutation call (`quality.stats` via `matrix.compare` and
    `quality.retrieval_report.evaluate`), so a fixed seed plus fixed inputs is what makes two runs
    byte identical, not an accident of small sample sizes."""

    run_id: str
    git_sha: str
    corpus_version: str
    dataset_version: str
    chunker_config_hash: str
    k_retrieval: int = 5
    seed: int = 20260721
    n_top_configs: int = 2
    reranker_depths: tuple[int, ...] = DEPTHS
    gate_threshold: float = 0.0
    gate_variance_budget: float = 1.0


def _gen_key(config_id: str, generator_component_id: str) -> str:
    """The one join key between a `(retrieval config, generator)` pair and its `GenerationCell`,
    also the JSON safe dict key/manifest row id (a tuple cannot be a JSON object key)."""
    return f"{config_id}||{generator_component_id}"


def _embedder_id_for_config(config_id: str) -> str:
    """A bare embedder config_id (stage 1) has no `::`; a reranker config_id
    (`matrix.rerankers.config_id`'s own `embedder::reranker@depth` shape) always does, with the
    embedder id as its first segment. Assumes an embedder's own `component_id` never itself contains
    `::` (a naming discipline every fixture and every real axis in this task honors)."""
    return config_id.split("::", 1)[0]


def _report_for_config(
    config_id: str, embedder_cells: dict[str, EmbedderCell], reranker_cells: dict[str, RerankerCell]
) -> RetrievalReport:
    if config_id in embedder_cells:
        return embedder_cells[config_id].report
    return reranker_cells[config_id].report


def _report_dict(report: RetrievalReport) -> dict:
    return {
        "n": report.n,
        "k": report.k,
        "hit_rate_at_k": report.hit_rate_at_k,
        "hit_rate_at_k_ci": list(report.hit_rate_at_k_ci),
        "recall_at_k_ci": list(report.recall_at_k_ci),
        "mrr_ci": list(report.mrr_ci),
        "ndcg_at_k_ci": list(report.ndcg_at_k_ci),
        "detectable_effect_ndcg": report.detectable_effect_ndcg,
    }


def _gate_dict(decision: GateDecision) -> dict:
    return {
        "verdict": decision.verdict.value,
        "reason": decision.reason,
        "lower_bound": decision.lower_bound,
        "width": decision.width,
        "threshold": decision.threshold,
        "variance_budget": decision.variance_budget,
    }


def _off_diagonal_dict(check: OffDiagonalCheck) -> dict:
    return {
        "primary_config_id": check.primary_config_id,
        "secondary_config_id": check.secondary_config_id,
        "shared_generator_id": check.shared_generator_id,
        "primary_mean_correctness": check.primary_mean_correctness,
        "secondary_mean_correctness": check.secondary_mean_correctness,
        "retrieval_ranking_holds": check.retrieval_ranking_holds,
    }


def _per_case_ndcg(candidates: dict[str, tuple[dict, ...]], cases: Sequence[MatrixCase], k: int) -> list[float]:
    """Per case nDCG@k, in `cases`' own order (never a dict/set reorder), so two components' score
    lists pair up index by index -- the invariant `matrix.compare.compare_components` depends on.
    Cheap, a pure recomputation from already cached candidates: no new retrieval call, no new cache
    entry, just `quality.ir_metrics.ndcg_at_k` applied again."""
    return [
        ir_metrics.ndcg_at_k(tuple(c["doc_id"] for c in candidates.get(case.case_id, ())), case.relevant_doc_ids, k)
        for case in cases
    ]


def _embedder_stage_row(cell: EmbedderCell, config: MatrixRunConfig) -> dict:
    return {
        "component_id": cell.component_id,
        "is_baseline": cell.is_baseline,
        "lineage": build_manifest_row(
            run_id=config.run_id,
            git_sha=config.git_sha,
            prompt_hash=NOT_APPLICABLE,  # retrieval only: no prompt/model call at this stage
            model_snapshot=None,
            request_params={"k": config.k_retrieval},
            embedding_model=cell.embedding_model,
            index_build_id=cell.index_build_id or NOT_APPLICABLE,
            corpus_version=config.corpus_version,
            chunker_config_hash=config.chunker_config_hash,
            retrieval_config={"stage": "embedder", "component_id": cell.component_id, "k": config.k_retrieval},
            dataset_version=config.dataset_version,
            judge_id=None,
        ),
        "metrics": _report_dict(cell.report),
    }


def _reranker_stage_row(config_id: str, cell: RerankerCell, config: MatrixRunConfig, embedder_cells: dict) -> dict:
    embedder_cell = embedder_cells[cell.embedder_component_id]
    return {
        "config_id": config_id,
        "lineage": build_manifest_row(
            run_id=config.run_id,
            git_sha=config.git_sha,
            prompt_hash=NOT_APPLICABLE,
            model_snapshot=None,
            request_params={"k_final": config.k_retrieval, "depth": cell.depth},
            embedding_model=embedder_cell.embedding_model,
            index_build_id=embedder_cell.index_build_id or NOT_APPLICABLE,
            corpus_version=config.corpus_version,
            chunker_config_hash=config.chunker_config_hash,
            retrieval_config={
                "stage": "reranker",
                "embedder": cell.embedder_component_id,
                "reranker": cell.reranker_component_id,
                "depth": cell.depth,
            },
            dataset_version=config.dataset_version,
            judge_id=None,
        ),
        "metrics": _report_dict(cell.report),
    }


def _generation_stage_row(
    cell: GenerationCell,
    generator: GeneratorComponent,
    config: MatrixRunConfig,
    embedding_model: dict | None,
    judge_ids: Sequence[str],
) -> dict:
    correctness = [c["correctness"] for c in cell.per_case.values()]
    disagreed = [c["panel_disagreed"] for c in cell.per_case.values()]
    return {
        "config_id": cell.config_id,
        "generator_component_id": cell.generator_component_id,
        "lineage": build_manifest_row(
            run_id=config.run_id,
            git_sha=config.git_sha,
            prompt_hash=cell.prompt_hash,
            model_snapshot=generator.model_snapshot,
            request_params={},
            embedding_model=embedding_model,
            index_build_id=NOT_APPLICABLE,
            corpus_version=config.corpus_version,
            chunker_config_hash=config.chunker_config_hash,
            retrieval_config={"stage": "generator", "retrieval_config_id": cell.config_id},
            dataset_version=config.dataset_version,
            judge_id="|".join(judge_ids) if judge_ids else None,
        ),
        "mean_correctness": sum(correctness) / len(correctness) if correctness else 0.0,
        "disagreement_rate": sum(disagreed) / len(disagreed) if disagreed else 0.0,
    }


def run_matrix(
    cases: Sequence[MatrixCase],
    *,
    embedders: Sequence[EmbedderComponent],
    rerankers: Sequence[RerankerComponent],
    generators: Sequence[GeneratorComponent],
    judges: Sequence[BaseChatModel],
    judge_ids: Sequence[str],
    cache: MatrixCache,
    config: MatrixRunConfig,
    output_dir: Path,
    spend_gate: Optional[SpendGate] = None,
    variants: Optional[VariantsConfig] = None,
) -> dict:
    """Run every stage, assemble the manifest, write it plus per query result files to
    `output_dir`, and return the manifest dict (the SAME dict `manifest.json` was written from)."""
    output_dir = Path(output_dir)

    # Stage 1's own metric stays at k_retrieval; the CACHED candidate pool is widened to the
    # widest reranker depth this run sweeps, so stage 2 has real headroom to sweep depths at all
    # (a pool truncated to k_retrieval would leave every depth in `reranker_depths` with no effect).
    pool_size = max((*config.reranker_depths, config.k_retrieval))
    embedder_cells = run_embedder_stage(
        cases, embedders, k=config.k_retrieval, seed=config.seed, cache=cache,
        corpus_version=config.corpus_version, dataset_version=config.dataset_version, pool_size=pool_size,
    )
    reranker_cells = run_reranker_stage(
        cases, embedder_cells, rerankers, k_final=config.k_retrieval, seed=config.seed, cache=cache,
        corpus_version=config.corpus_version, dataset_version=config.dataset_version,
        depths=config.reranker_depths,
    )

    stage1_scores = {
        cid: _per_case_ndcg(cell.candidates, cases, config.k_retrieval) for cid, cell in embedder_cells.items()
    }
    stage1_stats = compare_components(stage1_scores, seed=config.seed) if len(stage1_scores) > 1 else []

    all_configs = all_retrieval_configs(embedder_cells, reranker_cells, k_final=config.k_retrieval)
    top_configs = select_top_configs(all_configs, n=config.n_top_configs)
    if not top_configs:
        raise ValueError("run_matrix needs at least one retrieval config; embedders/rerankers produced none")

    primary = top_configs[0]
    primary_report = _report_for_config(primary.config_id, embedder_cells, reranker_cells)
    _, lo, hi = primary_report.ndcg_at_k_ci
    promotion_gate = gate_on_lower_bound(
        (lo, hi), threshold=config.gate_threshold, variance_budget=config.gate_variance_budget
    )

    # SP9 task 5: the spend gate checks BEFORE a cell runs at all, never after. `gate` is rebound
    # (never mutated: SpendGate is immutable) as spend accrues, so a later cell sees an earlier
    # cell's own spend already reflected in its remaining budget. `spend_gate=None` (every caller
    # before this task) skips this entirely: `_admit` always allows, `dropped_cells` stays empty.
    gate = spend_gate
    dropped_cells: list[DroppedCell] = []

    def _admit(generator: GeneratorComponent, cell_id: str) -> bool:
        """FORWARD NOTE for whoever builds the live capture driver (SP9 task 5's own disclosed,
        not yet wired seam): this only ever gates on `generator.estimated_usd`, the caller
        declared PRE call estimate (`GeneratorComponent`'s own docstring), never the real
        measured cost `matrix.spend_gate.cost_from_usage` reads back from a cell's own
        `usage_metadata` after it runs. `estimated_usd` defaults to `0.0`, and
        `matrix.spend_gate.check_spend` now REFUSES a zero or unknown estimate against any paid
        provider (the live money safety net this task's own review demanded) -- so a live driver
        that forgets to compute an honest, positive estimate has its cell dropped rather than
        silently admitted for free, but that dropped cell also never runs at all. The live driver
        MUST compute `estimated_usd` from `matrix.spend_gate.generation_cost_usd` (or an
        equivalent honest upfront estimate) for every real provider cell before calling
        `run_matrix`, or that cell contributes nothing to the manifest."""
        nonlocal gate
        if gate is None:
            return True
        provider = generator.model_snapshot.get("provider", "unknown")
        decision = check_spend(gate, provider, generator.estimated_usd)
        if not decision.allowed:
            dropped_cells.append(dropped_cell_for(cell_id, decision))
            return False
        gate = record_spend(gate, provider, generator.estimated_usd)
        return True

    generation_cells: dict[str, GenerationCell] = {}
    for generator in generators:
        cell_id = _gen_key(primary.config_id, generator.component_id)
        if not _admit(generator, cell_id):
            continue
        generation_cells[cell_id] = run_generation_cell(primary, generator, cases, judges=judges)

    off_diagonal: OffDiagonalCheck | None = None
    if len(top_configs) > 1 and generators:
        secondary = top_configs[1]
        shared_generator = generators[0]
        primary_key = _gen_key(primary.config_id, shared_generator.component_id)
        secondary_key = _gen_key(secondary.config_id, shared_generator.component_id)
        # Both halves of the off diagonal check need a real cell: the primary one may already be
        # missing (dropped above), and the secondary is its own separate call, gated the same way.
        if primary_key in generation_cells and _admit(shared_generator, secondary_key):
            secondary_cell = run_generation_cell(secondary, shared_generator, cases, judges=judges)
            generation_cells[secondary_key] = secondary_cell
            off_diagonal = off_diagonal_validation(
                generation_cells[primary_key], secondary_cell, shared_generator_id=shared_generator.component_id,
            )

    stage3_stats = []
    if generators:
        generator_scores = {
            gen.component_id: [
                generation_cells[_gen_key(primary.config_id, gen.component_id)].per_case[c.case_id]["correctness"]
                for c in cases
            ]
            for gen in generators
            if _gen_key(primary.config_id, gen.component_id) in generation_cells
        }
        stage3_stats = compare_components(generator_scores, seed=config.seed)

    # SP9 final review (finding I1): `variants=None` (every caller before this stage existed) skips
    # this entirely, `manifest["variant_comparison"]` stays `None` -- the same "absent config,
    # absent effect" shape `spend_gate=None` already established. `asyncio.run` is the ONE bridge
    # point between this still synchronous function and the variant stage's own async LangGraph calls.
    variant_rows = None
    if variants is not None:
        variant_results = asyncio.run(
            run_variant_comparison(
                cases, retriever=variants.retriever, reranker=variants.reranker, graph=variants.graph,
                gateway=variants.gateway, k=variants.k,
            )
        )
        variant_rows = variant_comparison_rows(variant_results)

    manifest = _assemble_manifest(
        config=config, embedder_cells=embedder_cells, reranker_cells=reranker_cells,
        generation_cells=generation_cells, generators=generators, top_configs=[c.config_id for c in top_configs],
        stage1_stats=stage1_stats, stage3_stats=stage3_stats, promotion_gate=promotion_gate,
        off_diagonal=off_diagonal, judge_ids=judge_ids, dropped_cells=dropped_cells,
        variant_comparison=variant_rows,
    )
    per_query = _assemble_per_query(
        cases=cases, config=config, embedder_cells=embedder_cells, reranker_cells=reranker_cells,
        generation_cells=generation_cells,
    )
    _write_outputs(output_dir, manifest, per_query)
    return manifest


def _assemble_manifest(
    *,
    config: MatrixRunConfig,
    embedder_cells: dict[str, EmbedderCell],
    reranker_cells: dict[str, RerankerCell],
    generation_cells: dict[str, GenerationCell],
    generators: Sequence[GeneratorComponent],
    top_configs: list[str],
    stage1_stats,
    stage3_stats,
    promotion_gate: GateDecision,
    off_diagonal: OffDiagonalCheck | None,
    judge_ids: Sequence[str],
    dropped_cells: Sequence[DroppedCell] = (),
    variant_comparison: list[dict] | None = None,
) -> dict:
    generators_by_id = {g.component_id: g for g in generators}
    embedder_rows = [_embedder_stage_row(cell, config) for _, cell in sorted(embedder_cells.items())]
    reranker_rows = [
        _reranker_stage_row(cid, cell, config, embedder_cells) for cid, cell in sorted(reranker_cells.items())
    ]
    generator_rows = []
    for key in sorted(generation_cells):
        cell = generation_cells[key]
        generator = generators_by_id[cell.generator_component_id]
        embedding_model = embedder_cells[_embedder_id_for_config(cell.config_id)].embedding_model
        generator_rows.append(_generation_stage_row(cell, generator, config, embedding_model, judge_ids))

    return {
        "run_id": config.run_id,
        "stages": {"embedders": embedder_rows, "rerankers": reranker_rows, "generators": generator_rows},
        "baseline_component_ids": sorted(BASELINE_COMPONENT_IDS),
        "top_configs": top_configs,
        "promotion_gate": _gate_dict(promotion_gate),
        "stage1_stats": [delta_to_dict(d) for d in stage1_stats],
        "stage3_stats": [delta_to_dict(d) for d in stage3_stats],
        "off_diagonal": _off_diagonal_dict(off_diagonal) if off_diagonal is not None else None,
        # SP9 task 5: a cell SKIPPED for budget reasons, logged, never silent -- ALWAYS present
        # (an empty list, not an absent key, when nothing was dropped), sorted so two runs over the
        # SAME drop set agree byte for byte, the same determinism discipline every other list here holds to.
        "dropped_cells": sorted((d.to_dict() for d in dropped_cells), key=lambda d: (d["component_id"], d["provider"])),
        # SP9 final review (finding I1): `None` when no `VariantsConfig` was supplied (every caller
        # before this stage existed), a sorted list of one row per variant otherwise -- the naive vs
        # agentic vs graph comparison this repo's earlier matrix stages never ran.
        "variant_comparison": variant_comparison,
    }


def _assemble_per_query(
    *,
    cases: Sequence[MatrixCase],
    config: MatrixRunConfig,
    embedder_cells: dict[str, EmbedderCell],
    reranker_cells: dict[str, RerankerCell],
    generation_cells: dict[str, GenerationCell],
) -> dict[str, dict]:
    per_query: dict[str, dict] = {}
    for case in cases:
        entry: dict = {"embedders": {}, "rerankers": {}, "generators": {}}
        for cid, cell in embedder_cells.items():
            retrieved = tuple(c["doc_id"] for c in cell.candidates.get(case.case_id, ()))
            entry["embedders"][cid] = {
                "retrieved": list(retrieved),
                "recall_at_k": ir_metrics.recall_at_k(retrieved, case.relevant_doc_ids, config.k_retrieval),
                "ndcg_at_k": ir_metrics.ndcg_at_k(retrieved, case.relevant_doc_ids, config.k_retrieval),
            }
        for cid, cell in reranker_cells.items():
            retrieved = tuple(c["doc_id"] for c in cell.candidates.get(case.case_id, ()))
            entry["rerankers"][cid] = {
                "retrieved": list(retrieved),
                "recall_at_k": ir_metrics.recall_at_k(retrieved, case.relevant_doc_ids, config.k_retrieval),
                "ndcg_at_k": ir_metrics.ndcg_at_k(retrieved, case.relevant_doc_ids, config.k_retrieval),
            }
        for key, cell in generation_cells.items():
            if case.case_id in cell.per_case:
                entry["generators"][key] = cell.per_case[case.case_id]
        per_query[case.case_id] = entry
    return per_query


def _write_outputs(output_dir: Path, manifest: dict, per_query: dict[str, dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    per_query_dir = output_dir / "per_query"
    per_query_dir.mkdir(parents=True, exist_ok=True)
    for case_id, entry in per_query.items():
        (per_query_dir / f"{case_id}.json").write_text(json.dumps(entry, sort_keys=True, indent=2) + "\n")


__all__ = ["MatrixRunConfig", "run_matrix"]
