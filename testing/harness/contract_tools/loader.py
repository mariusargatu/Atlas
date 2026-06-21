"""Load the contract schemas and their golden examples from the repo root contracts/ directory."""

from __future__ import annotations

import json
from pathlib import Path

CONTRACTS_DIR = Path(__file__).resolve().parents[3] / "contracts"
FAMILIES: tuple[str, ...] = ("trace", "dataset", "manifest", "sse")

# Reserved in trace v0.1 so their later use is never a breaking bump (HLD D13 revised, D27).
# `atlas.turn.seq` (trace 1.1.0, I1 fix, SP6 final review, added AFTER the v1.0.0 freeze -- the one
# member of this tuple not reserved ahead of use, but emitted for real the SAME commit it was
# reserved in): the join key between the response envelope/log trace id and the real exported span
# it names, stamped by `OtelTracer.open()` on every span (`backend/atlas/adapters/otel_tracer.py`).
RESERVED_TRACE_ATTRIBUTES: tuple[str, ...] = (
    "atlas.semconv.version",
    "atlas.variant",
    "atlas.config.hash",
    "atlas.corpus.version",
    "atlas.index.build_id",
    "atlas.stage.embed_ms",
    "atlas.stage.retrieve_ms",
    "atlas.stage.rerank_ms",
    "atlas.stage.assemble_ms",
    "atlas.stage.ttft_ms",
    "atlas.degradation.mode",
    "atlas.retrieval.doc_ids",
    "atlas.rerank.scores_pre",
    "atlas.rerank.scores_post",
    "atlas.guard.decision",
    "atlas.judge.id",
    "atlas.judge.verdict",
    "atlas.judge.rubric_version",
    "atlas.cost.input_tokens",
    "atlas.cost.output_tokens",
    "atlas.cost.usd",
    "atlas.subject.pseudonym",
    "atlas.privacy.redaction_policy_version",
    "atlas.privacy.synthetic",
    "atlas.privacy.content_captured",
    "atlas.contract.trace_version",
    "atlas.contract.dataset_version",
    "atlas.contract.manifest_version",
    "atlas.contract.sse_version",
    "atlas.turn.seq",
)

# The 12 field attribution tuple, exact names (HLD D26).
LINEAGE_FIELDS: tuple[str, ...] = (
    "run_id",
    "git_sha",
    "prompt_hash",
    "model_snapshot",
    "request_params",
    "embedding_model",
    "index_build_id",
    "corpus_version",
    "chunker_config_hash",
    "retrieval_config",
    "dataset_version",
    "judge_id",
)


def _family_dir(family: str) -> Path:
    if family not in FAMILIES:
        raise ValueError(f"unknown contract family: {family!r}")
    return CONTRACTS_DIR / family


def load_schema(family: str) -> dict:
    return json.loads((_family_dir(family) / "schema.json").read_text())


def load_examples(family: str) -> dict[str, dict | list]:
    examples_dir = _family_dir(family) / "examples"
    if not examples_dir.is_dir():
        return {}
    return {p.stem: json.loads(p.read_text()) for p in sorted(examples_dir.glob("*.json"))}


def contract_versions() -> dict[str, str]:
    return {family: load_schema(family)["x-contract-version"] for family in FAMILIES}
