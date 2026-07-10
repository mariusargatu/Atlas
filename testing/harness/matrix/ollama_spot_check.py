"""SP9 task 7: the "judge spot checked on local outputs" seam (D28; HLD 5.2, "Judge behavior on
local generator outputs is human spot checked before the kappa gate is trusted on that
distribution") -- REUSING SP8's EXISTING HITL adjudication surface, never a new labeling page. The
SP9 planning digest names this explicitly (section 3(f)): "recommend SP9 produce a small sample of
Ollama generated matrix answers as one more label items JSONL and route it through the existing
`/adjudicate` page, rather than build a second labeling surface."

`build_ollama_spot_check_items` turns one Stage 3 `matrix.generators.GenerationCell` (the SAME
`per_case` `{"answer", ...}` shape `matrix.runner` already assembles into the run manifest, never a
second data shape) into the SAME label item shape
`labeling.generate_label_set.generate_label_items` already produces --
`{"case_id", "trace_id", "question", "answer", "retrieved_chunks", "registry_facts", "source"}` --
so it loads through the identical `atlas.label_routes.build_label_router`/`LabelItemOut` seam
(`ATLAS_LABEL_ITEMS_PATH`, `task label:generate-live`'s own machinery) with no code changes on that
side at all. Writing the JSONL itself reuses `labeling.generate_label_set.write_label_items` (the
SAME byte reproducible writer), never a second writer.

`trace_id` here is a content digest (`determinism.canonical.digest`, the SAME digest function the
cassette key already uses) over `(run_id, case_id, generator_component_id)` -- a stable, unique per
item identity, but deliberately NOT a live Atlas `InMemoryTracer.trace_root`: these answers come
from `matrix.generators.run_generation_cell`'s own direct `gateway.invoke()` call in a benchmark
cell, never a served Atlas turn, so borrowing that other id's shape without saying so would be
misleading provenance, not a real trace.

Never fabricates an item (the same rule `generate_label_items` already holds): a case absent from
the cell's own `per_case`, or one whose answer is empty, is skipped, never padded.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from determinism.canonical import digest

from matrix.cases import MatrixCase
from matrix.generators import GenerationCell
from matrix.select import RetrievalConfigResult

#: Marks every item this module produces, the same "never mistake a canned/derived item for a real
#: one" discipline `label_items.fixture.jsonl`'s own `"source": "fixture"` already established.
SOURCE = "ollama-matrix-spot-check"


def _retrieved_chunks(config: RetrievalConfigResult, case_id: str) -> list[dict]:
    """Only the four fields `atlas.label_routes.RetrievedChunkOut` declares -- `config.candidates`'
    own dicts (`matrix.chunks.serialize_chunk`'s shape) carry more (parent_id, doc_version, ...),
    which Pydantic would silently ignore anyway, but naming exactly what this seam needs is the
    same "never carry an accidental extra field into a contract" discipline the rest of this
    package already holds to."""
    return [
        {"doc_id": c["doc_id"], "chunk_id": c["chunk_id"], "text": c["text"], "score": c["score"]}
        for c in config.candidates.get(case_id, ())
    ]


def _registry_facts(case: MatrixCase) -> list[dict]:
    return [{"fact_id": f["fact_id"], "value": str(f["value"])} for f in case.expected_facts]


def _trace_id(run_id: str, case_id: str, generator_component_id: str) -> str:
    return digest({"run_id": run_id, "case_id": case_id, "generator_component_id": generator_component_id})


def build_ollama_spot_check_items(
    config: RetrievalConfigResult,
    cell: GenerationCell,
    cases: Sequence[MatrixCase],
    *,
    run_id: str,
    limit: Optional[int] = None,
) -> list[dict]:
    """One label item per case IN `cases`' OWN ORDER (fixed seed order, the same D30 discipline
    `generate_label_items` already holds -- never shuffled, never sampled at random), for every
    case `cell` actually answered. `limit` (the plan's own "small sample," never the full case
    set) takes the first N ITEMS PRODUCED in that same order, mirroring `load_seed_cases`'s own
    `limit` contract; `None` (the default) returns every answered case."""
    items: list[dict] = []
    for case in cases:
        if limit is not None and len(items) >= limit:
            break
        entry = cell.per_case.get(case.case_id)
        if entry is None or not entry.get("answer"):
            continue
        items.append({
            "case_id": case.case_id,
            "trace_id": _trace_id(run_id, case.case_id, cell.generator_component_id),
            "question": case.query,
            "answer": entry["answer"],
            "retrieved_chunks": _retrieved_chunks(config, case.case_id),
            "registry_facts": _registry_facts(case),
            "source": SOURCE,
        })
    return items


__all__ = ["SOURCE", "build_ollama_spot_check_items"]
