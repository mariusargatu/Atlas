"""The HITL adjudication backend route (SP8 Task 4, label collection half, pulled early).

Two endpoints, both small on purpose (D30's own "smallest honest tool" call, echoed in the plan's
own throughput math: ~200 items in 1-2 hours rules out anything heavier):

  - `GET /labels/items`: the label item set (produced by the batch answer generator,
    `testing/harness/labeling/generate_label_set.py`) in FIXED SEED ORDER -- the file's own row
    order, read once at router construction, never reshuffled, never a managed queue (D30). Also
    returns the current progress, so the page can resume where a labeler left off after a reload.
  - `POST /labels`: appends one label to the `LabelStore` (append only JSONL, see that module's own
    docstring) and returns the updated progress. `role` defaults to `"adjudicator"` (this page's own
    role); the end user thumbs widget, a SEPARATE smaller scope the plan names but this task does
    not build, would POST here too with `role="end_user"` once it exists.

Progress is read FRESH on every call (`store.labeled_trace_ids()`, never cached), the same
discipline `atlas.metrics.render()` already holds every gauge to: a label written by one browser
tab shows up on the very next request from another.

Phoenix annotation mirroring (D30: "S3 is system of record, Phoenix a view") is wired here (SP8
Task 4 remainder): `post_label` mirrors every STORED label (never a rejected one -- the mirror call
sits after `store.append` succeeds, so a 422 never reaches Phoenix either) via
`atlas.adapters.phoenix_annotations.mirror_label`, defaulting to `NullPhoenixAnnotationClient` (a
documented no op) so `task test` never makes a live Phoenix call. Wiring a REAL Phoenix client (an
HTTP call to Phoenix's own annotation REST endpoint) is an operator/live concern, not built here.

No bearer auth: this is an internal adjudication tool for the person running the labeling session,
not a customer facing route (unlike `/chat`, which derives identity from a bearer token per
ADR-028). Scoping this behind auth is a reasonable later hardening step, out of this task's scope.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from atlas.adapters.label_store import LabelStore
from atlas.adapters.phoenix_annotations import (
    NullPhoenixAnnotationClient,
    PhoenixAnnotationClient,
    mirror_label,
)


class RetrievedChunkOut(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    score: float = 0.0


class RegistryFactOut(BaseModel):
    fact_id: str
    value: str


class LabelItemOut(BaseModel):
    case_id: str
    trace_id: str
    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunkOut] = []
    registry_facts: list[RegistryFactOut] = []
    source: str | None = None


class ProgressOut(BaseModel):
    labeled: int
    total: int


class LabelItemsOut(BaseModel):
    items: list[LabelItemOut]
    progress: ProgressOut


class LabelIn(BaseModel):
    trace_id: str
    verdict: str
    critique: str
    role: str = "adjudicator"


class LabelOut(BaseModel):
    progress: ProgressOut


def _load_items(path: Path) -> list[dict]:
    """Absent file (the operator has not run the generator yet) is an empty set, never an error --
    the same "absence is not a fault" reading `LabelStore.read_all` already applies on the write
    side."""
    if not path.is_file():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def build_label_router(
    *, items_path: Path, store: LabelStore, phoenix_client: Optional[PhoenixAnnotationClient] = None
) -> APIRouter:
    """`items` is read ONCE, at router construction (fixed seed order, D30: no managed queue, so
    the item set is not expected to change mid session); `server.py` builds a fresh router (and so
    reads the file again) on every process start. Progress is NOT captured here -- computed fresh on
    every request from `store`, see this module's own docstring.

    `phoenix_client` defaults to `NullPhoenixAnnotationClient` (the hermetic no op): `server.py`
    passes nothing today, matching the "operator/live concern, not built here" scope this module's
    own docstring names for a real Phoenix client."""
    router = APIRouter()
    items = _load_items(Path(items_path))
    known_trace_ids = {item["trace_id"] for item in items}
    total = len(items)
    client = phoenix_client if phoenix_client is not None else NullPhoenixAnnotationClient()

    def _progress() -> ProgressOut:
        return ProgressOut(labeled=len(store.labeled_trace_ids()), total=total)

    @router.get("/labels/items", response_model=LabelItemsOut)
    def get_items() -> LabelItemsOut:
        return LabelItemsOut(items=[LabelItemOut(**item) for item in items], progress=_progress())

    @router.post("/labels", response_model=LabelOut)
    def post_label(body: LabelIn) -> LabelOut:
        if body.trace_id not in known_trace_ids:
            raise HTTPException(status_code=404, detail=f"unknown trace_id: {body.trace_id!r} is not in the label item set")
        try:
            record = store.append(trace_id=body.trace_id, role=body.role, verdict=body.verdict, critique=body.critique)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        mirror_label(client, trace_id=record.trace_id, verdict=record.verdict, critique=record.critique)
        return LabelOut(progress=_progress())

    return router


__all__ = ["build_label_router"]
