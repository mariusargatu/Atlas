"""Append only label JSONL writer (SP8 Task 4, label collection half, pulled early per the plan's
RESEQUENCING note at e5c82c3: the HITL page and its storage do not depend on the judge, Tasks 1-3;
the human labels the AGENT's answer for groundedness, and the judge is what those labels calibrate
AGAINST, never a prerequisite for collecting them).

Storage: a local path standing in for the S3 label prefix D30 names as the eventual system of
record (real S3/R2 sync is late binding, the plan's own Global Constraints: "label storage writes
locally first... real S3/R2 sync is late binding, never a prerequisite"). Every line is appended,
never rewritten -- `LabelStore` has no update/delete, only `append`/`read_all`, so a correction is
itself a new line, never an edit to history (the same "the audit log is queryable, never mutated"
discipline `domain/actions.py`'s own log applies to writes).

Keyed by `trace_id`, not `span_id`: the chat response envelope (`chat_app.py`'s `message_start`
event) carries only a turn level `trace_id`, never a real `span_id` (the digest's own documented
gap, resolved here the same way SP8's plan resolves it -- key by the identity that actually exists).

`role` distinguishes who produced the verdict: `"adjudicator"` (the HITL page this task builds) vs
`"end_user"` (the in chat thumbs widget -- a SEPARATE, smaller scope named in the plan but NOT built
by this task; the field exists so that widget has a home to write through later without a storage
migration, the same "declare the seam before it has a caller" discipline `atlas.metrics`'s own
`atlas_judge_pass_total` counter demonstrated before the SP8 Task 4 remainder wired it
(`record_judge_pass`/`record_judge_fail`, `judge.emission.emit_verdict`'s own call site)).

`created_at` comes from the injected clock (`determinism.sources.FrozenClock` in tests, a real
clock in a served process), never `datetime.now()` -- the determinism contract every runtime path
in this codebase holds itself to. Every line is written through `canonical_json` (the SAME
canonicalization the cassette key and run digest already use), so two writers fed the identical
call sequence under the identical frozen clock produce byte identical files.

Phoenix annotation mirroring (D30: "S3 is system of record, Phoenix a view") is wired at the
CALLER, not in this module: `backend/atlas/label_routes.py`'s `post_label` mirrors every stored
record via `atlas.adapters.phoenix_annotations.mirror_label`, after `append` below already returns.
`LabelStore` itself stays storage only -- no Phoenix import here, the same "the write and the
mirror are two separate, independently failing concerns" discipline that keeps the metrics counter
increment (`record_judge_pass`/`record_judge_fail`) a thin call beside the span it observes, never
folded into the write it is observing.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from determinism.canonical import canonical_json

_VALID_ROLES = ("adjudicator", "end_user")
_VALID_VERDICTS = ("pass", "fail")


@dataclass(frozen=True)
class LabelRecord:
    """One line of the append only JSONL. Immutable: a correction is a NEW record, never an edit."""

    trace_id: str
    role: str
    verdict: str
    critique: str
    created_at: str


class LabelStore:
    """Append only JSONL writer + reader over one local file, standing in for an S3 label prefix.

    `path` need not exist yet -- `append()` creates its parent directories on first write, the same
    "an operator running fresh needs no manual setup" discipline `server.py`'s own cassette dir
    check documents (there, failing fast if absent; here, creating it, since a label store starting
    empty is the expected first run, not a misconfiguration).
    """

    def __init__(self, path: Path, clock) -> None:
        self._path = Path(path)
        self._clock = clock

    def append(self, *, trace_id: str, role: str, verdict: str, critique: str) -> LabelRecord:
        """The only writer this file ever has. Raises `ValueError` on any malformed field -- the
        caller (the backend label route) turns that into a 422, never a silently dropped label."""
        if not trace_id:
            raise ValueError("trace_id is required: a label must be keyed to the turn it grades")
        if role not in _VALID_ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {_VALID_ROLES}")
        if verdict not in _VALID_VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}; expected one of {_VALID_VERDICTS}")
        if not critique or not critique.strip():
            raise ValueError("critique is required: a one sentence critique, never empty")
        record = LabelRecord(
            trace_id=trace_id, role=role, verdict=verdict, critique=critique,
            created_at=self._clock.now().isoformat(),
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(canonical_json(asdict(record)))
            fh.write("\n")
        return record

    def read_all(self) -> list[LabelRecord]:
        """Every record ever appended, in append order. An absent file (nothing labeled yet) is an
        empty list, never an error -- the same "absence is not a fault" reading `metrics.py`'s own
        `_corpus_staleness` applies to a missing manifest."""
        if not self._path.is_file():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(LabelRecord(**json.loads(line)))
        return records

    def labeled_trace_ids(self, role: str = "adjudicator") -> set[str]:
        """The DISTINCT trace ids with at least one record under `role`, the progress counter's own
        denominator source. A trace relabeled (a correction, append only per this module's own
        docstring) still counts ONCE toward progress even though both lines survive on disk."""
        return {r.trace_id for r in self.read_all() if r.role == role}


__all__ = ["LabelRecord", "LabelStore"]
