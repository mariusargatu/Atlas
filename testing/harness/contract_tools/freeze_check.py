"""SP6 task 7: the trace contract's v1.0.0 freeze gate.

Walks every one of the 30 `RESERVED_TRACE_ATTRIBUTES` (`contract_tools.loader`, the single checklist
authority) and requires TWO independent kinds of evidence before an attribute counts as safe to
freeze:

  (a) HERMETIC translation evidence (`hermetic_emitted_attributes`): attribute names that actually
      appear when the real production translation code (`backend/atlas/adapters/trace_translation.py`
      -- `translate_span`, `BUILD_ATTRIBUTES`, `STAGE_DURATION_ATTRIBUTE`) runs against the committed
      golden span inventory (`contracts/trace/span_inventory.json`), plus the two settings sourced
      turn attributes `otel_tracer.py` stamps directly (`atlas.config.hash`/`atlas.corpus.version`/
      `atlas.index.build_id` -- see `_SETTINGS_SOURCED_TURN_ATTRIBUTES` below). Entirely hermetic: no
      network, no live process, the same production code path `test_trace_translation.py` and
      `test_otel_tracer.py` already exercise, just walked here for full 29-name coverage in one place.

  (b) LIVE capture evidence (`live_emitted_attributes`): attribute names observed on a REAL exported
      span, captured once by an operator against the `docker compose --profile observability` stack
      with `ATLAS_TRACING=otel` (one healthy turn, one degraded turn) and committed as a raw OTLP
      JSON evidence artifact (`contracts/trace/freeze_evidence.json`) with the exact capture command
      recorded inside the file itself (the `_capture` key). Reading this file at check time needs no
      network and no live process either -- it is committed data, read once. Regenerating it is an
      operator only step (see the module docstring on `live_emitted_attributes` and the committed
      file's own `_capture` field); the hermetic lane never reruns the capture.

Evidence (a) alone proves the CODE would produce the right vocabulary. Evidence (b) alone proves
something real actually left a real process through the real OTel pipeline. Neither alone is enough
(a hermetic simulation can be wrong about what real infrastructure actually exports; a hand edited
"live" artifact could claim anything) -- an attribute is "emitted" only with BOTH.

An attribute named in the narrowed set (`contracts/trace/freeze_narrowed.yaml`, the machine readable
companion `ADR-029` names) is exempt from both: its
absence from (a)/(b) is expected and documented, never a silent skip. Any OTHER attribute missing
either source STOPS the freeze -- `main()` exits 1 and prints exactly which attribute(s) and which
evidence source(s) are missing, so a reviewer never has to eyeball a diff to find the gap.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from atlas.adapters import trace_translation

from contract_tools.loader import CONTRACTS_DIR, RESERVED_TRACE_ATTRIBUTES
from contract_tools.trace_inventory import INVENTORY_PATH

REPO_ROOT = CONTRACTS_DIR.parent
LIVE_EVIDENCE_PATH = CONTRACTS_DIR / "trace" / "freeze_evidence.json"
NARROWED_PATH = CONTRACTS_DIR / "trace" / "freeze_narrowed.yaml"
ADR_PATH = REPO_ROOT / "docs" / "adr" / "ADR-029-trace-contract-v1-freeze-narrowing.md"

# `atlas.config.hash`/`atlas.corpus.version`/`atlas.index.build_id` are stamped directly by
# `otel_tracer.py`'s own turn kind branch (settings sourced identity), never through
# `trace_translation.translate_span`'s merge -- `trace_translation.py` itself stays settings free
# (its own purity claim, `test_trace_translation_module_stays_pure`). Named here explicitly, never
# guessed: `otel_tracer.py`'s `open()` is the single source of truth this list transcribes, and
# `test_otel_tracer.py`'s own settings sourced tests are what keeps the transcription honest.
_SETTINGS_SOURCED_TURN_ATTRIBUTES: frozenset[str] = frozenset({
    "atlas.config.hash", "atlas.corpus.version", "atlas.index.build_id",
})

# `atlas.turn.seq` (trace 1.1.0, I1 fix, SP6 final review): stamped by `otel_tracer.py`'s `open()`
# on EVERY span kind, unconditionally, never gated on the golden span inventory's content the way
# `translate_span`'s own output is -- structurally true wherever `open()` runs at all, the same
# reasoning `_SETTINGS_SOURCED_TURN_ATTRIBUTES` above already applies to the three turn scoped
# settings sourced attributes, just not turn scoped itself (every kind gets it, not only "turn").
_ADAPTER_STAMPED_ATTRIBUTES: frozenset[str] = frozenset({"atlas.turn.seq"})


class NarrowedFormatError(Exception):
    """`contracts/trace/freeze_narrowed.yaml` is missing, malformed, or names an attribute the
    trace contract does not reserve (a typo would otherwise silently exempt nothing, or the wrong
    thing)."""


@dataclass(frozen=True)
class NarrowedEntry:
    owner: str
    reason: str


@dataclass(frozen=True)
class AttributeStatus:
    attribute: str
    status: str  # "emitted" | "narrowed" | "missing_hermetic" | "missing_live" | "missing_both"

    def __post_init__(self) -> None:
        allowed = {"emitted", "narrowed", "missing_hermetic", "missing_live", "missing_both"}
        if self.status not in allowed:
            raise ValueError(f"invalid AttributeStatus status {self.status!r}: expected one of {allowed}")

    @property
    def ok(self) -> bool:
        return self.status in ("emitted", "narrowed")


def _load_span_inventory_entries() -> list[dict]:
    return json.loads(INVENTORY_PATH.read_text())


def hermetic_emitted_attributes(entries: list[dict] | None = None) -> frozenset[str]:
    """Evidence (a): every attribute name the real translation code produces when run against the
    committed golden span inventory, plus the settings sourced turn attributes `otel_tracer.py`
    stamps outside that translation (see the module docstring). `entries` is test only (the same
    `inventory=`/`entries=` override style `trace_translation`/`contract_tools.redaction` already
    use for exactly this reason); production callers always read the one committed
    `contracts/trace/span_inventory.json` file."""
    if entries is None:
        entries = _load_span_inventory_entries()
    keys: set[str] = set(trace_translation.STAGE_DURATION_ATTRIBUTE.values())
    keys |= _SETTINGS_SOURCED_TURN_ATTRIBUTES
    keys |= _ADAPTER_STAMPED_ATTRIBUTES
    for entry in entries:
        if entry["kind"] == "stage":
            continue  # a stage span carries no informal attrs; its duration is already counted above
        attrs = {key: True for key in entry["attrs"]}
        record = trace_translation.translate_span(entry["name"], entry["kind"], attrs)
        keys.update(record["attributes"])
    return frozenset(keys)


def _walk_otlp_attribute_keys(export: dict) -> set[str]:
    """Every attribute key across every span in one OTLP `ExportTraceServiceRequest`, JSON encoded
    (the shape the OTel collector's `file` exporter writes, `resourceSpans[].scopeSpans[].spans[]`,
    each span's `attributes` an array of `{"key": ..., "value": {...}}` pairs -- OTLP's own JSON
    mapping, never Python's `dict`, so a key is read positionally, not via `.get`)."""
    keys: set[str] = set()
    for resource_span in export.get("resourceSpans", []):
        for scope_span in resource_span.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                for attribute in span.get("attributes", []):
                    key = attribute.get("key")
                    if key is not None:
                        keys.add(key)
    return keys


def live_emitted_attributes(path: Path = LIVE_EVIDENCE_PATH) -> frozenset[str]:
    """Evidence (b): every attribute key observed in the committed live capture artifact. The file
    is a JSON object with two top level keys: `_capture` (a human readable record of exactly how it
    was produced -- the compose profile, the env, the two curl/HTTP calls -- so a reader never has
    to reconstruct the capture procedure from memory) and `exports`, a list of raw OTLP
    `ExportTraceServiceRequest` JSON objects, one per file the collector's `file/raw_archive`
    exporter wrote lines for during the capture window (`infra/observability/otel-collector.yaml`).
    Regenerating this file is an OPERATOR step (bring up `docker compose --profile observability`
    with `ATLAS_TRACING=otel`, drive one healthy and one degraded turn, extract
    `/var/log/otel/raw-otlp-archive.jsonl` from the `otel-collector` container) -- this function only
    ever READS the committed result, never triggers a capture itself, so the hermetic lane never
    depends on live infrastructure being up."""
    data = json.loads(path.read_text())
    keys: set[str] = set()
    for export in data.get("exports", []):
        keys |= _walk_otlp_attribute_keys(export)
    return frozenset(keys)


def load_narrowed(path: Path = NARROWED_PATH) -> dict[str, NarrowedEntry]:
    """The narrowed set, read from its own committed YAML file (never restated in Python): each
    entry names the RESERVED_TRACE_ATTRIBUTES member it narrows, the future sub project (or
    explicitly "unscheduled") that owns emitting it for real, and why it has no producer today.
    `ADR-029` is this file's own prose companion --
    the two are meant to be read together, this function is what keeps the SCRIPT from ever
    restating the ADR's list by hand (the drift `freeze_check.py`'s own module docstring names)."""
    try:
        text = path.read_text()
    except FileNotFoundError as exc:
        raise NarrowedFormatError(f"cannot read {path}: no such file") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise NarrowedFormatError(f"cannot parse {path}: invalid YAML ({exc})") from exc
    entries = (raw or {}).get("narrowed")
    if not isinstance(entries, list):
        raise NarrowedFormatError(f"{path} has no top level 'narrowed' list")
    narrowed: dict[str, NarrowedEntry] = {}
    for item in entries:
        try:
            attribute = item["attribute"]
            owner = item["owner"]
            reason = item["reason"]
        except (KeyError, TypeError) as exc:
            raise NarrowedFormatError(
                f"{path} has a 'narrowed' entry missing 'attribute'/'owner'/'reason': {item!r}"
            ) from exc
        if attribute not in RESERVED_TRACE_ATTRIBUTES:
            raise NarrowedFormatError(
                f"{path} narrows {attribute!r}, which is not a RESERVED_TRACE_ATTRIBUTES member "
                "(contract_tools.loader) -- a typo would otherwise silently exempt nothing real."
            )
        narrowed[attribute] = NarrowedEntry(owner=owner, reason=reason)
    return narrowed


def evaluate(
    *,
    reserved: tuple[str, ...] = RESERVED_TRACE_ATTRIBUTES,
    hermetic_emitted: frozenset[str],
    live_emitted: frozenset[str],
    narrowed: Mapping[str, NarrowedEntry],
) -> tuple[AttributeStatus, ...]:
    """The pure checklist walk: every parameter is injectable (no file I/O here at all), so a test
    can stub the green / missing attribute / narrowed attribute paths directly, and `main()` below
    is the only caller that ever wires this to the real committed sources."""
    statuses = []
    for attribute in reserved:
        if attribute in narrowed:
            statuses.append(AttributeStatus(attribute, "narrowed"))
            continue
        has_hermetic = attribute in hermetic_emitted
        has_live = attribute in live_emitted
        if has_hermetic and has_live:
            statuses.append(AttributeStatus(attribute, "emitted"))
        elif has_hermetic and not has_live:
            statuses.append(AttributeStatus(attribute, "missing_live"))
        elif has_live and not has_hermetic:
            statuses.append(AttributeStatus(attribute, "missing_hermetic"))
        else:
            statuses.append(AttributeStatus(attribute, "missing_both"))
    return tuple(statuses)


def _report_line(status: AttributeStatus, narrowed: Mapping[str, NarrowedEntry]) -> str:
    if status.status == "emitted":
        return f"  [emitted]  {status.attribute}"
    if status.status == "narrowed":
        entry = narrowed[status.attribute]
        return f"  [narrowed] {status.attribute} -- owner: {entry.owner}"
    missing = {
        "missing_hermetic": "no hermetic translation evidence",
        "missing_live": "no live capture evidence",
        "missing_both": "no hermetic AND no live evidence",
    }[status.status]
    return f"  [STOP]     {status.attribute} -- {missing}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    try:
        # NARROWED_PATH/LIVE_EVIDENCE_PATH read as module globals HERE (call time), not as the
        # functions' own default parameter values (bound once at def time): this is what lets a
        # test monkeypatch either module attribute and have `main()` actually honor the override.
        narrowed = load_narrowed(NARROWED_PATH)
    except NarrowedFormatError as exc:
        print(f"error: {exc}")
        return 2

    hermetic = hermetic_emitted_attributes()
    try:
        live = live_emitted_attributes(LIVE_EVIDENCE_PATH)
    except FileNotFoundError:
        print(
            f"error: no live capture evidence at {LIVE_EVIDENCE_PATH}; run the operator capture "
            "step (this module's own docstring on live_emitted_attributes) before checking the freeze."
        )
        return 2

    statuses = evaluate(hermetic_emitted=hermetic, live_emitted=live, narrowed=narrowed)
    for status in statuses:
        print(_report_line(status, narrowed))

    failing = [s for s in statuses if not s.ok]
    if failing:
        print(
            f"\nfreeze STOPPED: {len(failing)} of {len(statuses)} reserved attribute(s) lack real "
            "evidence and are not narrowed. Either add a real emitter (translation table + "
            "inventory regeneration + tests) or narrow it via an ADR "
            f"({ADR_PATH.relative_to(REPO_ROOT)}) and its machine readable companion "
            f"({NARROWED_PATH.relative_to(REPO_ROOT)}), in the SAME commit -- never a silent skip."
        )
        return 1
    print(f"\nfreeze clean: all {len(statuses)} reserved attributes are emitted or narrowed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ADR_PATH",
    "AttributeStatus",
    "LIVE_EVIDENCE_PATH",
    "NARROWED_PATH",
    "NarrowedEntry",
    "NarrowedFormatError",
    "evaluate",
    "hermetic_emitted_attributes",
    "live_emitted_attributes",
    "load_narrowed",
    "main",
]
