"""SP6 task 3: generates the OTel Collector's redaction processor config from the trace contract's
single sourced attribute list, mirroring `contract_tools.mcp_snapshot`/`contract_tools.trace_inventory`'s
own dump and diff pattern (D11/D22's "regenerate and assert equality" mechanism, reused here for
D13's "one redacting collector" and D27's "schema generated redaction allowlist").

`allowed_attributes()` unions three named categories, matching D13's own "Base: pinned OTel GenAI
semconv version + openinference.span.kind (Phoenix compatibility) + atlas.* extensions" framing:
`RESERVED_TRACE_ATTRIBUTES` (`contract_tools.loader`'s own 30 name single checklist authority, ALL
reserved in v1 per D13 regardless of whether each one has a real emitter yet), every `gen_ai.*` key
`backend/atlas/adapters/trace_translation.py`'s table actually produces today, and
`OPENINFERENCE_SPAN_KIND_ATTRIBUTE` (below). The gen_ai category is derived, never hand copied:
`emitted_gen_ai_attributes` walks the committed golden span inventory
(`contracts/trace/span_inventory.json`, Task 2's own artifact) and translates every real
`(name, kind, attrs)` shape through the SAME `translate_attributes` function `otel_tracer.py` calls
on the real export path, collecting every key with a `gen_ai.` prefix. A key trace_translation.py
starts emitting without a matching entry here is a REAL redaction decision gap, not a bookkeeping
one: an informal diagnostic attribute with no namespace (`reason`, `tool`, `input`, `output`, ...)
that may carry real turn content must never be silently allowlisted, and a newly namespaced key
needs a human to look at it once before it leaves the collector unredacted.

Regeneration:

    uv run python -m contract_tools.redaction --write

then review the resulting `git diff infra/observability/otel-collector.yaml` in the SAME change:
`testing/tests/test_redaction.py` fails loud on any drift the committed file does not already
reflect. `--check` (the default; also what the hermetic test itself calls, via `render_collector_
config`, not this CLI) prints a diff without writing.

Pipeline shape the rendered file encodes (see `render_collector_config` for the exact bytes): one
OTLP receiver, ONE redaction processor (the allowlist above), then fan out to Phoenix (wired) and a
local file archive (D13's "raw OTLP archived") -- redaction runs exactly once, upstream of BOTH fan
out targets, so neither Phoenix nor the archive ever sees an unredacted span. LangSmith/LangWatch
exist only as commented exporter blocks (D13): proving pluggability without ever being a live export
target, and without either name ever appearing in pyproject.toml (a Go binary's own YAML is the only
place either name may appear at all, per this plan's Global Constraints).

SP6 task 7 review fix round 1 (Important 1, the versioning trap): `atlas.privacy.redaction_policy_
version` used to be a literal copy of the trace schema's own `x-contract-version`. That is a real
gap, not just an identity: the `gen_ai.*` component of `allowed_attributes()` above is derived from
`trace_translation.py`'s table, a source entirely independent of `contracts/trace/schema.json`, so a
future PR could add a new `gen_ai.*` emitter and silently change the REAL allowlist with no bump
obligation attached anywhere, while the copied version string never moved. `redaction_policy_
version()` below fixes this by making the version CONTENT ADDRESSED: a short sha256 prefix over the
allowlist's own canonical, sorted key list, the same "hash the canonical content, truncate to a
short hex prefix" philosophy `chunk_id`/`index_build_id`/`config_hash` already use elsewhere in this
repo. Every one of the three unioned categories participates, so adding or removing a single key
from ANY of them changes this value automatically -- there is no longer a silent no bump gap,
because the version is never a copy of something else's identity, it is the allowlist's own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

from atlas.adapters import trace_translation

from contract_tools.loader import RESERVED_TRACE_ATTRIBUTES
from contract_tools.trace_inventory import INVENTORY_PATH

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SPAN_INVENTORY_PATH = INVENTORY_PATH
COLLECTOR_CONFIG_PATH = REPO_ROOT / "infra" / "observability" / "otel-collector.yaml"

# The same short hex prefix length `rag_tools.chunker`/`rag_tools.fingerprint` already use for their
# own content addressed ids (`_HASH_HEX_LEN`), reused here so every content addressed id in this
# repo reads the same length at a glance.
_POLICY_VERSION_HEX_LEN = 16

# D13's own "Base" framing names three peer categories, not two: "pinned OTel GenAI semconv version
# + openinference.span.kind (Phoenix compatibility) + atlas.* extensions." `otel_tracer.py` stamps
# this key on every exported span as the OTel wire encoding of the JSON contract's own top level
# `span_kind` field (`trace_translation.span_kind_for`); it is neither a `RESERVED_TRACE_ATTRIBUTES`
# entry nor a `gen_ai.*` key, so it cannot be derived from either generated source above, but
# omitting it from the allowlist strips exactly the field Phoenix's own UI uses to categorize spans
# (LLM/RETRIEVER/CHAIN/...). Found live (SP6 task 3's own operator verification): a real turn through
# the compose `observability` profile showed `openinference.span.kind` in the collector's own
# `redaction.redacted.keys` diagnostic attribute on every single span before this constant existed.
# Included here explicitly, a third named category alongside the two generated ones, never guessed.
OPENINFERENCE_SPAN_KIND_ATTRIBUTE = "openinference.span.kind"


def _load_span_inventory_entries() -> list[dict]:
    return json.loads(SPAN_INVENTORY_PATH.read_text())


def emitted_gen_ai_attributes(entries: list[dict] | None = None) -> frozenset[str]:
    """Every `gen_ai.*` key `trace_translation.translate_attributes` actually produces today,
    derived by translating every real `(name, kind, attrs)` shape the committed golden span
    inventory records -- never a hand maintained literal that could silently drift from the
    translation table's own rules. `entries` is test only (the same `inventory=` override style
    `trace_translation.translate_attributes` itself already uses); production callers always read
    the one committed `contracts/trace/span_inventory.json` file."""
    if entries is None:
        entries = _load_span_inventory_entries()
    keys: set[str] = set()
    for entry in entries:
        attrs = {key: True for key in entry["attrs"]}
        translated = trace_translation.translate_attributes(entry["name"], entry["kind"], attrs)
        keys.update(key for key in translated if key.startswith("gen_ai."))
    return frozenset(keys)


def allowed_attributes() -> tuple[str, ...]:
    """The exact, sorted allowlist: `RESERVED_TRACE_ATTRIBUTES`, every real `gen_ai.*` emitter, and
    `OPENINFERENCE_SPAN_KIND_ATTRIBUTE` (D13's third named "Base" category), duplicate free (there is
    no overlap today between any of the three). Sorted so both the rendered OTTL list and any future
    diff read deterministically, the same discipline `contract_tools.mcp_snapshot`'s own sorted
    rendering already established."""
    return tuple(sorted(
        set(RESERVED_TRACE_ATTRIBUTES) | emitted_gen_ai_attributes() | {OPENINFERENCE_SPAN_KIND_ATTRIBUTE}
    ))


def redaction_policy_version() -> str:
    """`atlas.privacy.redaction_policy_version`'s real, content addressed value (module docstring's
    "SP6 task 7 review fix round 1" note): a short sha256 prefix over the canonical (already sorted,
    duplicate free) `allowed_attributes()` key list, newline joined. Depends on `allowed_attributes()`
    alone, so it changes automatically the moment ANY of the three unioned categories changes -- a
    new `RESERVED_TRACE_ATTRIBUTES` name, a new `gen_ai.*` emitter, or (never expected to change, but
    covered anyway) `OPENINFERENCE_SPAN_KIND_ATTRIBUTE` itself. Written into the generated collector
    config as a comment (`render_collector_config`) and re derived independently, without an import,
    by `backend/atlas/adapters/trace_translation.py`'s own `BUILD_ATTRIBUTES` (a backend module may
    never import this harness one, `testing/tests/test_import_lint.py`'s own boundary) -- the two are
    cross checked by `testing/tests/test_trace_translation.py`, never assumed to agree by
    construction alone."""
    canonical = "\n".join(allowed_attributes())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_POLICY_VERSION_HEX_LEN]


def render_collector_config() -> str:
    """The exact bytes a committed `infra/observability/otel-collector.yaml` holds (and what
    `--write` writes). A plain Python template, not a YAML dump library: the file's own human
    authored comments (the LangSmith/LangWatch pluggability note, D13's own reasoning) are part of
    the committed artifact, the same way `render_snapshot`/`render_inventory` build their own exact
    bytes directly rather than round tripping through a generic serializer that would drop them."""
    keys_block = "\n".join(f"      - {key}" for key in allowed_attributes())
    policy_version = redaction_policy_version()
    return f"""\
# GENERATED FILE, do not hand edit. Source: testing/harness/contract_tools/redaction.py
# (`uv run python -m contract_tools.redaction --write`), the SAME regenerate and assert equality
# mechanism contract_tools.mcp_snapshot and contract_tools.trace_inventory already established
# (D11/D22), reused here for D13's "one redacting collector" and D27's "schema generated redaction
# allowlist."
#
# redaction_policy_version: {policy_version} -- contract_tools.redaction.redaction_policy_version(),
# a short sha256 prefix over the canonical, sorted allowed_keys list below (SP6 task 7 review fix
# round 1, Important 1). Content addressed: this changes automatically if allowed_keys below ever
# changes, adding OR removing any key. Exposed on every exported span as
# atlas.privacy.redaction_policy_version (backend/atlas/adapters/trace_translation.py's own
# BUILD_ATTRIBUTES, derived independently there without an import back into this harness module).
#
# allowed_keys below unions three categories, matching D13's own "Base: pinned OTel GenAI semconv
# version + openinference.span.kind (Phoenix compatibility) + atlas.* extensions" framing:
# RESERVED_TRACE_ATTRIBUTES (testing/harness/contract_tools/loader.py, the 30 name single checklist
# authority, all reserved in v1 per D13 whether or not each one has a real emitter yet), every
# gen_ai.* key backend/atlas/adapters/trace_translation.py's table actually emits today (derived by
# walking the committed contracts/trace/span_inventory.json and translating every real entry, never
# a hand maintained literal), and openinference.span.kind itself (the OTel wire encoding of the
# contract's own top level span_kind field, needed for Phoenix's own span categorization, found live
# during this task's own operator verification). testing/tests/test_redaction.py holds the gen_ai
# category to set equality against the translation table's own live output: a new emitted key with
# no redaction decision fails that hermetic test the moment this file is next regenerated and
# reviewed, it does not silently pass through unredacted.
#
# D13: one OTel collector, redaction with a generated allowlist, then fan out. Phoenix is the ONLY
# deployed backend; LangSmith/LangWatch stay commented configuration below, proving pluggability
# without ever being a live export target (this plan's Global Constraints: neither name may ever
# appear in pyproject.toml, only here, a Go binary's own YAML). Raw OTLP archives to a local file
# (D13's own "raw OTLP archived"; the compose/helm deployment mounts this path to a volume an
# operator can ship to object storage, not built by this task). Redaction runs exactly ONCE,
# upstream of every fan out target below, so neither Phoenix nor the archive ever sees an unredacted
# span.

receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  redaction:
    allow_all_keys: false
    allowed_keys:
{keys_block}
    ignored_keys: []
    summary: debug

exporters:
  otlp/phoenix:
    endpoint: phoenix:4317
    tls:
      insecure: true
  file/raw_archive:
    path: /var/log/otel/raw-otlp-archive.jsonl
    format: json

  # D13: LangSmith and LangWatch exist only as commented configuration here, proving pluggability
  # without ever being a live export target. This plan's Global Constraints: neither package/SDK may
  # ever appear in pyproject.toml; a Go binary's own YAML is the only place either name may appear
  # at all.
  # otlphttp/langsmith:
  #   endpoint: https://api.smith.langchain.com/otel/v1/traces
  #   headers:
  #     x-api-key: ${{LANGSMITH_API_KEY}}
  #     Langsmith-Project: atlas
  # otlphttp/langwatch:
  #   endpoint: https://app.langwatch.ai/api/otel/v1/traces
  #   headers:
  #     Authorization: Bearer ${{LANGWATCH_API_KEY}}

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [redaction]
      exporters: [otlp/phoenix, file/raw_archive]
      # LangSmith/LangWatch stay out of this list too: adding either here would be the actual
      # pluggability step, deliberately not taken (D13: Phoenix is the only DEPLOYED backend).
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true",
        help="write the current collector config to infra/observability/otel-collector.yaml",
    )
    args = parser.parse_args(argv)

    fresh = render_collector_config()
    if args.write:
        COLLECTOR_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        COLLECTOR_CONFIG_PATH.write_text(fresh)
        print(f"wrote {COLLECTOR_CONFIG_PATH}")
        return 0

    committed = COLLECTOR_CONFIG_PATH.read_text() if COLLECTOR_CONFIG_PATH.is_file() else None
    if committed == fresh:
        print("otel-collector.yaml: unchanged")
        return 0
    print(f"otel-collector.yaml: DRIFTED from {COLLECTOR_CONFIG_PATH} (run with --write to regenerate, then review the diff)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
