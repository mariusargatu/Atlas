"""Typed settings (SP6 task 1): `AtlasSettings.from_env()` reads every scattered `os.environ.get`
call `server.py` made directly, plus the reads other modules already make themselves
(`checkpointer_kind`, `select_retriever`, `PgvectorRetriever`'s own TEI url / index dir lookups), so
one dataclass is the readable record of what a process resolved at startup, and `config_hash()` is a
stable identity for it.

Narrower than D10's stated scope ("one typed settings module is the only configuration entry
point"): this first cut covers the reads enumerated on the dataclass below. `PgvectorRetriever`,
`checkpointer_kind`, and `select_retriever` still read their OWN env vars directly and keep
validating a typo exactly as they do today (`server.py` now passes them the value already resolved
instead of letting them read the process environment again, but the validation itself is untouched);
this module only replaces WHERE the read happens, not the fail fast contract each consumer already
enforces. Grow this dataclass in place as more of D10's scope lands; do not fork a second settings
type.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

from determinism.canonical import digest

# Anchored on this file (not the CWD), the same pattern server.py's own _DEFAULT_CASSETTES already
# uses, so a clone works from any directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CASSETTES = str(_REPO_ROOT / "testing" / "harness" / "cassettes" / "e2e")

# The local development defaults, PUBLIC and named once here because more than one module needs
# each of them: `adapters/pgvector_retriever.py` and `adapters/pg_knowledge_graph.py` both fall back
# to the DSN, and the retriever also needs both TEI urls and the index dir. Each of those modules
# used to carry its own private copy of the same literal, so the DSN alone was typed out in nine
# Python files. The env var each maps to and the fail fast behaviour of each consumer are unchanged:
# this names the DEFAULT once, it does not centralise the read.
#
# The password is a compose local development credential (`docker-compose.yml`'s own
# POSTGRES_PASSWORD), never a real secret; `pg_dsn` is still excluded from `config_hash()` below
# because a deployment overrides it with one that is.
DEFAULT_PG_DSN = "postgresql://atlas:atlas-dev-password@localhost:5433/atlas"
DEFAULT_TEI_EMBED_URL = "http://localhost:8081"
DEFAULT_TEI_RERANK_URL = "http://localhost:8082"
# Today's one committed index build. A real deployment points ATLAS_INDEX_DIR elsewhere; this
# default only keeps a bare `PgvectorRetriever()` construction working against the repo as checked
# out, and gives `AtlasSettings.index_dir` the same fallback the adapter uses.
DEFAULT_INDEX_DIR = str(_REPO_ROOT / "indexes" / "corpus-0.1.1-bge-m3-03f983e0")

# config_hash()'s preimage excludes exactly these two categories, named here once so the exclusion
# rule and the field list cannot drift apart:
#   - secrets: `pg_dsn` carries database credentials. A hash that changes only because a password
#     rotated would be a false "config changed" alert, and a hash is not a safe place to fingerprint
#     a credential regardless.
#   - identity: `git_sha` names WHICH build produced this process, not how it behaves. Two
#     deployments with identical resolved settings must hash identically regardless of which commit
#     produced them, so a deploy that changes no behaviour never trips a spurious drift alert.
_SECRET_FIELDS = frozenset({"pg_dsn"})
_IDENTITY_FIELDS = frozenset({"git_sha"})


def _read_index_manifest(index_dir: str) -> dict:
    """SP6 task 6 (D37): `corpus_version`/`index_build_id` are surfaced from the active index's own
    `build_manifest.json`, off whatever `index_dir` this settings object already resolved -- the
    SAME file `metrics._corpus_staleness` already reads for the same directory, so `/version` and
    `/metrics` agree on one authoritative source, never two.

    Not `fingerprint.json` (that file carries only embedding identity: `dim`/`model_id`/`revision`/
    `provider`/`normalize`, no `corpus_version` or `index_build_id` field at all) and not the
    directory name (`indexes/corpus-0.1.1-bge-m3-03f983e0` encodes `corpus_version` and a chunker
    hash PREFIX, but never the build id itself -- this repo's own committed build carries
    `index_build_id=a86bc176d5bf7d04`, unrelated to the `03f983e0` suffix in its directory name, so
    parsing the name would silently return the wrong id for the one field that most needs to be
    exact). Lenient by design, mirroring `metrics._corpus_staleness`'s own read: a missing or
    unreadable manifest yields an empty dict, never a raised error -- an operator running with a not
    yet built index must still be able to boot and read `/healthz`/`/version`.
    """
    manifest_path = Path(index_dir) / "build_manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


@dataclass(frozen=True)
class AtlasSettings:
    """Every env read this task threads through `server.py`, one frozen record per process.

    `atlas_mode`/`model_provider`/`cassette_dir`/`git_sha` are `server.py`'s own former direct
    reads (`ATLAS_MODE`, `MODEL_PROVIDER`, `ATLAS_CASSETTES`, `GIT_SHA`). `checkpointer_kind`/
    `retriever_kind` mirror `ATLAS_CHECKPOINTER`/`ATLAS_RETRIEVER`, resolved here so they
    participate in `config_hash()`'s identity even though `checkpointer_kind()`/`select_retriever()`
    still own the actual validation. `tei_embed_url`/`tei_rerank_url`/`index_dir` mirror
    `PgvectorRetriever`'s own env reads for the same reason -- captured here for identity, not yet
    threaded into that adapter's construction (a later task's concern once this module needs to
    grow that far). `tracing`/`otel_endpoint` are new: `ATLAS_TRACING`/`ATLAS_OTEL_ENDPOINT` back
    `server.py`'s opt in gate (`_tracer()`) for the sibling `OtelTracer` adapter. `pg_dsn`
    (`ATLAS_PG_DSN`) is the one secret this first cut carries. `fallback_model` mirrors
    `server.py`'s `_fallback_gateway` own read of `ATLAS_FALLBACK_MODEL` -- captured here for
    identity only, not yet threaded into that function's own construction, the same "captured but
    not threaded" boundary the TEI urls and index dir already draw. Empty string and unset both
    normalize to `""` (`from_env`'s default), matching `_fallback_gateway`'s own `if not raw:`
    treatment, so the settings value and the gateway agree on what "absent" means. `registry_version`
    (`ATLAS_REGISTRY_VERSION`, SP6 task 5) is the operator asserted "current" fact registry version
    (D29: "registry_version vs ingested corpus_version staleness gauge"); `atlas.metrics.render`
    compares it against the active index's own `build_manifest.json:corpus_version` to compute
    `atlas_corpus_staleness`. Unset (the default) means "no comparison configured," which reads as
    fresh (0), never a false drift alarm.

    `corpus_version`/`index_build_id` (SP6 task 6, D37) are NOT their own env reads: `from_env()`
    derives them from `index_dir`'s own `build_manifest.json` (`_read_index_manifest`, this module's
    own honest source, chosen over `fingerprint.json` and the directory name; see that function's
    docstring), so `/version` surfaces them rather than computing anything itself. Adjudicated as
    identity of WHAT this process serves, not identity of WHICH build produced the process (that is
    `git_sha`'s job): both fields participate in `config_hash()`'s preimage below, unlike `git_sha`.
    """

    atlas_mode: str = "replay"
    model_provider: str = "ollama"
    cassette_dir: str = _DEFAULT_CASSETTES
    git_sha: str = "dev"
    checkpointer_kind: str = "inmemory"
    retriever_kind: str = "inmemory"
    tei_embed_url: str = DEFAULT_TEI_EMBED_URL
    tei_rerank_url: str = DEFAULT_TEI_RERANK_URL
    index_dir: str = DEFAULT_INDEX_DIR
    corpus_version: str = ""
    index_build_id: str = ""
    tracing: str = ""
    otel_endpoint: str = "http://localhost:4318"
    pg_dsn: str | None = None
    fallback_model: str = ""
    registry_version: str = ""

    @classmethod
    def from_env(cls) -> "AtlasSettings":
        """Reads the process environment ONCE. No validation here: each value's own consumer
        (`_resolve_mode`, `_require_provider_sdk`, `checkpointer_kind`, `select_retriever`, ...)
        still fails fast on a typo exactly as it does today -- this classmethod only relocates
        WHERE the read happens, mechanical substitution, never the behaviour that follows it.

        `corpus_version`/`index_build_id` are the one pair NOT read from an env var directly: they
        are derived from the JUST resolved `index_dir`'s own `build_manifest.json`
        (`_read_index_manifest`), so ATLAS_INDEX_DIR stays the single knob that moves both the
        retriever's index AND this settings object's own idea of what corpus/build that index is."""
        index_dir = os.environ.get("ATLAS_INDEX_DIR", cls.index_dir)
        manifest = _read_index_manifest(index_dir)
        return cls(
            atlas_mode=os.environ.get("ATLAS_MODE", cls.atlas_mode),
            model_provider=os.environ.get("MODEL_PROVIDER", cls.model_provider),
            cassette_dir=os.environ.get("ATLAS_CASSETTES", cls.cassette_dir),
            git_sha=os.environ.get("GIT_SHA", cls.git_sha),
            checkpointer_kind=os.environ.get("ATLAS_CHECKPOINTER", cls.checkpointer_kind),
            retriever_kind=os.environ.get("ATLAS_RETRIEVER", cls.retriever_kind),
            tei_embed_url=os.environ.get("ATLAS_TEI_EMBED_URL", cls.tei_embed_url),
            tei_rerank_url=os.environ.get("ATLAS_TEI_RERANK_URL", cls.tei_rerank_url),
            index_dir=index_dir,
            corpus_version=manifest.get("corpus_version", cls.corpus_version),
            index_build_id=manifest.get("index_build_id", cls.index_build_id),
            tracing=os.environ.get("ATLAS_TRACING", cls.tracing),
            otel_endpoint=os.environ.get("ATLAS_OTEL_ENDPOINT", cls.otel_endpoint),
            pg_dsn=os.environ.get("ATLAS_PG_DSN", cls.pg_dsn),
            fallback_model=os.environ.get("ATLAS_FALLBACK_MODEL", cls.fallback_model),
            registry_version=os.environ.get("ATLAS_REGISTRY_VERSION", cls.registry_version),
        )

    def config_hash(self) -> str:
        """sha256 over the canonical JSON of the BEHAVIOUR AFFECTING fields only (ADR-007's own
        canonicalization, `determinism.canonical.digest`, the same one the cassette key and run
        digest already use, so a config identity and a cassette key are one rule, not two that can
        quietly drift). Secrets (`pg_dsn`) and pure identity (`git_sha`) are excluded from the
        preimage; see the module level `_SECRET_FIELDS`/`_IDENTITY_FIELDS` docstring for why.
        `corpus_version`/`index_build_id` are deliberately NOT excluded (SP6 task 6's own
        adjudication): they name what corpus/index a response was actually grounded against, a real
        behaviour affecting fact, not a mere build label.

        Narrower than D10's full scope: today's preimage is exactly the fields this dataclass
        covers right now (see its own docstring); it grows automatically as the dataclass grows,
        never redefined by hand elsewhere."""
        excluded = _SECRET_FIELDS | _IDENTITY_FIELDS
        preimage = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in excluded}
        return digest(preimage)


__all__ = ["AtlasSettings"]
