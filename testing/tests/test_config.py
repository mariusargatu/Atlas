"""AtlasSettings: the typed settings module (SP6 task 1, D10's narrower first cut).

`from_env()` round trips every env read `server.py`'s own scattered `os.environ.get` calls made
directly, plus the reads `checkpointer_kind`/`select_retriever`/`PgvectorRetriever` already make
themselves (`ATLAS_CHECKPOINTER`, `ATLAS_RETRIEVER`, TEI urls, index dir) and the new opt in tracing
flag this task introduces. `config_hash()` is what `atlas.config.hash` (one of the reserved trace
attributes, `contract_tools.loader.RESERVED_TRACE_ATTRIBUTES`) carries once a real tracer exists
(the OTel adapter skeleton stamps it on the turn span already; see test_otel_tracer.py).
"""
from __future__ import annotations

import json

import pytest

from atlas.config import AtlasSettings

_ALL_ENV_VARS = (
    "ATLAS_MODE", "MODEL_PROVIDER", "ATLAS_CASSETTES", "GIT_SHA", "ATLAS_CHECKPOINTER",
    "ATLAS_RETRIEVER", "ATLAS_TEI_EMBED_URL", "ATLAS_TEI_RERANK_URL", "ATLAS_INDEX_DIR",
    "ATLAS_TRACING", "ATLAS_OTEL_ENDPOINT", "ATLAS_PG_DSN", "ATLAS_FALLBACK_MODEL",
    "ATLAS_REGISTRY_VERSION",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_from_env_defaults_match_every_scattered_read_today():
    settings = AtlasSettings.from_env()
    assert settings.atlas_mode == "replay"
    assert settings.model_provider == "ollama"
    assert settings.git_sha == "dev"
    assert settings.checkpointer_kind == "inmemory"
    assert settings.retriever_kind == "inmemory"
    assert settings.tei_embed_url == "http://localhost:8081"
    assert settings.tei_rerank_url == "http://localhost:8082"
    assert settings.tracing == ""
    assert settings.pg_dsn is None
    assert settings.fallback_model == ""
    assert settings.registry_version == ""
    assert settings.cassette_dir.endswith("testing/harness/cassettes/e2e")
    assert settings.index_dir.endswith("indexes/corpus-0.1.1-bge-m3-03f983e0")
    # SP6 task 6: derived from the default index dir's own committed build_manifest.json, not a new
    # env var; this repo ships that file, so the default settings object already names a real build.
    assert settings.corpus_version == "corpus-0.1.1"
    assert settings.index_build_id == "a86bc176d5bf7d04"


def test_from_env_round_trips_every_covered_var(monkeypatch):
    monkeypatch.setenv("ATLAS_MODE", "live")
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("ATLAS_CASSETTES", "/tmp/cassettes")
    monkeypatch.setenv("GIT_SHA", "deadbeef")
    monkeypatch.setenv("ATLAS_CHECKPOINTER", "postgres")
    monkeypatch.setenv("ATLAS_RETRIEVER", "pgvector")
    monkeypatch.setenv("ATLAS_TEI_EMBED_URL", "http://tei-embed:8081")
    monkeypatch.setenv("ATLAS_TEI_RERANK_URL", "http://tei-rerank:8082")
    monkeypatch.setenv("ATLAS_INDEX_DIR", "/data/index")
    monkeypatch.setenv("ATLAS_TRACING", "otel")
    monkeypatch.setenv("ATLAS_OTEL_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("ATLAS_PG_DSN", "postgresql://u:p@host/db")
    monkeypatch.setenv("ATLAS_FALLBACK_MODEL", "anthropic:claude-haiku-4-5-20251001")
    monkeypatch.setenv("ATLAS_REGISTRY_VERSION", "corpus-0.1.1")

    settings = AtlasSettings.from_env()
    assert settings == AtlasSettings(
        atlas_mode="live", model_provider="anthropic", cassette_dir="/tmp/cassettes",
        git_sha="deadbeef", checkpointer_kind="postgres", retriever_kind="pgvector",
        tei_embed_url="http://tei-embed:8081", tei_rerank_url="http://tei-rerank:8082",
        index_dir="/data/index", tracing="otel", otel_endpoint="http://collector:4318",
        pg_dsn="postgresql://u:p@host/db", fallback_model="anthropic:claude-haiku-4-5-20251001",
        registry_version="corpus-0.1.1",
        # "/data/index" does not exist on this machine, so the manifest read comes up empty and
        # both derived fields fall back to their class default (""), exactly what from_env() itself
        # resolves for the same nonexistent path -- see test_corpus_version_and_index_build_id_...
        # below for the case where the directory IS real.
        corpus_version="", index_build_id="",
    )


def test_settings_cannot_be_mutated_after_construction():
    settings = AtlasSettings.from_env()

    def _mutate():
        settings.atlas_mode = "live"

    with pytest.raises(Exception):
        _mutate()


def test_config_hash_is_stable_for_the_same_resolved_settings():
    a = AtlasSettings(atlas_mode="replay")
    b = AtlasSettings(atlas_mode="replay")
    assert a.config_hash() == b.config_hash() and a is not b


def test_config_hash_is_a_sha256_hex_digest():
    h = AtlasSettings().config_hash()
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_config_hash_changes_when_a_behaviour_affecting_field_changes():
    a = AtlasSettings(atlas_mode="replay")
    b = AtlasSettings(atlas_mode="live")
    assert a.config_hash() != b.config_hash()


def test_config_hash_changes_when_retriever_kind_changes():
    a = AtlasSettings(retriever_kind="inmemory")
    b = AtlasSettings(retriever_kind="pgvector")
    assert a.config_hash() != b.config_hash()


def test_config_hash_excludes_the_pg_dsn_secret_from_the_preimage():
    a = AtlasSettings(pg_dsn="postgresql://u:p@host/db")
    b = AtlasSettings(pg_dsn=None)
    assert a.config_hash() == b.config_hash()


def test_config_hash_excludes_git_sha_identity_from_the_preimage():
    a = AtlasSettings(git_sha="abc123")
    b = AtlasSettings(git_sha="def456")
    assert a.config_hash() == b.config_hash()


def test_config_hash_changes_when_fallback_model_is_set(monkeypatch):
    """`ATLAS_FALLBACK_MODEL` picks which second model the ladder's `provider_fallback` rung falls
    back to (`server.py`'s `_fallback_gateway`) in live/record mode -- a genuine behaviour fork that
    belongs in the identity, not a secret or pure identity field."""
    monkeypatch.delenv("ATLAS_FALLBACK_MODEL", raising=False)
    unset = AtlasSettings.from_env()
    monkeypatch.setenv("ATLAS_FALLBACK_MODEL", "anthropic:claude-haiku-4-5-20251001")
    set_to_a_model = AtlasSettings.from_env()
    assert unset.config_hash() != set_to_a_model.config_hash()


def test_config_hash_treats_unset_and_empty_fallback_model_as_the_same_absent_value(monkeypatch):
    """Mirrors `_fallback_gateway`'s own `if not raw: return None` -- unset and an explicit empty
    string both mean absent, so the settings value and the gateway must agree on what "absent" is,
    and the hash must not distinguish the two."""
    monkeypatch.delenv("ATLAS_FALLBACK_MODEL", raising=False)
    unset = AtlasSettings.from_env()
    monkeypatch.setenv("ATLAS_FALLBACK_MODEL", "")
    empty = AtlasSettings.from_env()
    assert unset.fallback_model == empty.fallback_model == ""
    assert unset.config_hash() == empty.config_hash()


def test_config_hash_changes_when_registry_version_changes():
    """`registry_version` (SP6 task 5, `ATLAS_REGISTRY_VERSION`) is a behaviour affecting field: it
    is what `/metrics`' own `atlas_corpus_staleness` gauge compares against the active index's
    `corpus_version`, so two settings that disagree on it must hash differently, the same identity
    discipline `retriever_kind`/`atlas_mode` already hold themselves to above."""
    a = AtlasSettings(registry_version="")
    b = AtlasSettings(registry_version="corpus-0.1.1")
    assert a.config_hash() != b.config_hash()


def test_from_env_captures_the_fallback_model(monkeypatch):
    """Round trip proof independent of `test_from_env_round_trips_every_covered_var`'s combined
    assertion: `ATLAS_FALLBACK_MODEL` alone lands on `AtlasSettings.fallback_model` unchanged."""
    monkeypatch.setenv("ATLAS_FALLBACK_MODEL", "anthropic:claude-haiku-4-5-20251001")
    settings = AtlasSettings.from_env()
    assert settings.fallback_model == "anthropic:claude-haiku-4-5-20251001"


# ---- corpus_version / index_build_id (SP6 task 6, D37): surfaced from the active index's own -----
# ---- build_manifest.json off ATLAS_INDEX_DIR, never a new env var, never computed at /version time.


def test_corpus_version_and_index_build_id_are_read_from_the_active_index_manifest(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "build_manifest.json").write_text(
        json.dumps({"corpus_version": "corpus-9.9.9", "index_build_id": "deadbeef01234567"})
    )
    monkeypatch.setenv("ATLAS_INDEX_DIR", str(index_dir))
    settings = AtlasSettings.from_env()
    assert settings.corpus_version == "corpus-9.9.9"
    assert settings.index_build_id == "deadbeef01234567"


def test_corpus_version_and_index_build_id_default_to_empty_when_the_manifest_is_missing(tmp_path, monkeypatch):
    """An operator running against an index dir that was never built (or a typo'd path) must still
    boot: absence of a signal here is never a crash, the same lenient discipline
    `metrics._corpus_staleness` already holds itself to for this exact file."""
    monkeypatch.setenv("ATLAS_INDEX_DIR", str(tmp_path / "never-built"))
    settings = AtlasSettings.from_env()
    assert settings.corpus_version == ""
    assert settings.index_build_id == ""


def test_corpus_version_and_index_build_id_default_to_empty_on_a_malformed_manifest(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "build_manifest.json").write_text("not valid json")
    monkeypatch.setenv("ATLAS_INDEX_DIR", str(index_dir))
    settings = AtlasSettings.from_env()
    assert settings.corpus_version == ""
    assert settings.index_build_id == ""


def test_config_hash_changes_when_corpus_version_changes():
    """corpus_version/index_build_id are identity of WHAT is served (this task's own adjudication,
    unlike `git_sha`, which names WHICH build produced the process): included in the preimage, not
    excluded alongside the secret/identity fields above."""
    a = AtlasSettings(corpus_version="corpus-0.1.1")
    b = AtlasSettings(corpus_version="corpus-0.2.0")
    assert a.config_hash() != b.config_hash()


def test_config_hash_changes_when_index_build_id_changes():
    a = AtlasSettings(index_build_id="a86bc176d5bf7d04")
    b = AtlasSettings(index_build_id="deadbeef01234567")
    assert a.config_hash() != b.config_hash()
