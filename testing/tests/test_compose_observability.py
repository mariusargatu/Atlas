"""Hermetic render test for docker-compose.yml's `observability` profile (SP6 task 3), mirroring
`test_infra_manifests.py`'s own "render, zero cluster, zero credentials" discipline: `docker compose
config` needs no daemon running and makes no network call, so this proves what the profile gate,
image digests, and backend wiring actually resolve to, not merely what the YAML looks like on a
read.

Two things this task's compose work must hold, both asserted here:
  1. `otel-collector`/`phoenix`/`phoenix-db-init`/`otel-archive-init` are OFF by default (a plain
     `docker compose config` never resolves them) and ON only behind `--profile observability` (D13/
     this task's own "off by default" requirement).
  2. every image reference added is digest shaped (D37); Phoenix's storage is the existing postgres
     service, a new `phoenix` database, never SQLite and never a second Postgres instance (SP6
     digest question 5). `otel-archive-init` itself is a fix found live, not originally planned: the
     collector image runs as a fixed non root UID that cannot write into a freshly created named
     volume without this ownership fix running first.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

_DIGEST_RE_OTEL = "otel/opentelemetry-collector-contrib@sha256:"
_DIGEST_RE_PHOENIX = "arizephoenix/phoenix@sha256:"
_DIGEST_RE_PGVECTOR = "pgvector/pgvector@sha256:"
_DIGEST_RE_BUSYBOX = "busybox@sha256:"


def _docker_compose_missing_reason() -> str | None:
    if shutil.which("docker") is None:
        return "docker is not on PATH: this hermetic render test needs `docker compose config` (zero daemon, zero network)."
    probe = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True, timeout=15)
    if probe.returncode != 0:
        return "`docker compose` is not available (the docker CLI is present but the compose plugin is not)."
    return None


@pytest.fixture(scope="module", autouse=True)
def _require_docker_compose():
    reason = _docker_compose_missing_reason()
    if reason is not None:
        pytest.skip(reason)


def _config(*, profile: str | None) -> dict:
    # A clean, hermetic env: an operator's own exported MODEL_PROVIDER/ATLAS_TRACING/etc. (from a
    # local .env sourced into their shell) must never leak into what this test asserts, the same
    # discipline test_infra_manifests.py's own _render() holds itself to for its env vars.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ATLAS_", "MODEL_", "GIT_SHA", "OLLAMA_"))}
    env.setdefault("PATH", os.environ.get("PATH", ""))
    args = ["docker", "compose"]
    if profile is not None:
        args += ["--profile", profile]
    args += ["config", "--format", "json"]
    result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"docker compose config failed:\n{result.stdout}\n{result.stderr}"
    return json.loads(result.stdout)


def test_default_profile_never_resolves_the_observability_services():
    services = _config(profile=None)["services"]
    assert "otel-collector" not in services
    assert "phoenix" not in services
    assert "phoenix-db-init" not in services
    assert "otel-archive-init" not in services


def test_default_profile_backend_tracing_is_off_by_default():
    backend = _config(profile=None)["services"]["backend"]
    assert backend["environment"]["ATLAS_TRACING"] == ""


def test_observability_profile_resolves_the_four_new_services():
    services = _config(profile="observability")["services"]
    assert {"otel-collector", "phoenix", "phoenix-db-init", "otel-archive-init"} <= set(services)


def test_otel_collector_waits_on_the_archive_volume_ownership_fix_first():
    """Found live: otel/opentelemetry-collector-contrib runs as a fixed non root UID (10001) baked
    into the image; a freshly created named volume mounts root owned, which that UID cannot write --
    the collector fails closed (restart loops on "permission denied") without this ordering."""
    collector = _config(profile="observability")["services"]["otel-collector"]
    assert collector["depends_on"]["otel-archive-init"]["condition"] == "service_completed_successfully"


def test_observability_profile_images_are_digest_shaped():
    services = _config(profile="observability")["services"]
    assert services["otel-collector"]["image"].startswith(_DIGEST_RE_OTEL)
    assert services["phoenix"]["image"].startswith(_DIGEST_RE_PHOENIX)
    assert services["phoenix-db-init"]["image"].startswith(_DIGEST_RE_PGVECTOR)
    assert services["otel-archive-init"]["image"].startswith(_DIGEST_RE_BUSYBOX)


def test_phoenix_is_backed_by_the_existing_postgres_service_in_a_new_database():
    services = _config(profile="observability")["services"]
    phoenix = services["phoenix"]
    dsn = phoenix["environment"]["PHOENIX_SQL_DATABASE_URL"]
    assert dsn.startswith("postgresql://")
    assert "@postgres:5432/phoenix" in dsn  # the SAME postgres service, a new "phoenix" database
    assert "sqlite" not in dsn.lower()
    # the init job creates that database against the SAME postgres service, never a second instance
    db_init = services["phoenix-db-init"]
    assert "postgres" in db_init["depends_on"]
    assert "CREATE DATABASE phoenix" in " ".join(db_init["entrypoint"])


def test_phoenix_depends_on_its_db_init_job_completing_first():
    phoenix = _config(profile="observability")["services"]["phoenix"]
    assert phoenix["depends_on"]["phoenix-db-init"]["condition"] == "service_completed_successfully"


def test_otel_collector_mounts_the_committed_redaction_config_read_only():
    collector = _config(profile="observability")["services"]["otel-collector"]
    mounts = {v["target"]: v for v in collector["volumes"] if v["type"] == "bind"}
    assert "/etc/otelcol-contrib/otel-collector.yaml" in mounts
    mount = mounts["/etc/otelcol-contrib/otel-collector.yaml"]
    assert mount["read_only"] is True
    assert mount["source"].endswith("infra/observability/otel-collector.yaml")


def test_otel_collector_archives_raw_otlp_to_a_named_volume():
    collector = _config(profile="observability")["services"]["otel-collector"]
    volume_mounts = [v for v in collector["volumes"] if v["type"] == "volume"]
    assert any(v["source"] == "otel-raw-archive" and v["target"] == "/var/log/otel" for v in volume_mounts)


def test_backend_otel_endpoint_points_at_the_in_network_collector_by_default():
    backend = _config(profile="observability")["services"]["backend"]
    assert backend["environment"]["ATLAS_OTEL_ENDPOINT"] == "http://otel-collector:4318"


def test_backend_tei_urls_stay_parity_preserving_when_unset():
    """The interface note's own "MAY convert to passthroughs" requirement: unset env vars must still
    resolve to the exact in network service names a plain `docker compose up` always used, both with
    and without the observability profile active."""
    for profile in (None, "observability"):
        backend = _config(profile=profile)["services"]["backend"]
        assert backend["environment"]["ATLAS_TEI_EMBED_URL"] == "http://tei-embed:80"
        assert backend["environment"]["ATLAS_TEI_RERANK_URL"] == "http://tei-rerank:80"


def test_backend_tei_urls_are_overridable_via_the_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ATLAS_", "MODEL_", "GIT_SHA", "OLLAMA_"))}
    env["ATLAS_TEI_EMBED_URL"] = "http://87.99.155.117:8080"
    env["ATLAS_TEI_RERANK_URL"] = "http://87.99.155.117:8081"
    result = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"docker compose config failed:\n{result.stdout}\n{result.stderr}"
    backend = json.loads(result.stdout)["services"]["backend"]
    assert backend["environment"]["ATLAS_TEI_EMBED_URL"] == "http://87.99.155.117:8080"
    assert backend["environment"]["ATLAS_TEI_RERANK_URL"] == "http://87.99.155.117:8081"
