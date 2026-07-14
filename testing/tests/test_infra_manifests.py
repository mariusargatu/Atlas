"""Hermetic chart render test for the infra/ helmfile scaffold (SP5 task 1).

Renders every currently defined release with `helmfile -e <environment> template`, zero cluster,
zero credentials, exactly what `helm template`/`helmfile template` already guarantee: no `--kube-
context`, no API server reachability, pure local rendering. This is the render test the plan asks
for, not a smoke test against a live k3d cluster (that is `task k3d:smoke`, later tasks, not the PR
lane).

Two things this task's scaffold proves, both asserted here:
  1. the environment ladder resolves for both `local` and `burst` (D3: "one set of charts", a
     a difference in values, not a difference in chart content);
  2. the SOPS + age mechanism is real, not faked down to one recipient: the committed
     `infra/environments/local/secrets.enc.yaml` decrypts through helmfile's own `vals` `ref+sops://`
     integration during rendering, not via a side channel the render path never actually exercises
     (D41: "a single age key must not be the root secret of the entire system").

`EXPECTED_TEI_DIGESTS` is the deliberate growth point: Task 3 adds the TEI releases and populates it
with the exact `--revision` values models.lock pins, at which point
`test_tei_digests_pinned_in_models_lock_appear_once_those_releases_exist` starts asserting for real
instead of asserting nothing. No rewrite of this file is needed when that happens.

Task 2 adds the CNPG operator + cluster assertions below. The operator chart is vendored locally
(`infra/charts/vendor/cloudnative-pg`, stripped of its optional network dependency, see that chart's
own Chart.yaml comment) specifically so this stays a zero network render like everything else here;
a regression back to a live Helm repository dependency would show up here as a `helmfile template`
failure, not a silent skip.

Task 3 adds the TEI embed/rerank story (generic `charts/tei`, rendered twice via `.Values.role`, D3:
one set of charts) and the `charts/atlas-indexes` PersistentVolume/PersistentVolumeClaim pair that
replaces atlas-jobs' Task 2 raw hostPath mount. TEI itself is gated by `tei.mode`: "inCluster" (what
burst's real amd64 nodes use) renders the actual Deployment/Service/HF cache PVC with the pinned
digest and probes; "external" (local's own override, found live: an arm64 k3d dev machine cannot run
this amd64 only image reliably, full diagnosis in environments/local/values.yaml's own comment)
renders a connectivity-check Job against a real external endpoint instead. These assertions parse
rendered documents with `yaml.safe_load_all` rather than raw substring search where precision matters
(probe timing, the PV/PVC vs. hostPath distinction), because a `PersistentVolume` legitimately
contains its own `hostPath` block, which would make a bare "hostPath absent" substring check vacuous.

Task 5 claims the two SP5 reserved helmfile slots: `otel-collector`/`phoenix` (SP6 task 3's own
"park, not claim" boundary, `test_observability_values.py`) and the prometheus-operator/kube-
prometheus stack slot (SP5 digest open decision 11). `infra/charts/atlas-monitoring`'s own
Chart.yaml names the scope boundary explicitly: a hand authored Prometheus + Alertmanager +
Pushgateway, not a vendored `prometheus-community/kube-prometheus-stack`, with PrometheusRule CRD
objects that render hermetically (no live API server is ever contacted by `helm`/`helmfile
template`) and, separately, reach the ACTUALLY RUNNING Prometheus this release deploys as a native
rule file rendered off the exact same `.Values.rules.groups` (see
`test_prometheusrule_crd_and_the_native_rule_configmap_never_drift` below) -- the D29 rule set is
genuinely live and alerting in this reference deployment, not merely committed YAML waiting on
infrastructure this task does not install.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
INFRA_DIR = ROOT / "infra"

# D41: two age recipients, private keys never committed. `infra/README.md` documents this location;
# the render test defaults to it (a bare `sops -d`/`helmfile template` on an operator's own machine
# needs no extra env var), but never overrides an operator's own SOPS_AGE_KEY_FILE if already set.
DEFAULT_AGE_KEY_FILE = pathlib.Path.home() / ".config" / "atlas-age" / "keys.txt"

# Task 3: the exact BAAI/bge-m3 and BAAI/bge-reranker-v2-m3 `--revision` pins from models.lock,
# copied verbatim (D26: never a floating alias), now that the tei-embed/tei-rerank releases exist.
EXPECTED_TEI_DIGESTS: tuple[str, ...] = (
    "5617a9f61b028005a4858fdac845db406aefb181",  # BAAI/bge-m3
    "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",  # BAAI/bge-reranker-v2-m3
)

# D37: every image reference in a values file is digest shaped (repo@sha256:<64 hex chars>), never a
# floating tag. Matches the placeholder (all zero sha256) committed today as much as a real digest
# `task k3d:up` records after building and pushing: the render test asserts the SHAPE, which never
# goes stale, not a specific value, which would go stale on the very next local rebuild.
_DIGEST_SHAPED_IMAGE_RE = re.compile(r"@sha256:[0-9a-f]{64}\b")

# ride along (a), Task 1 review: sops's own failure text for "no private key present that can
# decrypt this file" always contains this substring (verified against a real `SOPS_AGE_KEY_FILE=
# /dev/null` run: "Error getting data key: 0 successful groups required, got 0"). Special cased so
# the render test's own failure message self diagnoses the single most likely local cause (a missing
# or misconfigured age identity) instead of leaving a bare helmfile stack trace for the next reader
# to decode by hand.
_MISSING_AGE_KEY_STDERR_SUBSTRING = "data key"

_AGE_KEY_SETUP_HINT = (
    "This looks like a missing or misconfigured age identity, not a chart bug: sops could not find "
    "a private key able to decrypt infra/environments/local/secrets.enc.yaml. See infra/README.md's "
    "'SOPS + age: two recipients, mechanism proven now' section to provision "
    "~/.config/atlas-age/keys.txt (or point SOPS_AGE_KEY_FILE at your own identity file)."
)


def _helmfile_missing_reason() -> str | None:
    if shutil.which("helmfile") is not None:
        return None
    return (
        "helmfile is not on PATH: the hermetic chart render test needs it to render infra/ "
        "manifests with zero cluster and zero credentials (`helmfile template`). Install helmfile "
        "(https://helmfile.readthedocs.io) to run this test locally; a CI image that lacks it should "
        "install it rather than rely on this skip staying in place indefinitely."
    )


@pytest.fixture(scope="module", autouse=True)
def _require_helmfile():
    reason = _helmfile_missing_reason()
    if reason is not None:
        pytest.skip(reason)


def _render(environment: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("SOPS_AGE_KEY_FILE", str(DEFAULT_AGE_KEY_FILE))
    # Task 3: environments/local/values.yaml's `tei.mode: external` pulls these from the environment
    # via helmfile's own `ref+envsubst://` value resolution. Fake, obviously not real defaults keep
    # this hermetic (no dependency on the gitignored .env.fastlane an operator provisions locally);
    # an operator's own exported values, if already set, win, same precedent as SOPS_AGE_KEY_FILE
    # above.
    env.setdefault("ATLAS_TEI_EMBED_URL", "http://tei-embed.render-test.invalid:8080")
    env.setdefault("ATLAS_TEI_RERANK_URL", "http://tei-rerank.render-test.invalid:8081")
    # Task 5: environments/burst/values.yaml's own certManager.baseDomain/acmeEmail, the same
    # ref+envsubst:// mechanism as the TEI URLs above, defaulted here for the same reason (hermetic,
    # no dependency on the gitignored .env.burst an operator provisions locally).
    env.setdefault("ATLAS_BURST_DOMAIN", "atlas.render-test.invalid")
    env.setdefault("ATLAS_BURST_ACME_EMAIL", "burst-ops@render-test.invalid")
    return subprocess.run(
        ["helmfile", "-e", environment, "template"],
        cwd=INFRA_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _failure_message(result: subprocess.CompletedProcess[str], environment: str) -> str:
    message = (
        f"helmfile -e {environment} template failed (exit {result.returncode}):\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    if _MISSING_AGE_KEY_STDERR_SUBSTRING in result.stderr.lower():
        message = f"{message}\n\n{_AGE_KEY_SETUP_HINT}"
    return message


def _documents(result: subprocess.CompletedProcess[str]) -> list[dict]:
    """Every rendered manifest as a parsed dict, `---` document boundaries and all. Used where a
    raw substring search would be imprecise or (worse) vacuously true, e.g. a `PersistentVolume`
    legitimately contains its own `hostPath` block, so telling it apart from a Job's volume that
    should no longer have one needs the actual document structure, not a grep across the whole stdout."""
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _find(documents: list[dict], *, kind: str, name: str) -> dict:
    for doc in documents:
        if doc.get("kind") == kind and doc.get("metadata", {}).get("name") == name:
            return doc
    raise AssertionError(f"no {kind} named {name!r} found in the rendered manifests")


def test_local_environment_renders_every_defined_release():
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    assert "kind: Namespace" in result.stdout
    assert "name: atlas" in result.stdout


def test_local_environment_decrypts_the_committed_sops_secret_through_helmfile():
    """A fake password, but a real round trip through encryption at rest: the committed secrets.enc.yaml
    only ever exists as ciphertext in git; this asserts the plaintext it decrypts to actually
    reaches a rendered manifest, not merely that `sops -d` works off to the side."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    assert "atlas-postgres-credentials" in result.stdout
    assert "atlas-dev-password" in result.stdout


def test_burst_environment_resolves_without_local_secrets():
    """The burst tier's real secrets are Task 5's job (credential gated); today it must still
    resolve as an environment, on the same helmfile and the same scaffold chart as local, and it
    must never inherit local's fake secret by accident."""
    result = _render("burst")
    assert result.returncode == 0, _failure_message(result, "burst")
    assert "kind: Namespace" in result.stdout
    assert "atlas-dev-password" not in result.stdout


def test_tei_digests_pinned_in_models_lock_appear_once_those_releases_exist():
    if not EXPECTED_TEI_DIGESTS:
        return  # Task 3 populates this; until then the assertion is correctly inert.
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    for digest in EXPECTED_TEI_DIGESTS:
        assert digest in result.stdout


# --- Task 2: CNPG operator + cluster ----------------------------------------------------------------


def test_local_environment_renders_the_cnpg_operator():
    """The operator chart is vendored locally (infra/charts/vendor/cloudnative-pg); this is the
    zero network render itself failing loud if that ever regresses back to a live Helm repository
    dependency, not a mock of the assertion."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    assert "kind: CustomResourceDefinition" in result.stdout
    assert "clusters.postgresql.cnpg.io" in result.stdout
    assert "imagecatalogs.postgresql.cnpg.io" in result.stdout
    assert "name: cnpg-operator-cloudnative-pg" in result.stdout


def test_local_environment_renders_the_cnpg_cluster_via_image_catalog():
    """D1: one Postgres, instances=1, no pooler. The Cluster CR references the ImageCatalog (never a
    bare imageName) so it always runs the pgvector layered image, not an upstream CNPG image that
    lacks the extension (SP5 digest section 2's flagged risk)."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    assert "kind: ImageCatalog" in result.stdout
    assert "kind: Cluster" in result.stdout
    assert "name: atlas-pg" in result.stdout
    assert "instances: 1" in result.stdout
    assert "kind: ImageCatalog\n    name: atlas-postgres-pgvector" in result.stdout


def test_cnpg_cluster_postinitsql_grants_createdb_to_the_owner_role_in_both_environments():
    """Fix round 1 (SP6 Task 5 review, Important finding 2): `cluster.yaml`'s postInitSQL block (SP6
    Task 5's own addition, a cross boundary touch from the phoenix db-init story) had no dedicated
    assertion, only the render tests already above, which check the chart still renders and
    check unrelated fields (instances, ImageCatalog wiring). Those pass even if the postInitSQL
    content were wrong or missing entirely, so they do not actually pin this behavior. This test
    parses the rendered Cluster manifest itself and asserts the exact grant line, in both
    environments (D3: one set of charts, so local and burst must render identically here).

    Origin, from cluster.yaml's own comment: the phoenix release's db-init Job (`CREATE DATABASE
    phoenix`) failed against a real k3d cluster with "permission denied to create database", because
    the "atlas" owner role CNPG creates has no CREATEDB privilege by default. postInitSQL runs as
    superuser in the "postgres" database before bootstrap.initdb's own database/owner exist, the
    documented place for a role level grant like this one.

    What this test does NOT prove: postInitSQL only ever runs once, at bootstrap time. The live
    k3d-atlas cnpg-cluster Helm release is at revision 1, bootstrapped before this grant was added,
    and was deliberately not recreated to avoid a destructive change against a shared, in use
    cluster. So nothing here, or anywhere else in this hermetic suite, confirms the grant actually
    resolves the phoenix permission error against a freshly bootstrapped cluster; that live
    verification is deferred to the SP6 final review, per the review's own request."""
    for environment in ("local", "burst"):
        result = _render(environment)
        assert result.returncode == 0, _failure_message(result, environment)
        documents = _documents(result)
        cluster = _find(documents, kind="Cluster", name="atlas-pg")
        post_init_sql = cluster["spec"]["bootstrap"]["initdb"]["postInitSQL"]
        assert post_init_sql == ["ALTER ROLE atlas CREATEDB;"], (
            f"{environment}: expected postInitSQL to be exactly the CREATEDB grant, got "
            f"{post_init_sql!r}"
        )


def test_local_environment_pgvector_and_backend_images_are_digest_shaped():
    """D37: every image reference is `repo@sha256:<64 hex>`, never a floating tag. This matches the
    committed placeholder (all zero sha256) as much as a real digest `task k3d:up` records after
    building and pushing, which is why this asserts the SHAPE via regex, not a literal value that
    would go stale the moment someone reruns the build locally."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    matches = _DIGEST_SHAPED_IMAGE_RE.findall(result.stdout)
    assert "atlas-registry:5000/atlas-postgres-pgvector@sha256:" in result.stdout
    assert "atlas-registry:5000/atlas-backend@sha256:" in result.stdout
    assert len(matches) >= 2, (
        f"expected at least 2 digest shaped (@sha256:<64 hex>) image refs (postgres-pgvector, "
        f"backend), found {len(matches)}. Full stdout:\n{result.stdout}"
    )


def test_local_environment_renders_the_one_shot_jobs():
    """Translated 1:1 from docker-compose.yml's checkpointer-migrate and rag-init services (compose
    parity, SP5 digest section 1): same backend image, same alembic/rag_tools.ingest invocations."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    assert "name: atlas-checkpointer-migrate" in result.stdout
    assert "name: atlas-rag-init" in result.stdout
    assert "alembic -c backend/atlas/persistence/alembic.ini upgrade head" in result.stdout
    assert "rag_tools.ingest" in result.stdout
    assert "--load-existing" in result.stdout


def test_burst_environment_renders_cnpg_without_leaking_local_image_digests():
    """Same charts as local (D3), placeholder burst values (Task 5 owns the real ones); must resolve
    cleanly and never accidentally pull in local's generated image digest block.

    Fix round 1 (reviewer finding): the negative assertion used to check for the literal string
    "k3d-atlas-registry", which never appears ANYWHERE in this repo (the real k3d local registry
    container is named plain "atlas-registry", verified live against a real cluster in Task 2's own
    report), making the guard vacuously true forever, dead weight rather than a real check. Fixed to
    assert against the real local only string ("atlas-registry:5000", environments/local/values.
    yaml's own repository value) instead. Proven non vacuous before this fix landed: temporarily
    editing environments/burst/values.yaml's own backend image repository to the local only string
    (simulating the exact leak this guard exists to catch) made this assertion fail as expected,
    then reverted; see the Task 2 report's "Fix round 1" section for the transcript."""
    result = _render("burst")
    assert result.returncode == 0, _failure_message(result, "burst")
    assert "kind: Cluster" in result.stdout
    assert "registry.invalid/atlas/atlas-postgres-pgvector@sha256:" in result.stdout
    assert "atlas-registry:5000" not in result.stdout


def test_missing_age_key_failure_self_diagnoses_with_a_pointer_to_the_readme():
    """Ride along from the Task 1 review: a missing/misconfigured age identity is the single most
    likely local failure mode of this test file, so its failure message should say so instead of
    leaving a bare sops/helmfile stack trace. Verified against sops's own real error text (not
    guessed): `SOPS_AGE_KEY_FILE=/dev/null` produces "Error getting data key: 0 successful groups
    required, got 0", which is why the special case matches on the substring "data key"."""
    env = os.environ.copy()
    env["SOPS_AGE_KEY_FILE"] = os.devnull
    result = subprocess.run(
        ["helmfile", "-e", "local", "template"],
        cwd=INFRA_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0, "expected this run to fail closed with no valid age identity"
    assert _MISSING_AGE_KEY_STDERR_SUBSTRING in result.stderr.lower()
    message = _failure_message(result, "local")
    assert "infra/README.md" in message
    assert "SOPS + age" in message


# --- Task 3: TEI embed/rerank (inCluster on burst, external on local), HF cache PVCs, indexes -------
#
# `tei.mode` (environments/base/values.yaml's own default "inCluster", overridden to "external" by
# environments/local/values.yaml) decides WHERE TEI runs, found live while standing the in cluster
# path up first on this arm64 dev machine: an OCI index platform refusal (fixable) and then a real
# ONNX Runtime warmup memory ceiling on tei-embed specifically (not fixable within a bounded k8s
# memory limit) -- see environments/local/values.yaml's own comment and the Task 3 report for the
# full diagnosis, including that tei-rerank DID stabilize under the same emulation once capped,
# proving execution itself was never the blocker. burst's real amd64 Hetzner nodes never hit either
# problem, so `test_burst_environment_renders_tei_in_cluster_with_pinned_digest_and_revisions` below
# is where the ORIGINAL Deployment shaped assertions (pinned digest, probes, resources, HF cache
# PVCs) now live; local's own tests assert the ABSENCE of those resources plus the external
# connectivity-check Job that replaces them.


def test_burst_environment_renders_tei_in_cluster_with_pinned_digest_and_revisions():
    """burst's real amd64 Hetzner nodes run `tei.mode: inCluster` (environments/base/values.yaml's
    own default, never overridden by environments/burst/values.yaml): the exact ghcr.io digest
    docker-compose.yml pins (D37 + D26, copied verbatim, never a floating alias), and
    --model-id/--revision matching models.lock exactly. `imagePullPolicy` is asserted explicit
    (IfNotPresent) rather than left to Kubernetes' own implicit default."""
    result = _render("burst")
    assert result.returncode == 0, _failure_message(result, "burst")
    documents = _documents(result)
    embed = _find(documents, kind="Deployment", name="tei-embed")
    rerank = _find(documents, kind="Deployment", name="tei-rerank")

    pinned_tei_image = (
        "ghcr.io/huggingface/text-embeddings-inference"
        "@sha256:ad950d30878eceb72aaf32024d26fa2b1d04a75304fa0b4776b49aa1941fea07"
    )
    for deployment in (embed, rerank):
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == pinned_tei_image
        assert container["imagePullPolicy"] == "IfNotPresent"

    embed_args = embed["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "BAAI/bge-m3" in embed_args
    assert "5617a9f61b028005a4858fdac845db406aefb181" in embed_args
    # docker-compose.yml's own choice: tei-embed is left uncapped (ONNX Runtime backend, no memory
    # ceiling to protect on real amd64 hardware).
    assert "--max-batch-tokens" not in embed_args

    rerank_args = rerank["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "BAAI/bge-reranker-v2-m3" in rerank_args
    assert "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e" in rerank_args
    # docker-compose.yml's own tei-rerank comment: the Candle CPU backend's warmup batch is sized
    # off --max-batch-tokens; uncapped, warmup was pathological (30+ min, refused connections).
    assert "--max-batch-tokens" in rerank_args
    assert "4096" in rerank_args


def test_burst_environment_tei_probes_carry_the_documented_compose_warmup_allowance():
    """docker-compose.yml's healthcheck start_period is 2700s (embed) / 900s (rerank). Kubernetes'
    direct equivalent of a docker healthcheck's start_period (a warmup grace window before failures
    count against the container) is a startupProbe, not a generous initialDelaySeconds bolted onto
    the steady state readiness probe, so that is what this asserts: periodSeconds * failureThreshold
    reproducing each exact window, not an eyeballed approximation. The readinessProbe underneath
    mirrors compose's own steady state interval/timeout/retries (10s/5s/6) once startup succeeds."""
    result = _render("burst")
    assert result.returncode == 0, _failure_message(result, "burst")
    documents = _documents(result)
    embed = _find(documents, kind="Deployment", name="tei-embed")
    rerank = _find(documents, kind="Deployment", name="tei-rerank")

    def container(deployment: dict) -> dict:
        return deployment["spec"]["template"]["spec"]["containers"][0]

    embed_startup = container(embed)["startupProbe"]
    assert embed_startup["httpGet"]["path"] == "/health"
    assert embed_startup["httpGet"]["port"] == 80
    assert embed_startup["periodSeconds"] * embed_startup["failureThreshold"] == 2700

    rerank_startup = container(rerank)["startupProbe"]
    assert rerank_startup["periodSeconds"] * rerank_startup["failureThreshold"] == 900

    for deployment in (embed, rerank):
        readiness = container(deployment)["readinessProbe"]
        assert readiness["httpGet"]["path"] == "/health"
        assert readiness["periodSeconds"] == 10
        assert readiness["timeoutSeconds"] == 5
        assert readiness["failureThreshold"] == 6


def test_burst_environment_tei_resources_are_modest_requests_without_warmup_choking_limits():
    """Resource awareness (this task's own controller instruction): modest requests so the scheduler
    never assumes more than a fair share, but no tight memory LIMIT that would OOM kill a warming up
    container. docker-compose.yml's own comment records observed peaks under Rosetta emulation
    (~10GB embed, ~4-10GB rerank); live testing on the local arm64 machine (see the Task 3 report's
    memory pressure section, the reason `tei.mode` exists at all) found even a 14Gi/10Gi margin
    insufficient there, so the chart's own defaults were raised to 18Gi/12Gi, a reasonable
    conservative floor kept here for burst's real hardware too. No CPU limit is set at all:
    constraining CPU would only slow the warmup down further."""
    result = _render("burst")
    assert result.returncode == 0, _failure_message(result, "burst")
    documents = _documents(result)
    embed = _find(documents, kind="Deployment", name="tei-embed")
    rerank = _find(documents, kind="Deployment", name="tei-rerank")

    def resources(deployment: dict) -> dict:
        return deployment["spec"]["template"]["spec"]["containers"][0]["resources"]

    for deployment in (embed, rerank):
        res = resources(deployment)
        assert res["requests"]["cpu"] == "250m"
        assert res["requests"]["memory"] == "512Mi"
        assert "cpu" not in res.get("limits", {})

    assert resources(embed)["limits"]["memory"] == "18Gi"
    assert resources(rerank)["limits"]["memory"] == "12Gi"


def test_burst_environment_renders_hf_cache_pvcs_for_both_tei_services():
    """The HF cache PVC (mounted at /data, TEI's own cache dir) is the k8s equivalent of compose's
    tei-embed-cache/tei-rerank-cache named volumes: it makes every restart AFTER the first one warm
    instead of a fresh multi minute download every time."""
    result = _render("burst")
    assert result.returncode == 0, _failure_message(result, "burst")
    documents = _documents(result)
    embed_pvc = _find(documents, kind="PersistentVolumeClaim", name="tei-embed-cache")
    rerank_pvc = _find(documents, kind="PersistentVolumeClaim", name="tei-rerank-cache")
    for pvc in (embed_pvc, rerank_pvc):
        assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"]
        assert pvc["spec"]["resources"]["requests"]["storage"]

    embed_deploy = _find(documents, kind="Deployment", name="tei-embed")
    volumes = embed_deploy["spec"]["template"]["spec"]["volumes"]
    hf_cache_volume = next(v for v in volumes if v["name"] == "hf-cache")
    assert hf_cache_volume["persistentVolumeClaim"]["claimName"] == "tei-embed-cache"


def test_local_environment_renders_no_tei_deployment_service_or_pvc():
    """The negative half of `tei.mode: external`: local must never schedule the Deployment/Service/
    HF cache PVC a real in cluster TEI service would need -- those resources are simply absent from
    the render, not present but broken."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    kinds_and_names = {(doc.get("kind"), doc.get("metadata", {}).get("name")) for doc in documents}
    assert ("Deployment", "tei-embed") not in kinds_and_names
    assert ("Deployment", "tei-rerank") not in kinds_and_names
    assert ("Service", "tei-embed") not in kinds_and_names
    assert ("Service", "tei-rerank") not in kinds_and_names
    assert ("PersistentVolumeClaim", "tei-embed-cache") not in kinds_and_names
    assert ("PersistentVolumeClaim", "tei-rerank-cache") not in kinds_and_names


def test_local_environment_renders_tei_external_connectivity_check_jobs_with_pinned_revisions():
    """`tei.mode: external` renders a small connectivity-check Job per role instead
    (infra/charts/tei/templates/connectivity-check-job.yaml): proof that a POD (not just this
    operator's own host shell) can reach the external endpoint, carrying the exact model_id/revision
    models.lock pins to compare the endpoint's own /info response against (D37 + D26: the pinned
    identity travels with the check even though TEI itself is not scheduled here)."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    embed_job = _find(documents, kind="Job", name="tei-embed-external-check")
    rerank_job = _find(documents, kind="Job", name="tei-rerank-external-check")

    def env_map(job: dict) -> dict:
        container = job["spec"]["template"]["spec"]["containers"][0]
        return {item["name"]: item["value"] for item in container["env"]}

    embed_env = env_map(embed_job)
    assert embed_env["TEI_URL"] == "http://tei-embed.render-test.invalid:8080"
    assert embed_env["EXPECTED_MODEL_ID"] == "BAAI/bge-m3"
    assert embed_env["EXPECTED_REVISION"] == "5617a9f61b028005a4858fdac845db406aefb181"

    rerank_env = env_map(rerank_job)
    assert rerank_env["TEI_URL"] == "http://tei-rerank.render-test.invalid:8081"
    assert rerank_env["EXPECTED_MODEL_ID"] == "BAAI/bge-reranker-v2-m3"
    assert rerank_env["EXPECTED_REVISION"] == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"

    # curlimages/curl (D37 digest pinned): a genuine manifest LIST for multiple architectures, unlike
    # the TEI image's single platform index, so it pulls natively on this arm64 node (see the chart's
    # own comment).
    for job in (embed_job, rerank_job):
        image = job["spec"]["template"]["spec"]["containers"][0]["image"]
        assert image.startswith("docker.io/curlimages/curl@sha256:")


def test_missing_tei_env_vars_render_empty_not_an_error_at_the_helmfile_layer():
    """Honest negative space, verified live rather than assumed: helmfile's own `ref+envsubst://`
    value resolution (environments/local/values.yaml) is deliberately PERMISSIVE, the same semantics
    as the shell's own `envsubst` -- an unset ATLAS_TEI_EMBED_URL/ATLAS_TEI_RERANK_URL resolves to an
    empty string, not a `helmfile template` failure (a `ref+env://` scheme that WOULD fail closed was
    tried first and does not exist in this vals build: "no provider registered for scheme env").

    The actual fail closed guarantee this task promises, a worded message naming .env.fastlane, lives
    in infra/scripts/k3d-up.sh, infra/scripts/k3d-verify.sh, and the `infra:render` Taskfile
    precondition, all of which check BEFORE ever invoking helmfile -- outside this render test's own
    scope (zero cluster, zero credentials, `helmfile template` only, per this file's own module
    docstring). This test exists so that scope boundary is asserted, not just claimed in a comment:
    a bare `helmfile template` with these variables unset must stay renderable, matching what the
    scripts above are themselves relying on when they choose to check first rather than let helmfile
    fail on their behalf."""
    env = os.environ.copy()
    env.setdefault("SOPS_AGE_KEY_FILE", str(DEFAULT_AGE_KEY_FILE))
    env.pop("ATLAS_TEI_EMBED_URL", None)
    env.pop("ATLAS_TEI_RERANK_URL", None)
    result = subprocess.run(
        ["helmfile", "-e", "local", "template"],
        cwd=INFRA_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    embed_job = _find(documents, kind="Job", name="tei-embed-external-check")
    container = embed_job["spec"]["template"]["spec"]["containers"][0]
    tei_url = next(item["value"] for item in container["env"] if item["name"] == "TEI_URL")
    assert tei_url == ""


def test_local_environment_renders_indexes_persistent_volume_and_claim():
    """Task 2 handoff (its own chart comments): replace atlas-jobs' raw hostPath mount with a proper
    PersistentVolume/PersistentVolumeClaim pair (SP5 digest section 2's own design), without changing
    the underlying k3d node path `k3d cluster create --volume $(pwd)/indexes:/indexes@all` already
    provides -- only the indirection through which rag-init reaches it changes."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    pv = _find(documents, kind="PersistentVolume", name="atlas-indexes-pv")
    pvc = _find(documents, kind="PersistentVolumeClaim", name="atlas-indexes-pvc")

    assert pv["spec"]["hostPath"]["path"] == "/indexes/corpus-0.1.1-bge-m3-03f983e0"
    assert pv["spec"]["hostPath"]["type"] == "Directory"
    assert pv["spec"]["storageClassName"] == "manual"
    assert pvc["spec"]["storageClassName"] == "manual"
    assert pvc["spec"]["volumeName"] == "atlas-indexes-pv"


def test_local_environment_rag_init_job_mounts_indexes_via_the_pvc_not_a_raw_hostpath():
    """The negative half of the PV/PVC migration: atlas-rag-init's OWN volume definition must no
    longer embed a hostPath directly (Task 2's shape) -- it must reference the PVC by name instead.
    Asserted against the Job document specifically, not a full stdout string search, because the
    PersistentVolume manifest asserted above legitimately contains a hostPath block of its own; a
    bare "hostPath absent" substring check against the whole render would be vacuously true for the
    wrong reason once that PV exists."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    job = _find(documents, kind="Job", name="atlas-rag-init")
    volumes = job["spec"]["template"]["spec"]["volumes"]
    indexes_volume = next(v for v in volumes if v["name"] == "indexes")
    assert "persistentVolumeClaim" in indexes_volume
    assert indexes_volume["persistentVolumeClaim"]["claimName"] == "atlas-indexes-pvc"
    assert "hostPath" not in indexes_volume


def test_burst_environment_renders_tei_deployments_alongside_the_indexes_pv():
    """D3: one set of charts, burst only ever diverges on values. burst's `tei.mode: inCluster`
    (environments/base/values.yaml's own default, unoverridden) and atlas-indexes' PV both render
    together without conflict, even though burst's own real indexes/ story is Task 5's job (this PV
    is a k3d node hostPath, meaningless off that specific node; environments/burst/values.yaml
    supplies a placeholder path today, proving the chart, not a real burst deploy, matching
    atlas-jobs' own precedent for burst)."""
    result = _render("burst")
    assert result.returncode == 0, _failure_message(result, "burst")
    documents = _documents(result)
    _find(documents, kind="Deployment", name="tei-embed")
    _find(documents, kind="Deployment", name="tei-rerank")
    _find(documents, kind="PersistentVolume", name="atlas-indexes-pv")


# --- Task 4: backend, web, ingress, D35 rate limit middleware, compose parity -------------------------
#
# docker-compose.yml's own "backend" and "web" services, translated to Deployments + Services (the
# same compose parity acceptance bar tasks 2-3 already held themselves to), plus a k3s Traefik
# IngressRoute and the D35 rate limit Middleware. Two translation choices carried over from what
# atlas-jobs and the tei chart already established: ATLAS_PG_DSN is assembled from the credentials
# Secret in a shell wrapper (Kubernetes has no native string interpolation across env vars, the same
# reason atlas-jobs' own Jobs do this), and ATLAS_TEI_EMBED_URL/ATLAS_TEI_RERANK_URL are gated by
# `.Values.tei.mode` (the Task 3 adjudication's own inherited requirement for this task): local's
# external passthrough URL, or burst's in cluster tei-embed/tei-rerank Service DNS names.


def test_local_environment_renders_the_backend_deployment_with_compose_parity_env():
    """docker-compose.yml's own "backend" service env, translated 1:1: ATLAS_MODE/MODEL_PROVIDER/
    MODEL_ID/OLLAMA_BASE_URL/ATLAS_RETRIEVER/ATLAS_CHECKPOINTER match compose's own defaults exactly,
    ATLAS_INDEX_DIR matches the same /app/indexes/<corpusDir> shape rag-init's own Job already mounts
    at, and ATLAS_PG_DSN is assembled from the credentials Secret at container start rather than a
    literal value."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    deployment = _find(documents, kind="Deployment", name="atlas-backend")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["imagePullPolicy"] == "IfNotPresent"
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}
    assert env["ATLAS_MODE"] == "replay"
    assert env["MODEL_PROVIDER"] == "ollama"
    assert env["MODEL_ID"] == "qwen2.5:7b"
    assert env["OLLAMA_BASE_URL"] == "http://host.docker.internal:11434"
    assert env["ATLAS_RETRIEVER"] == "pgvector"
    assert env["ATLAS_CHECKPOINTER"] == "postgres"
    assert env["ATLAS_INDEX_DIR"] == "/app/indexes/corpus-0.1.1-bge-m3-03f983e0"

    secret_refs = {
        item["name"]: item["valueFrom"]["secretKeyRef"]
        for item in container["env"] if "valueFrom" in item
    }
    assert secret_refs["PGUSER"] == {"name": "atlas-postgres-credentials", "key": "username"}
    assert secret_refs["PGPASSWORD"] == {"name": "atlas-postgres-credentials", "key": "password"}

    assembled = "\n".join(container["args"])
    assert "export ATLAS_PG_DSN=" in assembled
    assert "atlas-pg-rw" in assembled
    assert "uvicorn atlas.server:app" in assembled


def test_local_environment_backend_env_carries_the_tei_passthrough_urls():
    """Inherited requirement from the Task 3 adjudication: the local tier's backend Deployment must
    carry the SAME resolved external TEI URLs the connectivity-check Job already verifies
    (`.Values.tei.external.<role>Url`, `ref+envsubst://` resolved at render time), not the in cluster
    Service DNS names burst uses -- local has no in cluster TEI to point at (Task 3's own
    `tei.mode: external` finding)."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    deployment = _find(documents, kind="Deployment", name="atlas-backend")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}
    assert env["ATLAS_TEI_EMBED_URL"] == "http://tei-embed.render-test.invalid:8080"
    assert env["ATLAS_TEI_RERANK_URL"] == "http://tei-rerank.render-test.invalid:8081"


def test_burst_environment_backend_env_points_at_in_cluster_tei_services():
    """Inherited requirement: burst never overrides `tei.mode` (environments/base/values.yaml's own
    "inCluster" default), so its backend Deployment must point at the SAME in cluster Service DNS
    names tei-embed/tei-rerank's own service.yaml already names for exactly this purpose."""
    result = _render("burst")
    assert result.returncode == 0, _failure_message(result, "burst")
    documents = _documents(result)
    deployment = _find(documents, kind="Deployment", name="atlas-backend")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item["value"] for item in container["env"] if "value" in item}
    assert env["ATLAS_TEI_EMBED_URL"] == "http://tei-embed:80"
    assert env["ATLAS_TEI_RERANK_URL"] == "http://tei-rerank:80"


def test_both_environments_backend_env_carries_the_fallback_model_passthrough():
    """SP4 final fix wave (F2) parity follow up: compose gained ATLAS_FALLBACK_MODEL (empty means
    the provider_fallback rung stays absent, server.py _fallback_gateway); the chart must carry the
    same passthrough so k3d and burst can opt in via values instead of silently lacking the
    capability compose has."""
    for environment in ("local", "burst"):
        result = _render(environment)
        assert result.returncode == 0, _failure_message(result, environment)
        documents = _documents(result)
        deployment = _find(documents, kind="Deployment", name="atlas-backend")
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        env = {item["name"]: item["value"] for item in container["env"] if "value" in item}
        assert env["ATLAS_FALLBACK_MODEL"] == ""


def test_local_environment_backend_service_is_named_backend_for_nginx_proxy_parity():
    """frontend/nginx.conf (read only reference, never edited by this task) hardcodes
    `proxy_pass http://backend:8000/;`, the exact DNS shape docker-compose.yml's own network already
    gives the "backend" compose service. Matching that Service name here, not "atlas-backend", is
    what lets the SAME built web image proxy correctly with zero edits to frontend/."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    backend_svc = _find(documents, kind="Service", name="backend")
    assert backend_svc["spec"]["ports"][0]["port"] == 8000
    web_svc = _find(documents, kind="Service", name="atlas-web")
    assert web_svc["spec"]["ports"][0]["port"] == 80


def test_local_environment_backend_mounts_the_indexes_pvc_readonly():
    """The same `atlas-indexes-pvc` (Task 3) rag-init's Job already claims, now also mounted read
    only into the backend Deployment: `PgvectorRetriever` reads `fingerprint.json`/
    `build_manifest.json` off ATLAS_INDEX_DIR at construction (D9 fail closed discipline), so the
    served backend needs the SAME committed index build on disk, not baked into its own image
    (that would couple an image rebuild to every index bump)."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    deployment = _find(documents, kind="Deployment", name="atlas-backend")
    spec = deployment["spec"]["template"]["spec"]
    container = spec["containers"][0]
    mount = next(m for m in container["volumeMounts"] if m["name"] == "indexes")
    assert mount["mountPath"] == "/app/indexes/corpus-0.1.1-bge-m3-03f983e0"
    assert mount["readOnly"] is True
    volume = next(v for v in spec["volumes"] if v["name"] == "indexes")
    assert volume["persistentVolumeClaim"]["claimName"] == "atlas-indexes-pvc"


def test_local_environment_backend_and_web_images_are_digest_shaped():
    """D37: the backend and web images `task k3d:up` builds and pushes are digest shaped, matching
    the postgres-pgvector/backend precedent Task 2 already asserts (SHAPE, not a literal value that
    would go stale on the next local rebuild)."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    assert "atlas-registry:5000/atlas-backend@sha256:" in result.stdout
    assert "atlas-registry:5000/atlas-web@sha256:" in result.stdout


def test_local_environment_renders_the_ingressroute_and_rate_limit_middleware():
    """k3s ships Traefik as its default ingress controller (SP5 digest open decision 6): only the
    IngressRoute/Middleware CRDs against that existing instance, no separate ingress controller
    release. apiVersion confirmed live against the real cluster's own installed CRDs
    (`traefik.io/v1alpha1`, the Traefik v3 shape this k3s version ships; the older
    `traefik.containo.us` group is not installed)."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    middleware = _find(documents, kind="Middleware", name="atlas-rate-limit")
    assert middleware["apiVersion"] == "traefik.io/v1alpha1"
    assert middleware["spec"]["rateLimit"]["average"] > 0
    assert middleware["spec"]["rateLimit"]["burst"] > 0

    route = _find(documents, kind="IngressRoute", name="atlas-web")
    assert route["apiVersion"] == "traefik.io/v1alpha1"
    assert route["spec"]["entryPoints"] == ["web"]
    rule = route["spec"]["routes"][0]
    assert rule["services"][0]["name"] == "atlas-web"
    assert rule["services"][0]["port"] == 80
    assert rule["middlewares"][0]["name"] == "atlas-rate-limit"


def test_ingress_middleware_names_the_d35_backend_seam_in_a_comment():
    """Task 4's own instruction: "per session budget code is backend work, name it as the seam in a
    comment." Helm strips `{{- /* ... */ -}}` template comments from rendered output entirely, so
    this reads the chart's own source file directly rather than the render, the only way to actually
    assert the comment exists."""
    source = (INFRA_DIR / "charts" / "atlas-ingress" / "templates" / "middleware.yaml").read_text()
    assert "ATLAS_BURST_SPEND_CEILING_USD" in source
    assert "session" in source.lower()


def test_burst_environment_backend_env_carries_the_burst_spend_ceiling_seam():
    """The concrete half of the D35 seam (SP5 digest section 5's own recommendation): the VALUE lands
    in the Deployment's env, sourced from environments/burst/values.yaml -- reading it and tripping
    the degradation ladder to honest refusal is backend/SP4/SP6 work, not written here. Local never
    sets a spend ceiling (no real spend to cap in dev), so the env var must be absent there, not
    merely zero or empty."""
    burst_result = _render("burst")
    assert burst_result.returncode == 0, _failure_message(burst_result, "burst")
    burst_deployment = _find(_documents(burst_result), kind="Deployment", name="atlas-backend")
    burst_container = burst_deployment["spec"]["template"]["spec"]["containers"][0]
    burst_names = {item["name"] for item in burst_container["env"]}
    assert "ATLAS_BURST_SPEND_CEILING_USD" in burst_names

    local_result = _render("local")
    assert local_result.returncode == 0, _failure_message(local_result, "local")
    local_deployment = _find(_documents(local_result), kind="Deployment", name="atlas-backend")
    local_container = local_deployment["spec"]["template"]["spec"]["containers"][0]
    local_names = {item["name"] for item in local_container["env"]}
    assert "ATLAS_BURST_SPEND_CEILING_USD" not in local_names


def test_burst_environment_renders_backend_web_ingress_alongside_everything_else():
    """D3: one set of charts, burst only ever diverges on values. burst's own placeholder
    `registry.invalid` images render for backend/web too (the real burst registry is SP6's job,
    per infra/README.md's gap list), and this render must never leak local's real registry, the
    same guard the existing CNPG leak test already holds itself to."""
    result = _render("burst")
    assert result.returncode == 0, _failure_message(result, "burst")
    documents = _documents(result)
    _find(documents, kind="Deployment", name="atlas-backend")
    _find(documents, kind="Deployment", name="atlas-web")
    _find(documents, kind="IngressRoute", name="atlas-web")
    assert "registry.invalid/atlas/atlas-backend@sha256:" in result.stdout
    assert "registry.invalid/atlas/atlas-web@sha256:" in result.stdout
    assert "atlas-registry:5000" not in result.stdout


# --- Task 5: the wildcard cert (cert-manager DNS-01), burst only ------------------------------------
#
# infra/charts/atlas-cert: a ClusterIssuer (Cloudflare DNS-01 solver) and a wildcard Certificate,
# gated on .Values.certManager.enabled (environments/base/values.yaml's own shared default false,
# environments/burst/values.yaml the only place it flips true), the same conditional chart gating
# tei.mode already established for a resource that only makes sense on one tier. The cert-manager
# OPERATOR itself (CRDs, controller) is infra/tofu/cluster's own concern (kube-hetzner's
# enable_cert_manager toggle), never installed by k3d, which is exactly why this must render nothing
# at all for local rather than a ClusterIssuer/Certificate with no controller to reconcile it.


def test_local_environment_renders_no_cert_manager_resources():
    """The negative half of certManager.enabled: local must never render the ClusterIssuer, its
    Cloudflare token Secret, or the wildcard Certificate -- k3d has no cert-manager CRDs installed to
    even accept these kinds, so absence here is the correct behavior, not a broken one."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    kinds_and_names = {(doc.get("kind"), doc.get("metadata", {}).get("name")) for doc in documents}
    assert ("ClusterIssuer", "atlas-letsencrypt-dns01") not in kinds_and_names
    assert ("Certificate", "atlas-wildcard") not in kinds_and_names
    assert ("Secret", "atlas-cloudflare-api-token") not in kinds_and_names


def test_burst_environment_renders_the_dns01_clusterissuer_and_wildcard_certificate():
    """D3: "one wildcard cert via cert-manager DNS-01." The ClusterIssuer's Cloudflare token Secret
    lives in the "cert-manager" namespace (the controller's own cluster resource namespace default,
    NOT .Values.namespace) -- the single most common cert-manager DNS-01 misconfiguration, asserted
    directly here rather than left to be discovered live against a real burst cluster. The Certificate
    itself lives in .Values.namespace ("atlas"): Traefik's own IngressRoute.tls.secretName resolves
    against the SAME namespace as the IngressRoute, per that chart's own future wiring (named as a
    seam, not implemented, in this chart's own certificate.yaml comment)."""
    result = _render("burst")
    assert result.returncode == 0, _failure_message(result, "burst")
    documents = _documents(result)

    secret = _find(documents, kind="Secret", name="atlas-cloudflare-api-token")
    assert secret["metadata"]["namespace"] == "cert-manager"
    assert secret["stringData"]["api-token"]

    issuer = _find(documents, kind="ClusterIssuer", name="atlas-letsencrypt-dns01")
    assert "namespace" not in issuer["metadata"]  # cluster scoped, never namespaced
    solver = issuer["spec"]["acme"]["solvers"][0]["dns01"]["cloudflare"]
    assert solver["apiTokenSecretRef"] == {"name": "atlas-cloudflare-api-token", "key": "api-token"}
    assert issuer["spec"]["acme"]["email"] == "burst-ops@render-test.invalid"

    certificate = _find(documents, kind="Certificate", name="atlas-wildcard")
    assert certificate["metadata"]["namespace"] == "atlas"
    assert certificate["spec"]["secretName"] == "atlas-wildcard-tls"
    assert certificate["spec"]["issuerRef"] == {"name": "atlas-letsencrypt-dns01", "kind": "ClusterIssuer"}
    assert set(certificate["spec"]["dnsNames"]) == {
        "atlas.render-test.invalid",
        "*.atlas.render-test.invalid",
    }


def test_burst_environment_cloudflare_token_decrypts_through_the_same_sops_mechanism():
    """D41's two age recipient discipline extends to burst's own secret, not just local's fake
    password: environments/burst/secrets.enc.yaml decrypts through the SAME helmfile `vals`
    `ref+sops://` integration during rendering (infra/.sops.yaml's existing creation_rules already
    cover this path, path_regex `environments/.*\\.enc\\.yaml$`, no change needed there). The value is
    an obviously fake placeholder (never a real Cloudflare token), proving the mechanism, not a real
    credential."""
    result = _render("burst")
    assert result.returncode == 0, _failure_message(result, "burst")
    assert "REPLACE_WITH_A_REAL_CLOUDFLARE_DNS_EDIT_SCOPED_API_TOKEN_NEVER_COMMIT_A_REAL_ONE" in result.stdout


def test_missing_burst_env_vars_render_empty_not_an_error_at_the_helmfile_layer():
    """The same honest negative space test_missing_tei_env_vars_render_empty_not_an_error_at_the_
    helmfile_layer already proves for ATLAS_TEI_EMBED_URL/ATLAS_TEI_RERANK_URL, now for
    ATLAS_BURST_DOMAIN/ATLAS_BURST_ACME_EMAIL: helmfile's own `ref+envsubst://` resolution is
    permissive (an unset variable resolves to an empty string, not a `helmfile template` failure), so
    the actual fail closed guarantee lives in infra/scripts/burst-up.sh's own credential gate, not
    here -- this test asserts that scope boundary stays true rather than merely claiming it."""
    env = os.environ.copy()
    env.setdefault("SOPS_AGE_KEY_FILE", str(DEFAULT_AGE_KEY_FILE))
    env.setdefault("ATLAS_TEI_EMBED_URL", "http://tei-embed.render-test.invalid:8080")
    env.setdefault("ATLAS_TEI_RERANK_URL", "http://tei-rerank.render-test.invalid:8081")
    env.pop("ATLAS_BURST_DOMAIN", None)
    env.pop("ATLAS_BURST_ACME_EMAIL", None)
    result = subprocess.run(
        ["helmfile", "-e", "burst", "template"],
        cwd=INFRA_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, _failure_message(result, "burst")
    documents = _documents(result)
    certificate = _find(documents, kind="Certificate", name="atlas-wildcard")
    assert certificate["spec"]["dnsNames"] == ["", "*."]


# ---- Task 5: otel-collector + phoenix claim the SP6 task 3 parked digests --------------------------


def _source_block(stdout: str, source_comment: str) -> str:
    """The raw text of one `# Source: <chart>/templates/<file>` block, needed wherever a plain YAML
    `#` comment (stripped by `yaml.safe_load`, unlike Helm's own `{{- /* */ -}}` template comments)
    is itself the thing under test."""
    start = stdout.index(source_comment)
    next_marker = stdout.find("\n# Source: ", start + len(source_comment))
    return stdout[start : next_marker if next_marker != -1 else None]


def _base_observability_digests() -> dict:
    base_values = yaml.safe_load((ROOT / "infra" / "environments" / "base" / "values.yaml").read_text())
    return base_values["observability"]


def test_local_environment_renders_otel_collector_and_phoenix_with_the_parked_digests():
    """SP6 task 3 parked these exact digests for whichever task claimed this release (SP5 digest
    open decision 11); this task claims it, and must read the SAME field rather than declare a
    second, independently pinned digest for the identical image (test_observability_values.py's own
    parity check against docker-compose.yml already holds the parked field to account)."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    parked = _base_observability_digests()

    otel = _find(documents, kind="Deployment", name="otel-collector")
    otel_image = otel["spec"]["template"]["spec"]["containers"][0]["image"]
    expected_otel = f"{parked['otelCollector']['image']['repository']}@{parked['otelCollector']['image']['digest']}"
    assert otel_image == expected_otel

    phoenix = _find(documents, kind="Deployment", name="phoenix")
    phoenix_image = phoenix["spec"]["template"]["spec"]["containers"][0]["image"]
    expected_phoenix = f"{parked['phoenix']['image']['repository']}@{parked['phoenix']['image']['digest']}"
    assert phoenix_image == expected_phoenix


def test_otel_collector_configmap_carries_the_committed_redaction_config():
    """The ConfigMap is joined in from the real, committed, drift gated
    infra/observability/otel-collector.yaml via helmfile's own `readFile` templating
    (infra/environments/base/otel-collector-values.yaml.gotmpl), never a hand copied duplicate --
    asserted here against content that only the REAL file carries (an allowlisted attribute name,
    Phoenix's own fan out target), so a regression back to an empty or stale copy would fail this,
    not merely "a ConfigMap named otel-collector-config exists"."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    cm = _find(documents, kind="ConfigMap", name="otel-collector-config")
    config_text = cm["data"]["otel-collector.yaml"]
    assert "atlas.guard.decision" in config_text  # RESERVED_TRACE_ATTRIBUTES, via the allowlist
    assert "endpoint: phoenix:4317" in config_text
    parsed = yaml.safe_load(config_text)
    assert parsed["exporters"]["otlp/phoenix"]["endpoint"] == "phoenix:4317"


def test_phoenix_db_init_job_creates_a_new_database_never_a_second_postgres():
    """SP6 digest design question 5: Phoenix's own storage is a NEW database inside the SAME CNPG
    cluster every other component uses, never SQLite, never a second Postgres instance -- the Job
    connects to the EXISTING "atlas" database first (CREATE DATABASE cannot run against the database
    being created) and only creates "phoenix" if it is not already there."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    job = _find(documents, kind="Job", name="atlas-phoenix-db-init")
    args = "\n".join(job["spec"]["template"]["spec"]["containers"][0]["args"])
    assert "atlas-pg-rw" in args  # the SAME CNPG read/write Service atlas-jobs/atlas-backend use
    assert "-d atlas" in args    # connects through the EXISTING database first
    assert "CREATE DATABASE phoenix" in args
    assert job["metadata"]["annotations"]["helm.sh/hook"] == "pre-install,pre-upgrade"


def test_phoenix_deployment_assembles_its_own_database_url_from_the_shared_credentials_secret():
    """C2 fix (SP6 final review): the arizephoenix image has no shell, so the DSN is assembled via
    Kubernetes' own dependent env var expansion (`$(VAR)`), never a `command: ["sh", "-c"]` wrapper
    -- this chart carries no `command`/`args` override at all, letting the image's own entrypoint
    run unmodified, the same shape compose's own service already proves works."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    deployment = _find(documents, kind="Deployment", name="phoenix")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert "command" not in container  # the image's own baked in entrypoint, unmodified
    assert "args" not in container
    env_by_name = {item["name"]: item for item in container["env"]}
    secret_refs = {
        name: item["valueFrom"]["secretKeyRef"]
        for name, item in env_by_name.items() if "valueFrom" in item
    }
    assert secret_refs["PGUSER"] == {"name": "atlas-postgres-credentials", "key": "username"}
    assert secret_refs["PGPASSWORD"] == {"name": "atlas-postgres-credentials", "key": "password"}
    env_names = [item["name"] for item in container["env"]]
    # PGUSER/PGPASSWORD must be declared BEFORE the entry that references them via $(VAR): dependent
    # env var expansion only resolves an EARLIER entry in the same list, by name, in list order.
    assert env_names.index("PGUSER") < env_names.index("PHOENIX_SQL_DATABASE_URL")
    assert env_names.index("PGPASSWORD") < env_names.index("PHOENIX_SQL_DATABASE_URL")
    dsn = env_by_name["PHOENIX_SQL_DATABASE_URL"]["value"]
    assert dsn == "postgresql://$(PGUSER):$(PGPASSWORD)@atlas-pg-rw:5432/phoenix"  # the NEW database


def test_phoenix_deployment_disables_service_link_env_injection():
    """C2 fix, second half (SP6 final review): found live on a RESTART of this Deployment, after its
    own Service ("phoenix", `service.yaml`) already existed -- Kubernetes' legacy Docker links
    compatibility feature then auto injects `PHOENIX_PORT=tcp://<ClusterIP>:6006`, a URL shaped
    string that collides BY NAME with Phoenix's own `PHOENIX_PORT` config var (its own `config.py`
    requires a bare integer), crash looping the process at import time. `enableServiceLinks: false`
    is the standard Kubernetes escape hatch; a purely live, restart order dependent failure this
    hermetic render test cannot reproduce on its own, so it can only pin the fix's own shape."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    deployment = _find(documents, kind="Deployment", name="phoenix")
    assert deployment["spec"]["template"]["spec"]["enableServiceLinks"] is False


# ---- Task 5: the prometheus-operator/kube-prometheus stack slot (SP5 digest open decision 11) -----


def test_local_and_burst_render_the_monitoring_stack_with_digest_shaped_images():
    """D37: every image reference is digest shaped, the same discipline (a regex over the shape,
    never a literal value that would go stale) test_local_environment_pgvector_and_backend_images_are_
    digest_shaped already holds itself to."""
    for environment in ("local", "burst"):
        result = _render(environment)
        assert result.returncode == 0, _failure_message(result, environment)
        documents = _documents(result)
        for name in ("prometheus", "alertmanager", "pushgateway"):
            deployment = _find(documents, kind="Deployment", name=name)
            image = deployment["spec"]["template"]["spec"]["containers"][0]["image"]
            assert _DIGEST_SHAPED_IMAGE_RE.search(image), f"{name} image not digest shaped: {image!r}"


def test_alertmanager_has_exactly_one_webhook_receiver_and_every_route_matches_it():
    """D29: 'Prometheus + Alertmanager only, ONE webhook receiver.' Deterministic paging (D29's own
    phrase): every alert this release can raise pages the SAME one place, no severity based routing
    tree, no second channel."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    secret = _find(documents, kind="Secret", name="alertmanager-config")
    config = yaml.safe_load(secret["stringData"]["alertmanager.yml"])
    assert len(config["receivers"]) == 1
    receiver = config["receivers"][0]
    assert len(receiver["webhook_configs"]) == 1
    assert receiver["webhook_configs"][0]["url"]
    assert config["route"]["receiver"] == receiver["name"]


def test_prometheus_scrapes_the_backend_metrics_endpoint_and_the_pushgateway():
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    cm = _find(documents, kind="ConfigMap", name="prometheus-config")
    config = yaml.safe_load(cm["data"]["prometheus.yml"])
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert jobs["atlas-backend"]["metrics_path"] == "/metrics"
    assert jobs["atlas-backend"]["static_configs"][0]["targets"] == ["backend:8000"]
    assert jobs["pushgateway"]["static_configs"][0]["targets"] == ["pushgateway:9091"]
    assert jobs["pushgateway"]["honor_labels"] is True
    assert config["alerting"]["alertmanagers"][0]["static_configs"][0]["targets"] == ["alertmanager:9093"]
    assert config["rule_files"] == ["/etc/prometheus/rules/atlas-rules.yml"]


def test_prometheusrule_declares_exactly_the_d29_deterministic_paging_set():
    """This task's own contract text, verbatim: 'probe failure, staleness gauge threshold, breaker
    open, error rate.' SP8 Task 4 remainder adds a fifth alert (`AtlasJudgeFailRateHigh`) beyond
    D29's original four, once the judge counter pair actually has a writer -- see
    `test_judge_fail_rate_rule_is_wired_and_the_prometheusrule_comment_names_sp8` below for that
    alert's own dedicated assertions."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    rule = _find(documents, kind="PrometheusRule", name="atlas-sentinel-rules")
    alert_names = {
        alert["alert"] for group in rule["spec"]["groups"] for alert in group["rules"]
    }
    assert alert_names == {
        "AtlasProbeFailure", "AtlasCorpusStaleness", "AtlasCircuitBreakerOpen", "AtlasErrorRateHigh",
        "AtlasJudgeFailRateHigh",
    }


def test_prometheusrule_crd_and_the_native_rule_configmap_never_drift():
    """Both the PrometheusRule CRD (forward compatible with a future real operator swap in) and the
    ConfigMap the ACTUALLY RUNNING Prometheus in this release reads are rendered off the exact same
    `.Values.rules.groups` (this chart's own values.yaml comment); asserted equal here so a future
    edit to one template that forgets the other fails this test instead of silently drifting."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    rule = _find(documents, kind="PrometheusRule", name="atlas-sentinel-rules")
    cm = _find(documents, kind="ConfigMap", name="prometheus-config")
    native_rules = yaml.safe_load(cm["data"]["atlas-rules.yml"])
    assert rule["spec"]["groups"] == native_rules["groups"]


def test_judge_fail_rate_rule_is_wired_and_the_prometheusrule_comment_names_sp8():
    """SP8 Task 4 remainder's own contract: the judge counter pair (`atlas_judge_pass_total`/
    `atlas_judge_fail_total`) now backs a real alert (judge fail rate over 20 percent of the
    sampling window, SP8's own documented choice), and the chart template's trailing comment is
    corrected from its former stale 'SP7 adds its own rule group here once it wires the calibrated
    judge, not before' to name SP8, the sub project that actually wired it (ADR-029's own rule:
    owner changes travel with the emitter). A plain YAML `#` comment in the CHART TEMPLATE (unlike a
    values.yaml comment, which never reaches rendered output at all) survives into `helm template`'s
    own output, so the comment half is asserted against the raw rendered text, not the parsed
    document (`yaml.safe_load` discards comments)."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    rule = _find(documents, kind="PrometheusRule", name="atlas-sentinel-rules")
    judge_group = next(g for g in rule["spec"]["groups"] if g["name"] == "atlas.judge")
    alert = next(a for a in judge_group["rules"] if a["alert"] == "AtlasJudgeFailRateHigh")
    assert "atlas_judge_fail_total" in alert["expr"]
    assert "atlas_judge_pass_total" in alert["expr"]
    assert "0.2" in alert["expr"]
    assert alert["labels"]["severity"] == "page"

    block = _source_block(result.stdout, "# Source: atlas-monitoring/templates/prometheusrule.yaml")
    assert "atlas_judge_pass_total" in block
    assert "atlas_judge_fail_total" in block
    assert "SP8" in block
    assert "SP7" not in block


def test_prometheusrule_every_alert_has_the_d29_deterministic_paging_shape():
    """Schema level structural assertions (this task's own contract's fallback for when `promtool`
    is unavailable, see test_prometheus_rule_file_passes_promtool_check_rules below): every alert
    declares expr/for/severity/summary, so a future edit that drops one of those fields fails here
    even without promtool installed."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    rule = _find(documents, kind="PrometheusRule", name="atlas-sentinel-rules")
    for group in rule["spec"]["groups"]:
        assert group["name"]
        for alert in group["rules"]:
            assert alert["alert"]
            assert alert["expr"]
            assert alert["for"]
            assert alert["labels"]["severity"] == "page"
            assert alert["annotations"]["summary"]


def _promtool_missing_reason() -> str | None:
    if shutil.which("promtool") is not None:
        return None
    return (
        "promtool is not on PATH: install prometheus (`brew install prometheus` or "
        "https://prometheus.io/download/) to get promtool's own `check rules` validation. "
        "test_prometheusrule_every_alert_has_the_d29_deterministic_paging_shape above still runs "
        "the schema level YAML assertions regardless of this skip."
    )


def test_prometheus_rule_file_passes_promtool_check_rules(tmp_path):
    reason = _promtool_missing_reason()
    if reason is not None:
        pytest.skip(reason)
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    cm = _find(documents, kind="ConfigMap", name="prometheus-config")
    rule_file = tmp_path / "atlas-rules.yml"
    rule_file.write_text(cm["data"]["atlas-rules.yml"])
    check = subprocess.run(
        ["promtool", "check", "rules", str(rule_file)], capture_output=True, text=True, timeout=30,
    )
    assert check.returncode == 0, f"promtool check rules failed:\n{check.stdout}\n{check.stderr}"


def test_sentinel_probe_cronjob_reuses_the_backend_image_and_targets_the_backend_service():
    """D29: 'sentinel probe CronJob... every 5 minutes.' Reuses the SAME digest pinned atlas-backend
    image unmodified (no new image, testing/harness/sentinel/probe.py is already baked in via
    backend/Dockerfile's own COPY testing/harness), targets the SAME Service name atlas-backend's own
    chart names for nginx parity ("backend", not "atlas-backend")."""
    result = _render("local")
    assert result.returncode == 0, _failure_message(result, "local")
    documents = _documents(result)
    cronjob = _find(documents, kind="CronJob", name="atlas-sentinel-probe")
    assert cronjob["spec"]["schedule"] == "*/5 * * * *"
    assert cronjob["spec"]["concurrencyPolicy"] == "Forbid"
    pod_spec = cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    assert container["command"] == ["uv", "run", "--no-sync", "python", "-m", "sentinel.probe"]

    backend = _find(documents, kind="Deployment", name="atlas-backend")
    backend_image = backend["spec"]["template"]["spec"]["containers"][0]["image"]
    assert container["image"] == backend_image  # the exact SAME digest, no separate probe image

    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["ATLAS_PROBE_BASE_URL"] == "http://backend:8000"
    assert env["ATLAS_PROBE_PUSHGATEWAY_URL"] == "http://pushgateway:9091"
    assert env["ATLAS_PROBE_CUSTOMER_ID"]


def test_burst_environment_renders_the_full_observability_release_set():
    """D3: one set of charts, burst only ever diverges on values -- the same guard every other Task
    5 release already holds itself to above."""
    result = _render("burst")
    assert result.returncode == 0, _failure_message(result, "burst")
    documents = _documents(result)
    _find(documents, kind="Deployment", name="otel-collector")
    _find(documents, kind="Deployment", name="phoenix")
    _find(documents, kind="Deployment", name="prometheus")
    _find(documents, kind="Deployment", name="alertmanager")
    _find(documents, kind="Deployment", name="pushgateway")
    _find(documents, kind="CronJob", name="atlas-sentinel-probe")
    _find(documents, kind="PrometheusRule", name="atlas-sentinel-rules")
