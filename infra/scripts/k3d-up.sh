#!/usr/bin/env bash
# task k3d:up (SP5 task 2 phase 1, extended by tasks 3 and 4, and by SP6 task 5's observability
# releases): a k3d cluster running CNPG with pgvector, the migrate and rag-init one shot Jobs, the
# indexes/ PersistentVolume/PersistentVolumeClaim pair, TEI embed/rerank verification, the
# backend/web Deployments, the Traefik IngressRoute (D35 rate limit Middleware included), and
# (C1 fix, SP6 final review) phoenix/otel-collector/atlas-monitoring -- the D29 alerting surface and
# the Phoenix trace backend, previously deployed on this tier only by an operator hand syncing
# releases this script never named. Idempotent to rerun: skips cluster creation if "atlas" already
# exists, always rebuilds and repushes every locally built image (a local dev loop is supposed to
# pick up new code), and every one shot Job is a Helm pre-install/pre-upgrade hook that deletes and
# recreates on every `helmfile sync` rather than hitting Kubernetes' "Job spec is immutable" error on
# a plain `helm upgrade`.
#
# Ingress port mapping (Task 4): the cluster is created with `-p "${INGRESS_HTTP_PORT}:80@loadbalancer"`
# so k3s' own Traefik "web" entryPoint reaches the host. This only takes effect at CREATE time: a
# cluster already running from before this task (Tasks 2/3) needs one `task k3d:down` then
# `task k3d:up` cycle to pick it up. Host port 80/443 were already bound by an unrelated k3d cluster
# on the machine this task was developed on, hence a default port here that is not the usual 8080 (also
# already taken, by docker-compose.yml's own "web" service) or 80/443.
#
# TEI mode (Task 3, found live standing the in cluster path up first): this arm64 k3d dev machine
# cannot run the amd64 only TEI image reliably (an OCI index platform refusal, fixable, and then a
# real ONNX Runtime warmup memory ceiling on tei-embed specifically that was not, even after raising
# limits well above docker-compose.yml's own documented peaks -- full diagnosis in the Task 3 report
# and environments/local/values.yaml's own comment). `tei.mode: external` there points both services
# at a real external amd64 endpoint instead; this script sources `.env.fastlane` (repo root,
# gitignored) for its URL and fails with a named, worded message if that file or its variables are
# missing (helmfile's own `ref+envsubst://` resolution is deliberately permissive, an unset
# variable silently becomes an empty string, so this script's own check is the ONLY fail closed
# gate, not a backstop for one).
#
# `helmfile sync` (not `apply`): `apply` computes a helm-diff preview first, which needs the
# helm-diff plugin (not installed on this machine, and not worth adding for a one way local up
# script); `sync` runs `helm upgrade --install` unconditionally, exactly what an idempotent,
# non interactive script wants.
#
# Ordering matters and Kubernetes has no native `service_completed_successfully` primitive (the SP5
# digest's own flagged gap, section 2): this script is that missing orchestration, not a stand in
# for something Kubernetes should be doing itself. Each `helmfile sync -l name=<release>` call is
# scoped to exactly one release so an explicit `kubectl wait` can run between them.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INFRA_DIR="${ROOT_DIR}/infra"
CLUSTER_NAME="atlas"
REGISTRY_NAME="atlas-registry"
REGISTRY_INTERNAL="atlas-registry:5000"
GIT_SHA="$(git -C "${ROOT_DIR}" rev-parse --short HEAD)"
# Task 4: the host port mapped to k3s Traefik's "web" entryPoint (infra/scripts/k3d-smoke.sh reads
# the same default). Overridable if 8090 is already taken on some other machine.
INGRESS_HTTP_PORT="${INGRESS_HTTP_PORT:-8090}"

export SOPS_AGE_KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/atlas-age/keys.txt}"

log() { printf '\n>>> %s\n' "$1"; }

# --- 0. the local tier's external TEI endpoint ---------------------------------------------------
# environments/local/values.yaml's `tei.mode: external` needs ATLAS_TEI_EMBED_URL/
# ATLAS_TEI_RERANK_URL in the environment before `helmfile` ever runs. This check IS the fail closed
# gate (see this file's own header comment on why helmfile's own value resolution is not one).
ENV_FASTLANE="${ROOT_DIR}/.env.fastlane"
if [[ -f "${ENV_FASTLANE}" ]]; then
  log "sourcing ${ENV_FASTLANE} for the external TEI endpoint"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FASTLANE}"
  set +a
fi
if [[ -z "${ATLAS_TEI_EMBED_URL:-}" || -z "${ATLAS_TEI_RERANK_URL:-}" ]]; then
  echo "ATLAS_TEI_EMBED_URL and ATLAS_TEI_RERANK_URL must both be set (environments/local/values.yaml's tei.mode: external needs them). Create ${ENV_FASTLANE} with both variables set to a reachable amd64 TEI endpoint's base URL (see infra/README.md's k3d tier section), then rerun. This file is gitignored on purpose: never commit a real endpoint address." >&2
  exit 1
fi

# --- 1. cluster + registry ---------------------------------------------------------------------

if k3d cluster list "${CLUSTER_NAME}" >/dev/null 2>&1; then
  log "k3d cluster '${CLUSTER_NAME}' already exists, skipping create"
else
  log "creating k3d cluster '${CLUSTER_NAME}' with registry '${REGISTRY_NAME}'"
  # The registry-create + volume + port flags are all needed by this task (D37's local registry for
  # digest shaped refs, the indexes/ hostPath rag-init reads from, and Task 4's ingress port mapping);
  # combined into the one cluster create invocation rather than a second command against an already
  # running cluster.
  k3d cluster create "${CLUSTER_NAME}" \
    --registry-create "${REGISTRY_NAME}" \
    --volume "${ROOT_DIR}/indexes:/indexes@all" \
    --port "${INGRESS_HTTP_PORT}:80@loadbalancer" \
    --wait --timeout 300s
fi

kubectl config use-context "k3d-${CLUSTER_NAME}" >/dev/null

REGISTRY_HOST_PORT="$(docker port "${REGISTRY_NAME}" 5000/tcp | head -1 | sed 's/.*://')"
if [[ -z "${REGISTRY_HOST_PORT}" ]]; then
  echo "could not resolve the host published port for the '${REGISTRY_NAME}' registry container (docker port ${REGISTRY_NAME} 5000/tcp returned nothing)" >&2
  exit 1
fi
REGISTRY_HOST_ADDR="localhost:${REGISTRY_HOST_PORT}"
log "registry: host visible push address ${REGISTRY_HOST_ADDR}, cluster internal pull address ${REGISTRY_INTERNAL}"

# --- 2. build + push images ----------------------------------------------------------------------
# --provenance=false --sbom=false: without these, BuildKit attaches an attestation manifest and
# `docker push` reports a manifest LIST digest instead of the single image manifest digest, an
# unnecessary complication for a local, single platform pipeline. Verified directly against this
# Docker Desktop install before writing this script.

push_and_capture_digest() {
  local tag="$1"
  local out
  out="$(docker push "${tag}" 2>&1)"
  echo "${out}" >&2
  local digest
  digest="$(echo "${out}" | grep -oE 'digest: sha256:[0-9a-f]{64}' | tail -1 | awk '{print $2}')"
  if [[ -z "${digest}" ]]; then
    echo "could not parse a digest out of 'docker push ${tag}' output" >&2
    exit 1
  fi
  echo "${digest}"
}

log "building infra/images/postgres-pgvector (CNPG's own pg17 base + pgvector, D37 digest pinned FROM)"
PGVECTOR_TAG="${REGISTRY_HOST_ADDR}/atlas-postgres-pgvector:${GIT_SHA}"
docker build --provenance=false --sbom=false \
  -f "${INFRA_DIR}/images/postgres-pgvector/Dockerfile" \
  -t "${PGVECTOR_TAG}" \
  "${INFRA_DIR}/images/postgres-pgvector"
PGVECTOR_DIGEST="$(push_and_capture_digest "${PGVECTOR_TAG}")"
log "pushed ${PGVECTOR_TAG} -> ${PGVECTOR_DIGEST}"

log "building backend/Dockerfile (read only reference, unmodified; context is the repo root per its own header comment)"
BACKEND_TAG="${REGISTRY_HOST_ADDR}/atlas-backend:${GIT_SHA}"
docker build --provenance=false --sbom=false \
  -f "${ROOT_DIR}/backend/Dockerfile" \
  --build-arg "GIT_SHA=${GIT_SHA}" \
  -t "${BACKEND_TAG}" \
  "${ROOT_DIR}"
BACKEND_DIGEST="$(push_and_capture_digest "${BACKEND_TAG}")"
log "pushed ${BACKEND_TAG} -> ${BACKEND_DIGEST}"

log "building frontend/Dockerfile (read only reference, unmodified; context is the repo root per its own header comment, Task 4)"
WEB_TAG="${REGISTRY_HOST_ADDR}/atlas-web:${GIT_SHA}"
docker build --provenance=false --sbom=false \
  -f "${ROOT_DIR}/frontend/Dockerfile" \
  -t "${WEB_TAG}" \
  "${ROOT_DIR}"
WEB_DIGEST="$(push_and_capture_digest "${WEB_TAG}")"
log "pushed ${WEB_TAG} -> ${WEB_DIGEST}"

log "recording all three digests into infra/environments/local/values.yaml (D37: the up target records the digest into the local values file)"
python3 "${INFRA_DIR}/scripts/record_image_digests.py" \
  --values-file "${INFRA_DIR}/environments/local/values.yaml" \
  --registry-host "${REGISTRY_INTERNAL}" \
  --pgvector-digest "${PGVECTOR_DIGEST}" \
  --backend-digest "${BACKEND_DIGEST}" \
  --web-digest "${WEB_DIGEST}"

# --- 3. atlas namespace + postgres credentials secret --------------------------------------------

log "applying atlas-scaffold (namespace + postgres credentials secret)"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=atlas-scaffold)

# --- 3.5. TEI external endpoint verification (Task 3) ---------------------------------------------
# `tei.mode: external` on this tier (environments/local/values.yaml's own comment carries the full
# diagnosis of why): these two releases apply a small in cluster connectivity-check Job each instead
# of a Deployment (infra/charts/tei/templates/connectivity-check-job.yaml), proving a POD (not just
# this operator's own host shell) can reach ATLAS_TEI_EMBED_URL/ATLAS_TEI_RERANK_URL and that each
# endpoint actually serves the model_id/revision models.lock pins. A pre-install hook, same pattern
# as atlas-jobs below: `helmfile sync` blocks until it succeeds.

log "applying tei-embed, tei-rerank (external mode: in cluster connectivity + /info revision check against ${ATLAS_TEI_EMBED_URL} / ${ATLAS_TEI_RERANK_URL})"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=tei-embed)
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=tei-rerank)

# --- 4. CNPG operator -----------------------------------------------------------------------------

log "installing the CNPG operator (vendored chart, first pull downloads ghcr.io/cloudnative-pg/cloudnative-pg, can take a minute)"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=cnpg-operator)

log "waiting for the CNPG operator Deployment to be ready"
kubectl -n cnpg-system rollout status deployment/cnpg-operator-cloudnative-pg --timeout=180s

log "waiting for the CNPG CRDs to be established"
kubectl wait --for=condition=established --timeout=60s \
  crd/clusters.postgresql.cnpg.io crd/imagecatalogs.postgresql.cnpg.io

# --- 5. the pgvector Cluster ------------------------------------------------------------------------

log "applying the ImageCatalog + Cluster CR (D1: instances=1, no pooler)"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=cnpg-cluster)

log "waiting for the CNPG Cluster to report Ready (image pull + initdb; can take a few minutes on a cold registry)"
kubectl -n atlas wait --for=condition=Ready cluster/atlas-pg --timeout=600s

# --- 5.5. indexes/ PersistentVolume + PersistentVolumeClaim (Task 3) -----------------------------
# Replaces Task 2's raw hostPath mount (SP5 digest section 2's own design, per that task's handoff):
# a static PV/PVC pair over the SAME k3d node path (`--volume $(pwd)/indexes:/indexes@all` above,
# unchanged) that atlas-jobs' rag-init Job below now claims by name instead of embedding directly.
# Must run before atlas-jobs: rag-init's Job spec references this release's PVC.

log "applying atlas-indexes (PersistentVolume/PersistentVolumeClaim for indexes/, replacing Task 2's raw hostPath)"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=atlas-indexes)

# --- 6. the one shot Jobs (migrate, rag-init) ------------------------------------------------------

log "applying atlas-jobs (checkpointer-migrate, rag-init) -- helmfile sync blocks until both hook Jobs succeed"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=atlas-jobs)

# --- 7. backend, web, ingress (Task 4) ---------------------------------------------------------

log "applying atlas-backend (docker-compose.yml's own backend service, compose parity env)"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=atlas-backend)
kubectl -n atlas rollout status deployment/atlas-backend --timeout=180s

# --- 7.5. observability: phoenix, otel-collector, atlas-monitoring (SP6 task 5) ------------------
# C1 fix (SP6 final review): `burst-up.sh` runs a full, unselected `helmfile -e burst sync` (every
# release in infra/helmfile.yaml, no `-l` filter at all), so it picked up phoenix/otel-collector/
# atlas-monitoring automatically the moment SP6 task 5 added them to the releases list. This script
# instead syncs one release at a time by design (this file's own header comment: so an explicit
# `kubectl wait` can run between them, ordering what helmfile's own `needs:` graph only orders the
# SYNC of, never a live readiness wait) -- which means it has to be told about a NEW release by name,
# and never was for these three. A `task k3d:down && task k3d:up` therefore reported success while
# silently deploying none of them, reproduced live. The three lines below are that missing update,
# placed after atlas-backend (matching helmfile's own `needs:` order: phoenix only needs
# atlas-scaffold + cnpg-cluster, already synced and Ready long before this point; otel-collector
# needs phoenix; atlas-monitoring needs atlas-backend, just synced above), never before it.
log "applying phoenix (Postgres backed trace storage, SP6 digest design question 5)"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=phoenix)
kubectl -n atlas rollout status deployment/phoenix --timeout=120s

log "applying otel-collector (the redacting OTel collector, fanning out to phoenix + the raw archive)"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=otel-collector)
kubectl -n atlas rollout status deployment/otel-collector --timeout=120s

log "applying atlas-monitoring (Prometheus + Alertmanager + Pushgateway + the D29 sentinel probe CronJob)"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=atlas-monitoring)
kubectl -n atlas rollout status deployment/prometheus --timeout=120s
kubectl -n atlas rollout status deployment/alertmanager --timeout=120s
kubectl -n atlas rollout status deployment/pushgateway --timeout=120s

log "applying atlas-web (docker-compose.yml's own web service, nginx proxying to the backend Service)"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=atlas-web)
kubectl -n atlas rollout status deployment/atlas-web --timeout=120s

log "applying atlas-ingress (Traefik IngressRoute + D35 rate limit Middleware) -- reachable at http://localhost:${INGRESS_HTTP_PORT}"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=atlas-ingress)

log "phase 1+3+4+5 complete: cluster + registry + CNPG operator + pgvector cluster + jobs + indexes PV/PVC + TEI external endpoint + backend + phoenix + otel-collector + atlas-monitoring + web + ingress"

"${INFRA_DIR}/scripts/k3d-verify.sh"
