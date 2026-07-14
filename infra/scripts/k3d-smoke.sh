#!/usr/bin/env bash
# task k3d:smoke (SP5 task 4): the rag smoke, ported to the k3d tier and aimed at the real Traefik
# ingress rather than a directly published backend port (unlike compose's own `task rag:smoke`,
# which dials the compose backend's own published :8000 directly). Retrieval half always runs, no
# key needed; generation half only if a provider key is present in the env this target sources.
#
# CNPG's Service is ClusterIP only (never published to the host the way compose's own postgres:5433
# is), so this script opens a `kubectl port-forward` to atlas-pg-rw first (torn down on exit via a
# trap, whatever happens) and hands the retrieval half a DSN through it; the local tier's TEI
# endpoints are already host reachable (`tei.mode: external`, Task 3's own .env.fastlane), so no
# port-forward is needed for those. The actual HTTP work (retrieval half, chat endpoint/stream
# halves, generation half) is infra/scripts/k3d_smoke.py, kept as plain Python rather than bash + jq
# plumbing (this directory's own precedent, see record_image_digests.py/verify_tei_revisions.py).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INFRA_DIR="${ROOT_DIR}/infra"
NAMESPACE="atlas"
# Defensive: task k3d:smoke already sets this repo wide (Taskfile.yml's own top level `env:` block),
# but a direct invocation of this script (bypassing Task) still needs it to import atlas.*/replay.*.
export PYTHONPATH="${ROOT_DIR}/backend:${ROOT_DIR}/testing/harness:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# Matches infra/scripts/k3d-up.sh's own default, the host port mapped to k3s Traefik's "web"
# entryPoint at cluster create time.
INGRESS_HTTP_PORT="${INGRESS_HTTP_PORT:-8090}"
PG_LOCAL_PORT="${PG_LOCAL_PORT:-15432}"

log() { printf '\n>>> %s\n' "$1"; }

# --- 0. the local tier's external TEI endpoint (same fail closed gate as k3d-up.sh/k3d-verify.sh) --
ENV_FASTLANE="${ROOT_DIR}/.env.fastlane"
if [[ -f "${ENV_FASTLANE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FASTLANE}"
  set +a
fi
if [[ -z "${ATLAS_TEI_EMBED_URL:-}" || -z "${ATLAS_TEI_RERANK_URL:-}" ]]; then
  echo "ATLAS_TEI_EMBED_URL and ATLAS_TEI_RERANK_URL must both be set (create ${ENV_FASTLANE}, see infra/README.md's k3d tier section). The retrieval half needs a real TEI endpoint to embed the query." >&2
  exit 1
fi

# --- 1. a temporary port-forward to CNPG's read write Service --------------------------------------
# CNPG's atlas-pg-rw Service is ClusterIP only by design (never published to the host); this is the
# smoke script's own, self contained way to reach it, matching what compose's own rag_tools.smoke.py
# gets for free via postgres:5433's host published port.

PG_PASSWORD="$(kubectl -n "${NAMESPACE}" get secret atlas-postgres-credentials -o jsonpath='{.data.password}' | base64 -d)"
if [[ -z "${PG_PASSWORD}" ]]; then
  echo "could not read the atlas-postgres-credentials Secret's password key (is the cluster up? task k3d:up)" >&2
  exit 1
fi

PF_LOG="$(mktemp)"
kubectl -n "${NAMESPACE}" port-forward svc/atlas-pg-rw "${PG_LOCAL_PORT}:5432" >"${PF_LOG}" 2>&1 &
PF_PID=$!
cleanup() {
  kill "${PF_PID}" >/dev/null 2>&1 || true
  wait "${PF_PID}" 2>/dev/null || true
  rm -f "${PF_LOG}"
}
trap cleanup EXIT

log "waiting for the atlas-pg-rw port-forward (127.0.0.1:${PG_LOCAL_PORT}) to be ready"
READY=0
for _ in $(seq 1 30); do
  if grep -q "Forwarding from" "${PF_LOG}" 2>/dev/null; then
    READY=1
    break
  fi
  sleep 1
done
if [[ "${READY}" -ne 1 ]]; then
  echo "kubectl port-forward never reported ready within 30s; its own output:" >&2
  cat "${PF_LOG}" >&2
  exit 1
fi

# --- 2. run the smoke script itself -----------------------------------------------------------------
# ATLAS_PG_DSN/ATLAS_INDEX_DIR are set here (not read from the deployed backend's own env, which this
# operator's own host process cannot see): the SAME PgvectorRetriever class the served backend uses,
# pointed at this tier's real data through the port-forward above and the SAME committed index build
# on disk rag-init already loaded from.

export ATLAS_PG_DSN="postgresql://atlas:${PG_PASSWORD}@localhost:${PG_LOCAL_PORT}/atlas"
export ATLAS_INDEX_DIR="${ROOT_DIR}/indexes/corpus-0.1.1-bge-m3-03f983e0"
export ATLAS_SMOKE_INGRESS_URL="http://localhost:${INGRESS_HTTP_PORT}"

log "running the rag smoke against the ingress (${ATLAS_SMOKE_INGRESS_URL})"
(cd "${ROOT_DIR}" && uv run --group record python "${INFRA_DIR}/scripts/k3d_smoke.py")
