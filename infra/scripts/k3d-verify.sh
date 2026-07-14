#!/usr/bin/env bash
# Live verification for `task k3d:up` (SP5 task 2 phase 1, extended by tasks 3 and 4): the things
# compose parity actually promises once everything reports ready -- the rag corpus is loaded (45
# rows, the committed indexes/corpus-0.1.1-bge-m3-03f983e0 build's own chunk_count), the checkpointer
# schema exists (the four tables `langgraph-checkpoint-postgres`'s own setup() creates), the local
# tier's external TEI endpoint serves the exact model revisions models.lock pins (Task 3), reachable
# both from this operator's own host AND from inside the cluster, and (Task 4) the backend/web
# Deployments are Available and the Traefik ingress actually serves the SPA end to end. Postgres
# checks connect as the CNPG superuser over the pod's local socket (no password needed, CNPG's own
# pg_hba trusts local connections) rather than fetching the app role's password out of the Secret: a
# superuser can SELECT from any table in any database in the cluster, and this script only ever
# reads.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INFRA_DIR="${ROOT_DIR}/infra"
NAMESPACE="atlas"
CLUSTER_NAME="atlas-pg"
POD="${CLUSTER_NAME}-1"
DATABASE="atlas"
# Task 4: matches infra/scripts/k3d-up.sh's own default, the host port mapped to k3s Traefik's "web"
# entryPoint at cluster create time.
INGRESS_HTTP_PORT="${INGRESS_HTTP_PORT:-8090}"

export SOPS_AGE_KEY_FILE="${SOPS_AGE_KEY_FILE:-$HOME/.config/atlas-age/keys.txt}"

ENV_FASTLANE="${ROOT_DIR}/.env.fastlane"
if [[ -f "${ENV_FASTLANE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FASTLANE}"
  set +a
fi
if [[ -z "${ATLAS_TEI_EMBED_URL:-}" || -z "${ATLAS_TEI_RERANK_URL:-}" ]]; then
  echo "ATLAS_TEI_EMBED_URL and ATLAS_TEI_RERANK_URL must both be set (create ${ENV_FASTLANE}, see infra/README.md's k3d tier section)." >&2
  exit 1
fi

log() { printf '\n>>> %s\n' "$1"; }

psql_query() {
  kubectl -n "${NAMESPACE}" exec "${POD}" -c postgres -- psql -U postgres -d "${DATABASE}" -tAc "$1"
}

log "waiting for pod ${POD} to be ready"
kubectl -n "${NAMESPACE}" wait --for=condition=Ready "pod/${POD}" --timeout=120s

log "chunks table row count (expect 45: indexes/corpus-0.1.1-bge-m3-03f983e0/build_manifest.json's own chunk_count)"
ROW_COUNT="$(psql_query 'SELECT count(*) FROM chunks;' | tr -d '[:space:]')"
echo "chunks: ${ROW_COUNT} rows"
if [[ "${ROW_COUNT}" != "45" ]]; then
  echo "expected 45 rows in chunks, got '${ROW_COUNT}'" >&2
  exit 1
fi

log "checkpointer tables present (langgraph-checkpoint-postgres's own setup(), applied by alembic revision 0001)"
CHECKPOINT_TABLES="$(psql_query "SELECT string_agg(table_name, ',' ORDER BY table_name) FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('checkpoints', 'checkpoint_blobs', 'checkpoint_writes', 'checkpoint_migrations');")"
echo "checkpoint tables found: ${CHECKPOINT_TABLES}"
EXPECTED="checkpoint_blobs,checkpoint_migrations,checkpoint_writes,checkpoints"
if [[ "${CHECKPOINT_TABLES}" != "${EXPECTED}" ]]; then
  echo "expected all 4 checkpoint tables (${EXPECTED}), got '${CHECKPOINT_TABLES}'" >&2
  exit 1
fi

log "pgvector extension installed"
VECTOR_EXT="$(psql_query "SELECT extversion FROM pg_extension WHERE extname = 'vector';" | tr -d '[:space:]')"
echo "vector extension version: ${VECTOR_EXT:-<not installed>}"
if [[ -z "${VECTOR_EXT}" ]]; then
  echo "pgvector extension not installed in database ${DATABASE}" >&2
  exit 1
fi

log "checking the external TEI endpoints' /info revisions against models.lock from this host (scripted, not eyeballed)"
python3 "${INFRA_DIR}/scripts/verify_tei_revisions.py"

log "running the in cluster connectivity-check Jobs again (proves a POD, not just this host, reaches the same endpoints)"
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=tei-embed)
(cd "${INFRA_DIR}" && helmfile -e local sync -l name=tei-rerank)

log "backend and web Deployments Available (Task 4)"
kubectl -n "${NAMESPACE}" wait --for=condition=Available deployment/atlas-backend --timeout=60s
kubectl -n "${NAMESPACE}" wait --for=condition=Available deployment/atlas-web --timeout=60s

log "ingress reachable end to end at http://localhost:${INGRESS_HTTP_PORT} (Traefik -> atlas-web -> nginx)"
INGRESS_STATUS="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "http://localhost:${INGRESS_HTTP_PORT}/")"
echo "GET / through the ingress: HTTP ${INGRESS_STATUS}"
if [[ "${INGRESS_STATUS}" != "200" ]]; then
  echo "expected HTTP 200 from the ingress root (the served SPA's index.html), got ${INGRESS_STATUS}. Was the cluster created before Task 4's ingress port mapping existed? Run 'task k3d:down' then 'task k3d:up' once to pick it up (see infra/README.md)." >&2
  exit 1
fi

log "live verification passed: 45 rows in chunks, all 4 checkpoint tables present, pgvector ${VECTOR_EXT} installed, external TEI endpoints verified, backend/web Available, ingress serving HTTP 200"
