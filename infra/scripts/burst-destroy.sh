#!/usr/bin/env bash
# task burst:destroy (SP5 task 5): tear the Hetzner burst tier down completely. Never invoked by
# this task's own implementation work (HARD SAFETY RULE 1); every real run is the user's own.
#
# "destroy in always() discipline scripted" (this task's own instruction): every step past the
# credential gate runs best effort and the script always attempts the FULL sequence (backup, then
# destroy, then the orphan sanity check) rather than aborting on the first non critical failure --
# `set -e` is deliberately NOT used here (unlike burst-up.sh), because a script that dies on step 2
# and skips tofu destroy entirely is the exact opposite of "always() discipline." The final exit code
# reflects `tofu destroy` itself (the one step that must actually succeed for this to be a real
# teardown), reported after every other step has already had its chance to run.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INFRA_DIR="${ROOT_DIR}/infra"
TOFU_DIR="${INFRA_DIR}/tofu/cluster"
KUBECONFIG_OUT="${INFRA_DIR}/.kube/burst-config"
R2_BUCKET="atlas-durable"
TLS_SECRET_KEY="tls/wildcard-cert.yaml"

log() { printf '\n>>> %s\n' "$1"; }

ENV_BURST="${ROOT_DIR}/.env.burst"
if [[ -f "${ENV_BURST}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_BURST}"
  set +a
fi

# --- 1. the credential gate (fail closed, BEFORE any tofu/hcloud/helmfile/aws invocation) ---------
missing=()
[[ -n "${HCLOUD_TOKEN:-}" ]] || missing+=("HCLOUD_TOKEN (Hetzner Cloud API token, project scoped)")
[[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || missing+=("AWS_ACCESS_KEY_ID (R2 S3 compatible access key)")
[[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]] || missing+=("AWS_SECRET_ACCESS_KEY (R2 S3 compatible secret key)")
[[ -n "${ATLAS_R2_ENDPOINT:-}" ]] || missing+=("ATLAS_R2_ENDPOINT (R2 account S3 endpoint, e.g. https://<accountid>.r2.cloudflarestorage.com)")
SSH_PUBKEY_PATH="${HOME}/.config/atlas-hetzner/id_ed25519.pub"
[[ -f "${SSH_PUBKEY_PATH}" ]] || missing+=("an SSH public key at ${SSH_PUBKEY_PATH} (tofu's own ssh_public_key variable has no default, needed even to destroy: see infra/README.md)")

if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "task burst:destroy refused: the following are missing, nothing was destroyed:" >&2
  for m in "${missing[@]}"; do
    echo "  - ${m}" >&2
  done
  echo "See infra/README.md's Burst tier section for how to provision each one. This check runs before any tofu/hcloud/aws command." >&2
  exit 1
fi

log "credential gate passed: HCLOUD_TOKEN, R2 keys, and the SSH public key are all present"

# --- 2. back up the wildcard TLS secret to R2, best effort, BEFORE tearing the cluster down -------
# D3: "persisted as a secret across teardowns." A cluster that never finished coming up (backend up
# failed mid sequence) has no such secret; that is not fatal to a destroy, which must still run.
if [[ -f "${KUBECONFIG_OUT}" ]]; then
  log "backing up the wildcard TLS secret to R2 (best effort)"
  if KUBECONFIG="${KUBECONFIG_OUT}" kubectl -n atlas get secret atlas-wildcard-tls -o yaml > /tmp/atlas-wildcard-tls.yaml 2>/dev/null; then
    if aws s3 cp --endpoint-url "${ATLAS_R2_ENDPOINT}" /tmp/atlas-wildcard-tls.yaml "s3://${R2_BUCKET}/${TLS_SECRET_KEY}"; then
      echo "backed up atlas-wildcard-tls to s3://${R2_BUCKET}/${TLS_SECRET_KEY}"
    else
      echo "WARNING: could not upload the TLS secret backup to R2; continuing with destroy anyway (always() discipline)" >&2
    fi
  else
    echo "no atlas-wildcard-tls secret found (cluster never finished coming up, or cert-manager has not issued yet); nothing to back up"
  fi
  rm -f /tmp/atlas-wildcard-tls.yaml
else
  echo "no kubeconfig at ${KUBECONFIG_OUT}; skipping the TLS secret backup (cluster may already be gone)"
fi

# --- 3. tofu destroy -------------------------------------------------------------------------------

BACKEND_CONFIG_FILE="$(mktemp)"
cat > "${BACKEND_CONFIG_FILE}" <<EOF
bucket                      = "${R2_BUCKET}"
key                         = "tofu-state/cluster.tfstate"
region                      = "auto"
use_path_style              = true
skip_credentials_validation = true
skip_region_validation      = true
skip_requesting_account_id  = true
endpoints = {
  s3 = "${ATLAS_R2_ENDPOINT}"
}
EOF

log "tofu init (reattaching to the R2 backend)"
tofu -chdir="${TOFU_DIR}" init -input=false -backend-config="${BACKEND_CONFIG_FILE}"
rm -f "${BACKEND_CONFIG_FILE}"

export TF_VAR_hcloud_token="${HCLOUD_TOKEN}"
export TF_VAR_ssh_public_key
TF_VAR_ssh_public_key="$(cat "${SSH_PUBKEY_PATH}")"

log "tofu destroy (deletes every real Hetzner resource this module created)"
destroy_status=0
tofu -chdir="${TOFU_DIR}" destroy -auto-approve -input=false || destroy_status=$?

rm -f "${KUBECONFIG_OUT}"

# --- 4. the shared orphan sanity check (best effort, informational; the real weekly enforcement is
# .github/workflows/janitor.yml) --------------------------------------------------------------------

log "post destroy orphan check (infra/scripts/hcloud-orphans.sh)"
"$(dirname "${BASH_SOURCE[0]}")/hcloud-orphans.sh" || echo "WARNING: orphan check reported leftover resources or failed to run; investigate by hand (see the message above)." >&2

if [[ "${destroy_status}" -ne 0 ]]; then
  echo "tofu destroy exited ${destroy_status}; the burst tier may be partially torn down. Rerun 'task burst:destroy' (idempotent) or investigate with 'tofu -chdir=${TOFU_DIR} state list'." >&2
fi
exit "${destroy_status}"
