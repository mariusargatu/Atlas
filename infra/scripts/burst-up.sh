#!/usr/bin/env bash
# task burst:up (SP5 task 5): the Hetzner burst tier, complete code from the tofu apply onward, gated
# at two points before any real invocation: the credential check below, then a second, deliberate
# stop naming the one still unbuilt piece (burst's indexes restore from R2). Never invoked by this
# task's own implementation work (HARD SAFETY RULE 1: never create/modify/delete a real Hetzner
# resource); every real run of this script is the user's own, once their own credentials exist AND
# the indexes restore story below is implemented, per the SP5 plan
# Task 5's own framing ("the moment they exist, `task burst:up` is the complete path").
#
# Sequence: credential gate -> indexes restore stop (fail closed until built) -> tofu init (R2
# backend, generated at runtime, never committed) -> tofu apply -> restore the persisted wildcard TLS
# secret from R2 (best effort; a first ever run has nothing to restore) -> helmfile -e burst sync ->
# print the static DNS instructions (D3: external-dns dropped, a printed value, never an automated
# write).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INFRA_DIR="${ROOT_DIR}/infra"
TOFU_DIR="${INFRA_DIR}/tofu/cluster"
KUBECONFIG_OUT="${INFRA_DIR}/.kube/burst-config" # gitignored (infra/.kube/, .gitignore)
SSH_PUBKEY_PATH="${HOME}/.config/atlas-hetzner/id_ed25519.pub"
R2_BUCKET="atlas-durable" # infra/README.md's own R2 bucket layout doc
TLS_SECRET_KEY="tls/wildcard-cert.yaml" # D2 inventory: "the persisted TLS secret"

log() { printf '\n>>> %s\n' "$1"; }

# --- 0. .env.burst: the non secret, operator specific burst config -------------------------------
# Same discipline as .env.fastlane (Task 3): a real DNS domain and ACME contact email are operator
# specific, not secrets, and stay out of every committed file, injected here exactly the way
# k3d-up.sh already sources .env.fastlane for ATLAS_TEI_EMBED_URL/ATLAS_TEI_RERANK_URL.
ENV_BURST="${ROOT_DIR}/.env.burst"
if [[ -f "${ENV_BURST}" ]]; then
  log "sourcing ${ENV_BURST} for the burst domain and ACME contact"
  set -a
  # shellcheck disable=SC1090
  source "${ENV_BURST}"
  set +a
fi

# --- 1. the credential gate (fail closed, BEFORE any tofu/hcloud/helmfile invocation) -------------
# Every variable a real apply actually needs, named individually so the worded message says exactly
# which are missing, not a generic "credentials not configured." HCLOUD_TOKEN and the R2 keys are
# the two the task's own instruction names explicitly; the rest ("whatever the module needs") are the
# ones infra/tofu/cluster and infra/charts/atlas-cert actually reference, found by reading those
# files, not guessed.
missing=()
[[ -n "${HCLOUD_TOKEN:-}" ]] || missing+=("HCLOUD_TOKEN (Hetzner Cloud API token, project scoped)")
[[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || missing+=("AWS_ACCESS_KEY_ID (R2 S3 compatible access key)")
[[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]] || missing+=("AWS_SECRET_ACCESS_KEY (R2 S3 compatible secret key)")
[[ -n "${ATLAS_R2_ENDPOINT:-}" ]] || missing+=("ATLAS_R2_ENDPOINT (R2 account S3 endpoint, e.g. https://<accountid>.r2.cloudflarestorage.com)")
[[ -n "${ATLAS_BURST_DOMAIN:-}" ]] || missing+=("ATLAS_BURST_DOMAIN (the burst tier's base domain; set in ${ENV_BURST}, gitignored)")
[[ -n "${ATLAS_BURST_ACME_EMAIL:-}" ]] || missing+=("ATLAS_BURST_ACME_EMAIL (Let's Encrypt account contact; set in ${ENV_BURST}, gitignored)")
[[ -f "${SSH_PUBKEY_PATH}" ]] || missing+=("an SSH public key at ${SSH_PUBKEY_PATH} (generate with: ssh-keygen -t ed25519 -f ${SSH_PUBKEY_PATH%.pub}; see infra/README.md's Burst prerequisites)")

if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "task burst:up refused: the following are missing, nothing was applied:" >&2
  for m in "${missing[@]}"; do
    echo "  - ${m}" >&2
  done
  echo "See infra/README.md's Burst tier section for how to provision each one. This check runs before any tofu/hcloud/helmfile command." >&2
  exit 1
fi

log "credential gate passed: HCLOUD_TOKEN, R2 keys, ATLAS_BURST_DOMAIN, ATLAS_BURST_ACME_EMAIL, and the SSH public key are all present"

# The cloudflareApiToken itself is NOT a gate checked env var: it lives in
# infra/environments/burst/secrets.enc.yaml (SOPS + age, the same mechanism environments/local's own
# postgres password already proves), edited once via `sops environments/burst/secrets.enc.yaml`
# (infra/README.md), not sourced from the shell on every run. helmfile's own SOPS integration fails
# closed on a missing/misconfigured age identity (test_missing_age_key_failure_self_diagnoses_with_
# a_pointer_to_the_readme, testing/tests/test_infra_manifests.py) -- not checked again here, so this
# script never duplicates that guard.

# --- 1b. the indexes restore stop (fail closed, BEFORE any tofu/hcloud/helmfile invocation) -------
# infra/environments/burst/values.yaml's indexes.hostPath is still the k3d only path
# (/indexes/corpus-0.1.1-bge-m3-03f983e0), a single node hostPath mount shape. Task 5 never built the
# burst indexes restore story (pulling indexes/ down from R2 onto a node, or a PVC, before the backend
# release installs): that gap is real, not a stub left to fail obscurely once atlas-backend schedules
# onto a node with no such path. Stop here, worded, naming the owner, rather than let a real
# `task burst:up` proceed into a cluster that cannot serve its own retrieval index. See
# infra/README.md's "What Task 5 deliberately does not do" and docs/runbooks/corpus-bump.skeleton.md
# (the runbook that names this gap and its future, not yet assigned, owning sub project).
echo "task burst:up refused: burst's atlas-indexes restore from R2 is not yet implemented." >&2
echo "infra/environments/burst/values.yaml still points indexes.hostPath at the k3d only local path;" >&2
echo "nothing in this repo pulls indexes/ down from R2 onto a burst node or a PVC before the backend" >&2
echo "release installs. See infra/README.md's Burst tier section (What Task 5 deliberately does not" >&2
echo "do) and docs/runbooks/corpus-bump.skeleton.md, which names the future, not yet assigned, sub" >&2
echo "project that owns the retrieval index build pipeline and this wiring. Implementing it means" >&2
echo "restoring indexes/ from R2 onto the node or a PVC before the backend release installs. The" >&2
echo "burst postgres credentials story is equally unbuilt (no burst secrets.enc.yaml; the scaffold" >&2
echo "Secret renders only where .Values.postgres is set, and burst never sets it), see the same" >&2
echo "README list. Remove this stop only once BOTH the restore path and the burst credentials are" >&2
echo "real." >&2
exit 1

# --- 2. tofu init (R2 backend, generated at runtime) + apply --------------------------------------

BACKEND_CONFIG_FILE="$(mktemp)"
trap 'rm -f "${BACKEND_CONFIG_FILE}"' EXIT
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

log "tofu init (R2 backend: ${R2_BUCKET}/tofu-state/cluster.tfstate)"
tofu -chdir="${TOFU_DIR}" init -input=false -backend-config="${BACKEND_CONFIG_FILE}"

export TF_VAR_hcloud_token="${HCLOUD_TOKEN}"
export TF_VAR_ssh_public_key
TF_VAR_ssh_public_key="$(cat "${SSH_PUBKEY_PATH}")"
# TF_VAR_ssh_private_key deliberately never set: infra/tofu/cluster/variables.tf defaults it to null,
# kube-hetzner's own ssh-agent based auth path. `ssh-add ${SSH_PUBKEY_PATH%.pub}` before running this.

log "tofu apply (creates real, billed Hetzner resources: 1x CX23 control plane, 2x CX33 workers, a load balancer, a network, a firewall)"
tofu -chdir="${TOFU_DIR}" apply -auto-approve -input=false

mkdir -p "$(dirname "${KUBECONFIG_OUT}")"
tofu -chdir="${TOFU_DIR}" output -raw kubeconfig > "${KUBECONFIG_OUT}"
chmod 600 "${KUBECONFIG_OUT}"
LB_IPV4="$(tofu -chdir="${TOFU_DIR}" output -raw load_balancer_public_ipv4)"
export KUBECONFIG="${KUBECONFIG_OUT}"

log "cluster up. Load balancer public IPv4: ${LB_IPV4}"
echo "D3 (static DNS, external-dns dropped): point ${ATLAS_BURST_DOMAIN} and *.${ATLAS_BURST_DOMAIN} at ${LB_IPV4} now, at your DNS provider, by hand. This script never writes a DNS record."

# --- 3. restore the persisted wildcard TLS secret from R2, before cert-manager reconciles ---------
# D3: "one wildcard cert... persisted as a secret across teardowns." Best effort: a first ever burst
# spin up has nothing to restore yet, which is the honest bootstrap case, not a failure.
log "restoring the wildcard TLS secret from R2 (best effort; absent on a first ever run)"
if aws s3api head-object --endpoint-url "${ATLAS_R2_ENDPOINT}" --bucket "${R2_BUCKET}" --key "${TLS_SECRET_KEY}" >/dev/null 2>&1; then
  aws s3 cp --endpoint-url "${ATLAS_R2_ENDPOINT}" "s3://${R2_BUCKET}/${TLS_SECRET_KEY}" - | kubectl apply -f -
  echo "restored ${TLS_SECRET_KEY} from R2"
else
  echo "no persisted TLS secret found at s3://${R2_BUCKET}/${TLS_SECRET_KEY} (first ever run, or never backed up); cert-manager will solve a fresh DNS-01 challenge"
fi

# --- 4. helmfile sync (D3: the SAME charts the k3d tier already proves daily) ----------------------
# Tier parity strategy (C1 fix, SP6 final review, named here since the two tiers' sync strategies
# genuinely differ): this is a full, UNSELECTED `helmfile sync`, every release in
# infra/helmfile.yaml, ordered entirely by its own `needs:` graph, no per release `kubectl wait`
# between them. `infra/scripts/k3d-up.sh` deliberately does the opposite (one release at a time via
# `-l name=<release>`, an explicit `kubectl wait` between each) so a slow, staged local bringup can
# report exactly which stage it is on -- but that means EVERY new release added to helmfile.yaml
# must also be named, by hand, in k3d-up.sh's own list, or it silently never deploys on that tier
# (exactly the gap phoenix/otel-collector/atlas-monitoring fell into, C1). This script needs no
# equivalent update when a new release is added: a full sync picks it up automatically, which is why
# burst's own render test already covered these three releases before k3d-up.sh ever did.

log "helmfile -e burst sync"
(cd "${INFRA_DIR}" && helmfile -e burst sync)

log "burst tier up. kubeconfig: ${KUBECONFIG_OUT} (export KUBECONFIG=${KUBECONFIG_OUT} or use --kubeconfig)"
