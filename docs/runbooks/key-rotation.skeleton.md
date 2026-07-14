# Runbook: key rotation

**Owning sub project.** SP5 (infra), an infra native procedure per D41's own ownership split.

**Trigger.** Suspected exposure of any credential this repo's infra layer depends on; routine
rotation on whatever cadence the operator sets (no automated schedule exists); a lost age private
key; adding or replacing an operator (SOPS recipients).

## SOPS age keys (D41: two recipients, never one root secret)

1. Generate a fresh keypair: `age-keygen -o ~/.config/atlas-age/atlas-<name>.txt` (never inside this
   repo checkout).
2. Add the new public key to `infra/.sops.yaml`'s `creation_rules` alongside the existing ones
   (never replace both at once: at least one previously valid recipient's private key must remain
   available to re wrap the data key).
3. Re encrypt every `*.enc.yaml` file with the new recipient list:
   `sops updatekeys infra/environments/local/secrets.enc.yaml` and the same for
   `infra/environments/burst/secrets.enc.yaml`.
4. Verify both files still decrypt with EVERY current recipient's private key independently (the
   same proof `infra/README.md`'s own SOPS section documents for the original two keys), then
   remove the retired public key from `infra/.sops.yaml` once every encrypted file is confirmed
   updated.
5. Run `task test` (the render test, `testing/tests/test_infra_manifests.py`, decrypts both
   secrets files through the real helmfile path) to prove the rotation did not break rendering.

## Hetzner (`HCLOUD_TOKEN`)

1. Create a new project scoped API token in the Hetzner Cloud console.
2. Update wherever the operator's shell exports `HCLOUD_TOKEN` from (never this repo's own `.env`;
   see `infra/README.md`'s own "Credentials" section on why).
3. If a GitHub Actions `HCLOUD_TOKEN` secret is also configured (`.github/workflows/janitor.yml`),
   rotate it there too, in the repo's Settings, Secrets and variables, Actions.
4. Revoke the old token in the Hetzner console only after confirming a fresh
   `task burst:up`/`burst:destroy` cycle (or, cheaply, `infra/scripts/hcloud-orphans.sh` alone)
   succeeds with the new token.

## R2 (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`)

1. Create a new R2 API token (Cloudflare dashboard, scoped to the `atlas-durable` bucket only, not
   account wide).
2. Update the operator's own shell export; the tofu S3 compatible backend and `aws` CLI both read
   these from the environment, never a committed file.
3. Confirm `tofu -chdir=infra/tofu/cluster init -backend-config=...` still reattaches to the
   existing state before revoking the old token.

## Cloudflare DNS-01 API token

1. Create a new token, DNS edit scoped to the `ATLAS_BURST_DOMAIN` zone only.
2. `sops infra/environments/burst/secrets.enc.yaml`, replace `cloudflareApiToken`, save (SOPS re
   encrypts on write).
3. `task test` proves the new file still decrypts and renders; a live rotation check needs a real
   burst session (`task burst:up`, confirm the `atlas-cloudflare-api-token` Secret in the
   `cert-manager` namespace carries the new value).

## SSH keypair (`~/.config/atlas-hetzner/`)

Rotating this means new nodes: kube-hetzner has no in place SSH key swap for already provisioned
servers. Generate a new keypair, `ssh-add` it, and the next `task burst:up` after a
`task burst:destroy` uses it; there is no partial rotation path for a live cluster.
