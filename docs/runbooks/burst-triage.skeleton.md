# Runbook: burst triage

**Owning sub project.** SP5 (infra), an infra native procedure per D41's own ownership split.

**Trigger.** Anything unexpected during or after a burst session: `task burst:up` failed partway,
the weekly janitor (`.github/workflows/janitor.yml`) reported an orphan, a burst run's results look
wrong in a way the k3d tier never reproduced, or `task burst:destroy` exited non zero.

## First: is the tier actually still up

```bash
env HCLOUD_TOKEN=... infra/scripts/hcloud-orphans.sh
```

Lists every hcloud resource labeled `cluster=atlas-burst` (never a scan of the whole account, see
that script's own header comment on why). Empty output means the tier is already down; skip to the
janitor false positive section below. Any output means real resources exist and billing is live.

## `task burst:up` failed partway

1. Read which step failed: the credential gate (fixable per `infra/README.md`'s own Burst tier
   section), `tofu apply` (a real provisioning error, read the tofu error text directly, it is
   usually specific: quota, an invalid server type combination, a region capacity issue), or
   `helmfile sync` (a chart level failure, the same class of thing `task k3d:verify` already
   diagnoses on the k3d tier, since both tiers run the same charts).
2. If `tofu apply` itself failed and left partial resources: rerun `task burst:up` (tofu is
   idempotent, it reconciles from wherever state says the cluster is); if that also fails, prefer
   `task burst:destroy` and start clean over hand editing state.
3. If `helmfile sync` failed on a specific release: `helmfile -e burst sync -l name=<release>`
   against the burst kubeconfig (`infra/.kube/burst-config`) to retry just that one, the same
   pattern `infra/scripts/k3d-up.sh` already uses per release.

## `task burst:destroy` exited non zero

The script already reports which step failed and that the tier may be partially torn down (see its
own final message). Rerun it; it is idempotent by design (`infra/README.md`'s own "always()
discipline" section). If a rerun also fails, fall back to `tofu -chdir=infra/tofu/cluster destroy`
directly, or `hcloud <type> list -l cluster=atlas-burst` plus hand deletion as a last resort, then
confirm with `infra/scripts/hcloud-orphans.sh` that nothing remains.

## The janitor reported an orphan

1. Confirm it is real: `hcloud <type> list -l cluster=atlas-burst -o json` for the specific type the
   janitor named.
2. If tofu state still tracks it: `task burst:destroy` from wherever `infra/tofu/cluster` state
   points (may need `tofu init -backend-config=...` against the same R2 backend first, see the key
   rotation runbook's own R2 section if credentials changed since).
3. If tofu state does NOT track it (a resource created outside tofu, or state was lost): delete by
   hand with `hcloud <type> delete <id>`, then confirm with `infra/scripts/hcloud-orphans.sh`.
4. A genuine, intentionally active burst session running the same week the janitor fires is an
   accepted false positive (`infra/README.md`'s own note on this trade off); confirm it really is
   intentional (this session's own operator remembers starting it) before treating it as one.

## A burst result looks wrong the k3d tier never reproduced

Burst and k3d run the SAME charts (D3); a divergence usually means either real amd64 hardware
behavior the arm64 k3d tier could never exercise (see `infra/README.md`'s own TEI section for a
worked example of exactly this class of finding), or a burst only values difference
(`environments/burst/values.yaml`) that was never exercised locally. Diff the two environments'
rendered manifests directly: `helmfile -e local template` vs `helmfile -e burst template`
(`task infra:render`), read what actually differs, do not guess.
