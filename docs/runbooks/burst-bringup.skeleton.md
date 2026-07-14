# Runbook: burst bringup

**Owning sub project.** SP5 (infra), an infra native procedure per D41's own ownership split: this
runbook documents machinery SP5 already built (`infra/tofu/cluster`, `infra/scripts/burst-up.sh`,
`infra/charts/atlas-cert`), not a future placeholder.

**Trigger.** A public demo, a load lane run, or a headline benchmark (D3: the only three reasons the
burst tier exists; local numbers are never quoted). Manual only, never a standing cadence.

## Prerequisites

See `infra/README.md`'s own "Burst tier" section for the full credential and tool list. In short:
`HCLOUD_TOKEN`, R2 keys (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`ATLAS_R2_ENDPOINT`), an SSH
keypair at `~/.config/atlas-hetzner/` with the private half loaded in `ssh-agent`, `.env.burst`
(`ATLAS_BURST_DOMAIN`/`ATLAS_BURST_ACME_EMAIL`), and a real Cloudflare DNS-01 token edited into
`infra/environments/burst/secrets.enc.yaml` via `sops`.

The R2 backend's S3 compatibility flags (`use_path_style`, `skip_requesting_account_id`, and the rest
of `infra/tofu/cluster/backend.tf`'s generated block) were written from documented convention, never
verified against a live R2 bucket; the very first `tofu init` this runbook triggers may need an
additional backend flag, not only a credential.

## Steps

1. `task burst:up`. Refuses closed with a worded message if any credential above is missing, before
   any `tofu`/`hcloud`/`helmfile` command runs. It also stops, worded, right after that gate: burst's
   `atlas-indexes` restore from R2 is not yet implemented (see `infra/README.md`'s "What Task 5
   deliberately does not do" and `docs/runbooks/corpus-bump.skeleton.md`), so this runbook cannot be
   completed for a real burst session until that restore path exists. A cold apply is real cloud
   provisioning, not minutes, budget real time for node boot, k3s join, and the CNPG/TEI warmup
   allowances `infra/README.md` already documents for the k3d tier (the same charts, real amd64
   hardware here instead).
2. Point `ATLAS_BURST_DOMAIN` and `*.ATLAS_BURST_DOMAIN` at the load balancer IPv4 the script prints
   (D3: static DNS, no automation writes this record).
3. Wait for the wildcard certificate: `kubectl get certificate atlas-wildcard -n atlas` until
   `READY: True`, or confirm the persisted secret restored cleanly from R2 (`burst-up.sh`'s own log
   line names which happened).
4. **Not yet implemented** (named here as the open gap, not silently skipped): a sentinel probe go or
   no go gate (D41's own phrase) that runs a scripted smoke check against the live burst endpoint
   before any real traffic, demo, or benchmark run is pointed at it, and refuses to proceed on a red
   signal. `infra/scripts/k3d-smoke.sh`/`k3d_smoke.py` are the k3d tier's own equivalent; a burst
   flavored port of that script, run automatically at the end of `task burst:up`, is the natural next
   step, not built by this task.
5. Run whatever the burst session actually needs (a demo, `task benchmark` against the burst
   endpoint, a load sweep); this runbook's own scope ends at "the tier is up and verified reachable,"
   not at what any particular session does with it.
6. When done: run the burst triage runbook if anything looked wrong, otherwise go straight to
   `task burst:destroy` (see that command's own always run discipline in `infra/README.md`). Never
   leave the burst tier up between sessions; the weekly janitor
   (`.github/workflows/janitor.yml`) exists specifically to catch a forgotten teardown.
