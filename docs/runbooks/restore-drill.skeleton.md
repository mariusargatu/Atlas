# Runbook: CNPG restore drill

**Owning sub project.** SP5 (infra), an infra native procedure per D41's own ownership split.

**Trigger.** Quarterly, or before any burst session the operator wants strong confidence in
(headline benchmarks, public demos), or after any change to `infra/charts/cnpg-cluster`.

## Current state (read before running this)

`infra/charts/cnpg-cluster` (Task 2) configures `bootstrap.initdb` and `postInitApplicationSQL`
(the pgvector extension), not a `backup.barmanObjectStore` destination. The R2 bucket layout doc
(`infra/README.md`, `pg-backup/` prefix) names the TARGET this drill exercises once that wiring
lands; it is not yet real. This is a named, open gap, not a silent omission: wiring CNPG's own
`backup.barmanObjectStore` (Hetzner burst's `atlas-pg` `Cluster` CR) at the R2 `pg-backup/` prefix,
and a matching `ScheduledBackup` resource, is the next piece of work this runbook assumes exists.

## Steps (once barman backup wiring lands)

1. Confirm a recent scheduled backup exists: `kubectl -n atlas get backups.postgresql.cnpg.io`.
2. Record the current row counts this drill will verify survive a restore: `chunks` (expect 45 per
   the committed `corpus-0.1.1-bge-m3-03f983e0` build), the four `langgraph-checkpoint-postgres`
   tables, and the accounts table (whatever the live burst session has written).
3. Provision a SEPARATE `Cluster` CR (never restore over the live one) with
   `bootstrap.recovery.source` pointed at the same `pg-backup/` R2 prefix and object store
   credentials; wait for it to report `Ready`.
4. Re run the row count checks from step 2 against the recovered cluster
   (`infra/scripts/k3d-verify.sh`'s own psql pattern is the template: `kubectl exec` plus `psql` as
   the `postgres` superuser).
5. Tear the recovery cluster down; it exists only to prove the backup restores cleanly, never to
   serve traffic.
6. Record the result (row counts matched or did not, restore duration) in this runbook's own commit
   history or a dated note; there is no dashboard for this yet.

## What a failed drill means

A backup that does not restore is not a backup: treat a failed drill as equivalent to no backup
existing at all, escalate before the next burst session that would depend on it, and do not mark the
drill complete until a restore has actually been exercised end to end, not merely confirmed to exist
in R2.
