# Runbook: model deprecation response (skeleton)

**Owning sub project.** Not SP5. D41's own inventory names "model deprecation response" as its own
procedure; the ownership split sentence that explicitly names judge recalibration and corpus_version
bump as belonging to "their respective sub projects" does not separately name this one, but it is
also absent from SP5's own explicitly owned list (burst bringup, CNPG restore drill, key rotation,
burst triage), so it belongs with the other two: a future sub project's job, most likely whichever
owns the model gateway and `models.lock` (provider client code, retry and fallback ladder, the pin
movement policy). No sub project number is assigned to this yet in the planning documents this task
read.

**Why this is a skeleton, not the content.** SP5 (infra) never touches `models.lock`, the model
gateway, or provider client code (explicit off limits territory for this task); writing the actual
response procedure without owning that surface would be guessing.

## What the eventual runbook needs to cover (a checklist for whoever writes it, not instructions)

- How a provider deprecation notice (a model id or snapshot being retired) is detected: today
  nothing in this repo watches for this automatically, the same gap `infra/README.md`'s own
  `postgres-pgvector` Dockerfile pin comment names for its own apt package version ("nothing bumps
  this automatically").
- The HLD's own pin movement policy tiers by blast radius: `models.lock` changes are manual, coupled
  to judge recalibration and a noise floor rerun (not the hermetic lane's automatic dependency bot
  path). A deprecation response should follow that same coupling, not skip straight to a code change.
- Whether a deprecated model affects only the live/record lane (this repo's hermetic `task test`
  lane never calls a real model) or also invalidates committed cassettes recorded against the
  now retired snapshot, and if so which ones need re recording.
- Whether burst's own `ATLAS_FALLBACK_MODEL` passthrough (`infra/charts/atlas-backend`, the SP4
  final fix wave's own provider_fallback rung) is the intended stopgap while a full migration off a
  deprecated model completes, or a separate mechanism is expected.
