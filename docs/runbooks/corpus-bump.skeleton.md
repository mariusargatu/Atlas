# Runbook: corpus_version bump (skeleton)

**Owning sub project.** Not SP5. D41's own ownership split names this a procedure that "belongs to
its own sub project": whichever sub project owns `corpus_tools`/`rag_tools` and the retrieval index
build pipeline (`indexes/`, `rag_tools.ingest`, the TEI embedding pins). No sub project number is
assigned to this yet in the planning documents this task read.

**Why this is a skeleton, not the content.** SP5 (infra) consumes a corpus build (mounts
`indexes/<corpus_version>` read only into both the k3d and burst tiers, `infra/charts/atlas-indexes`,
`ATLAS_INDEX_DIR`), it does not build or version one. Writing the actual bump procedure without
owning that pipeline would be guessing at steps this task never exercised.

## What the eventual runbook needs to cover (a checklist for whoever writes it, not instructions)

- The HLD's own corpus bump migration rule (the HLD section on registry derived fields: "when
  corpus_version bumps, registry derived fields re render mechanically") and what triggers a bump
  versus what is a routine content edit within the same version.
- How a bump propagates to the infra layer this runbook's own owner does NOT control: `indexes.corpusDir`
  in `infra/environments/base/values.yaml`, `ATLAS_INDEX_DIR` in `infra/charts/atlas-backend`, and the
  `indexes/` host path both `infra/scripts/k3d-up.sh` (local) and a future burst equivalent read from
  (today: a k3d node hostPath locally, R2's own `indexes/` prefix per `infra/README.md`'s bucket
  layout doc on burst, not yet wired into a Job that pulls it down). That wiring gap on the burst
  side is real and open, named here rather than silently assumed solved.
- Whether a corpus bump requires a judge recalibration (the HLD's own pin movement policy couples
  `models.lock` changes to judge recalibration and a noise floor rerun; confirm whether a corpus bump
  carries the same coupling).
- A pointer to `infra/README.md`'s "R2 bucket layout" section for the `indexes/` prefix this
  eventual pipeline should read from once it exists.
