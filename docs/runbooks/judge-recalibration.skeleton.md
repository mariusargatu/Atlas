# Runbook: judge recalibration (skeleton)

**Owning sub project.** Not SP5. D41's own ownership split names this a procedure that "belongs to
its own sub project": the sub project that rebuilds judge calibration under the dual plane
architecture's D15 identity rules (`judge_id` lineage, kappa >= 0.6, agreement >= 80%, AC1,
prevalence; recalibration required on any identity change) and the D25 contract regime this repo's
current `testing/harness/judge/` machinery already lives under. No sub project number is
assigned to this yet in the planning documents this task read; whoever picks it up should replace
this stub with the real procedure, not append to it piecemeal.

**Why this is a skeleton, not the content.** A full runbook already exists at
`docs/runbooks/judge-recalibration.md` (the "cassette era" version, D41's own phrase, marked
superseded per section 7.7 now that the dual plane architecture changes the judge's own operating
environment). SP5 owns infra, not the quality plane's judge machinery; writing the CONTENT of this
runbook (the actual trigger thresholds, the recalibration steps, the automation bar) without owning
or having rebuilt that machinery under the new architecture would be prose masquerading as authority.
This file exists only to hold the D41 inventory slot and point at where the real work lands.

## What the eventual runbook needs to cover (a checklist for whoever writes it, not instructions)

- The actual trigger cadence and conditions under the new architecture (the superseded version used
  a monthly cadence plus a staleness workflow; confirm both still apply).
- How `judge_id` lineage bearing (D15) interacts with a dual plane deployment: does a judge run in
  one plane, both, and does plane identity become part of the lineage key.
- Whether the existing `AUTOMATION_BAR` mechanism (`testing/harness/judge/calibration.py`, absorbed
  from the pre rewrite `evals/judge/calibration.py` by SP8's own task 2, its kappa gate now routed
  through `quality.gate.gate_on_lower_bound` rather than a hand rolled comparison) carries over
  unchanged or needs its own migration.
- A pointer to `docs/runbooks/judge-recalibration.md` (superseded) for the prior procedure's shape,
  useful as a starting draft, not as something to trust unmodified.
