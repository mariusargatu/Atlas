# Runbook: judge recalibration

**Trigger.** Monthly cadence; the `staleness` workflow failing (newest dated artifact in
`testing/harness/judge/artifacts/live_provisional/` older than 45 days); any change to a judge
rubric or judge model; or a drift investigation that implicates the judge rather than the agent.

**Why monthly.** Both ends of the comparison drift: pinned provider snapshots get deprecated or
silently re served, and the humans' criteria move as they grade. A judge calibrated once and never
re sampled goes stale without any code changing.

## Two separate numbers, never conflated (KAPPA HONESTY)

`task judge-live` (step 1 below) produces TWO PROVISIONAL signals only: registry truth agreement
(manufactured cases with ground truth by construction, `judge.provisional.registry_truth_agreement`)
and judge vs judge kappa (agreement between two judge contracts, no ground truth at all,
`judge.provisional.judge_vs_judge_kappa`). NEITHER licenses a production deployment;
`judge.provisional`'s own source never even imports the deployment gate, and
`testing/tests/test_judge_provisional_honesty.py` proves that by inspection. The ONLY number that
licenses a deployment is Cohen's kappa against REAL human labels
(`judge.calibration.CalibrationReport.licensed`, D15), collected through the HITL adjudication
page and computed by `judge.calibration.calibrate()`. This runbook covers both procedures in
separate steps so neither number is ever read as the other.

## Steps

1. Run the keyed live provisional probe (needs `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`; never
   the PR lane): trigger the `judge-live` workflow from the Actions tab, or locally
   `task judge-live`. This overwrites both `latest.md` and the dated snapshot under
   `testing/harness/judge/artifacts/live_provisional/` with identical content, each number labeled
   by its own source and closing with an explicit statement that neither licenses deployment. Diff
   `latest.md` against its last committed state
   (`git diff -- testing/harness/judge/artifacts/live_provisional/latest.md`) to see what moved,
   then commit both files: there is no bar to clear here, only a provisional reading to record.
2. Real, deployment licensing calibration needs the human labelled set: once the roughly 200 item
   HITL adjudication session has produced real labels, run them through
   `judge.calibration.calibrate()` against the judge's current verdicts to get a
   `CalibrationReport`. Decide against `AUTOMATION_BAR` in `testing/harness/judge/calibration.py`
   (routed through `quality.gate.gate_on_lower_bound`, the same rule a release uses):
   - **At or above the bar** (the interval's lower bound, not the point): the judge is licensed to
     automate this metric.
   - **Below the bar**: the judge is demoted, its verdicts become review queue routing signals,
     not graded results, until a rubric fix restores agreement. Fix the rubric, not the labels.
3. Any rubric or judge model change voids every previous calibration by definition, both the
   provisional readings and any real human calibration: repeat step 1 (and step 2, once real
   labels exist) before the judge grades anything again. Scores across judge versions are not
   comparable.
4. Refresh the human labelled set periodically with fresh triaged production cases (see the
   promotion runbook) so calibration measures today's failure shapes, not an old one.

**Never** restate an agreement number in prose, a task description, or a docstring without its
source label attached (registry truth agreement, judge vs judge kappa, or human kappa): reference
the committed artifact and say which of the three it is. A hand copied, unlabeled number drifts and
invites exactly the conflation this runbook exists to prevent; the artifact is the record.
