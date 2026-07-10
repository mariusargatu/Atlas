# Measurement: SP10 CI lanes, the local only and small n honesty notes

Committed factual record for SP10 Task 6, the last task of the last sub project (SP11 and SP12 were
cancelled by user directive on 2026-07-21; see Section 7). Follows `docs/measurements/
sp3-rag-spine.md`, `docs/measurements/sp7-datasets-metrics.md`, `docs/measurements/
sp8-judge-human-loop.md`, and `docs/measurements/sp9-variants-matrix.md`'s own pattern: every claim
below is source labeled and cross referenced to the specific test, workflow, or Taskfile target that
proves it. MECHANICS are gated by the hermetic suite; LIVE outcomes are named as not yet run, never
silently implied.

## LOCAL ONLY, DORMANT UNTIL PUSHED: what "verified" means in this document

Every workflow file this sub project wrote or reviewed (`ci.yml`, `live-pr.yml`, `full-sweep.yml`,
`simulator.yml`, `burst-benchmark.yml`) has never been pushed and has never run on GitHub. That
posture is deliberate, not an oversight, and it is the same one `janitor.yml` and `release.yml`
already carried before SP10 started. "VERIFIED" therefore has exactly one meaning throughout this
document, repeated from the plan's own Global Constraints rather than softened anywhere below:
actionlint clean, every referenced Taskfile target and script actually exists and runs where it can
run locally without a live provider or a live billed tier, every secret name a workflow reads matches
what the script or driver it feeds actually checks for, and NEVER "observed running on GitHub." A
maintainer still has to push this branch, configure the `burst-benchmark` protected environment's
required reviewers, and add every secret each workflow reads before any of these five lanes runs for
real. Section 1's table names, lane by lane, exactly how much of each file that claim actually covers
today.

## 1. The five lanes: actionlint validated only, hermetically tested, and locally exercised

Five HLD section 7.3 lanes now have a named, purpose built workflow file (SP10 Tasks 1 through 5).
None of the four new files (`live-pr.yml`, `full-sweep.yml`, `simulator.yml`, `burst-benchmark.yml`)
has ever executed; `ci.yml` is the one exception, since its own steps ARE the commands a developer
already runs locally on every commit. The table below states, for each lane, exactly which parts fall
into which of the three buckets the task's own spec names.

| Lane (workflow file) | Trigger, gates merge? | Actionlint validated only today (needs a push, a secret, or a live billed tier to run for real) | Hermetically tested today (`task test`, keyless, REPLAY gateways) | Locally exercised today (a real command that already ran clean in a verification worktree) |
|---|---|---|---|---|
| Hermetic (`ci.yml`) | every PR and push to main; the ONLY unconditional gate (D18) | the `e2e` job's Playwright run (no Taskfile target mirrors it byte for byte, per SP10 Task 1's own finding) and the `security` job's Trivy scan (a third party action with no Taskfile equivalent at all) | `testing/tests/test_ci_workflow_targets.py` (4 tests): reads `ci.yml` and `Taskfile.yml` fresh and fails the moment the mirrored `lint`/`test`/`web-test` commands drift apart | `task lint`, `uv run pytest -q` (run twice, proving byte stability), and `task web-test` all ran clean in a plain `uv sync` worktree (SP10 Task 1's own verification); this is the one lane whose entire hermetic content is also the same gate every commit in this repo already passes through, not a separate thing |
| Live PR (`live-pr.yml`) | `pull_request`, paths filtered on `backend/atlas/**`/`corpus/**`/`contracts/**`/`testing/harness/quality,judge,dataset_tools/**`; gates ONLY on deterministic floors (D18) | the `live-sweep` job itself (needs a real Postgres/TEI stack plus at least one of `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`; never invoked) | `testing/tests/test_live_pr_workflow.py` (9 tests, path filter globs and target/env var wiring) plus `testing/tests/test_judge_live_pr_lane.py` (22 tests, the full `run()` proven end to end against REPLAY gateways, both a clean pass and a `hallucination_bait` floor failure) | `task contracts:diff REF=main` (the git aware floor, its own separate job) runs clean and keyless right now against this branch's own local main; the workflow's own `REF=origin/main` form only runs clean once this branch is pushed, since `origin/main` is a stale pre rewrite remote head today, not a defect of this commit; `task live-pr:sweep` needs a live provider key and was not run |
| Full sweep (`full-sweep.yml`) | `push` to `main` only; never gates (D18) | the `full-sweep` job itself (same live retrieval plus judge key precondition; never invoked) | `testing/tests/test_full_sweep_workflow.py` (10 tests) plus `testing/tests/test_judge_full_sweep.py` (11 tests, the full `run()` proven end to end against REPLAY gateways, a mixed PASS/FAIL rate and an all FAIL run both proving nothing gates) | none of the live path ran without a provider key; the workflow's own YAML plus the driver's assembly functions are what is exercised today |
| Simulator (`simulator.yml`) | `workflow_dispatch` only; never gates (D18, HLD table) | the `live-simulation` job's live cross model loop (needs BOTH `OPENAI_API_KEY` AND `ANTHROPIC_API_KEY` plus a reachable TEI endpoint; never invoked) | `testing/tests/test_simulator_workflow.py` (13 tests) plus `testing/tests/test_judge_simulator_lane.py` (22 tests, the full persona loop proven end to end with REPLAY gateways on all three roles: persona player, SUT graph, evaluator) | `task simulation` (the deterministic roster plus the mind changer fixture proof, zero keys) runs clean right now, and is the workflow's own unconditional first step |
| Burst benchmark (`burst-benchmark.yml`) | `workflow_dispatch`, `environment: burst-benchmark`; never gates (D18, HLD table) | almost the entire pipeline: `task burst:up`, the sentinel gate's own live invocation, the `xk6-sse` binary build, `task matrix:live`, `task load:k6`/`task load:join`, and `task burst:destroy` all need a real, billed Hetzner tier plus every secret the protected environment would hold; none of it ran | `testing/tests/test_sentinel_ci_gate.py` (11 tests, tamper proven: flips exactly one probe class red at a time and asserts the gate goes NO GO every time) plus `testing/tests/test_burst_benchmark_workflow.py` (28 tests, including the exhaustive `always()` fire drill: a pure simulator of GitHub Actions' own step condition semantics proves the destroy step still runs no matter which earlier step index fails, not just the three named examples) | none of the live pipeline ran; `infra/scripts/burst-up.sh`/`burst-destroy.sh` are pre existing SP5 scripts, already proven credential gated fail closed in SP5's own verification, not re run by this task |

**Reading this table honestly**: the Hermetic lane is qualitatively different from the other four, not
just first in the list. Its content is not a separate thing SP10 built and hopes works once pushed; it
is the literal `task lint`/`task test`/`task web-test` commands every contributor already runs, wrapped
in workflow YAML. The other four lanes are complete, reviewed, actionlint clean code whose ASSEMBLY
(every function the live driver calls) is hermetically proven end to end against REPLAY gateways, but
whose LIVE invocation, the actual point of each lane, has never once executed. That gap is not a defect
in SP10's own work; it is what "local only" means for a lane whose entire purpose is to make a live
call, precisely the honesty the plan's own Global Constraints demand this document state plainly
rather than let a reader infer confidence the record does not support.

**The hermetic gate's own clean baseline needed one honest correction along the way, worth recording
here since it directly bears on every "hermetically tested" cell above.** SP10 Task 1 discovered that
the plan's own stated baseline, "2072 passed, 2 skipped, 15 deselected," was a composite of two
different environments, not a single reproducible number: a long lived, `--all-groups` contaminated
dev virtualenv reads 2072 passed with 0 skipped (`deepeval` happens to be importable there), while a
genuinely clean `git worktree add --detach` plus a plain `uv sync` reads 2070 passed, 2 skipped, 15
deselected, the number this document's own final verification (Section "Verification" of the task
report this document accompanies) reproduces again. `2070 + 2 = 2072`: both numbers describe the
identical collected suite, differing only by whether `deepeval` happens to be importable, never by a
difference in the suite itself. Every task in this sub project measured its own clean delta against
this corrected baseline: Task 1 add 4 (2074), Task 2 add 31 (2105), Task 3 add 21 (2126), Task 4 add 35
(2161), Task 5 add 39 (2200), a byte stable double run confirmed identical at every step, zero
regressions across the whole sub project.

## 2. The OIDC to protected environment substitution, named plainly

HLD section 7.3's own table names the Burst benchmark row's trigger as "manual, protected
environment... OIDC tofu apply." Hetzner Cloud has no OIDC or STS federation mechanism at all, unlike
AWS or GCP (verified against `infra/scripts/burst-up.sh` and `infra/README.md`'s own credentials
table, both of which authenticate via a static, project scoped `HCLOUD_TOKEN`, never a federated token
exchange; `research/15-ci-lanes.md`'s own worked OIDC example is itself an AWS one, not a Hetzner one).
A literal "OIDC tofu apply" step cannot be built against this provider without an intermediary this
repository does not have, a secrets broker federating GitHub's own OIDC token into a vault that in
turn holds `HCLOUD_TOKEN`, real added infrastructure with its own blast radius, out of scope here.

The substitution, documented directly in `burst-benchmark.yml`'s own header, the same discipline
`ADR-030` already applied to the local generator
swap: `environment: burst-benchmark`, a GitHub Actions protected environment with required reviewers
configured on it, is the human approval gate before any real spend, satisfying the decision's real
intent (a human clicks approve before money is spent) through a different mechanism than literal
token federation, exactly the substitution research 15's own words describe: "Put this workflow
behind a GitHub environment with required reviewers, that is the native expensive lane needs a human
click mechanism." What is NOT preserved, named plainly rather than blurred: there is no cryptographic
proof binding this specific workflow run's identity to the credential the way OIDC's short lived,
audience scoped token would; the static `HCLOUD_TOKEN` this job reads is a long lived credential,
exactly like `janitor.yml`'s own `HCLOUD_TOKEN` already is.

A second limitation, equally load bearing for what "done" means under LOCAL ONLY: the protected
environment's required reviewers rule is a repository Settings action (Settings, Environments,
`burst-benchmark`, Required reviewers), configured by a maintainer after a real push, not committed
code. `burst-benchmark.yml` can and does declare `environment: burst-benchmark`, but this file alone
cannot prove that environment is actually protected. This is the same class of limit `janitor.yml`'s
`HCLOUD_TOKEN` secret and `release.yml`'s GHCR push already carry: complete, reviewed code that
activates only once a maintainer both pushes and configures the setting the file merely declares. A
later reviewer must check that repository setting by hand rather than trust the workflow file's own
`environment:` line as proof it is protected.

## 3. The 76 case dataset honesty: Live PR and Full sweep draw on the SAME small set

D16's own sizing text names two different target sizes: 50 to 80 cases for a PR smoke lane (drawn
from a larger full set) and 300 to 500 for a full sweep (250 or more frozen test cases). The dataset
SP7 Task 6 actually built and this sub project's two golden set lanes actually run,
`testing/harness/dataset_tools/seed_cases.jsonl`, has 76 cases total, split across dev and test, as
documented in `docs/measurements/sp7-datasets-metrics.md` and cited again, not re derived, in
`docs/measurements/sp9-variants-matrix.md` Section 3.

Both `live-pr.yml` and `full-sweep.yml` run over this identical 76 case set today. The two lanes
differ in judge tier (`live-pr.yml`'s cheap tier, OpenAI nano or Anthropic Haiku, against
`full-sweep.yml`'s calibrated frontier tier, `gpt-5.6-sol` or `claude-opus-4-8`) and in whether the
result gates (Live PR gates on two deterministic floors; Full sweep never gates), never in the size
of the data each one draws from. Both workflow headers say this plainly, and both SP10 Task 2 and
Task 3's own reports name it as a carried disclosure rather than a surprise a reader should discover
on their own. A larger golden set actually reaching D16's 50 to 80 or 300 to 500 targets is SP7
dataset growth work, explicitly out of SP10's own scope (the plan's own Deferred section), and it has
no future owner today (Section 7).

## 4. Why no judged numeric gate exists on any lane yet: the noise floor is unmeasured

D18's own binding text: "regression gates only tighten after a measured noise floor (5 to 10 repeat
runs of one pinned config)." That repeat run has not happened anywhere in this repository; it remains
an open SP9 backlog item, named, not built, across every SP10 task's own carries section (Tasks 2, 3,
4, and 5 all name it identically). Until 5 to 10 identical config runs of the Live PR lane's cheap
judge tier, the Full sweep lane's frontier judge tier, or the Simulator lane's pass to the k power
statistic actually exist, any delta threshold on any of them would be a guess dressed up as a gate,
exactly research 15's own words for the identical trap: "Noise floor is unmeasured today. Until 5 to
10 identical config runs exist, any delta threshold is a guess."

Concretely, this is why every live lane SP10 built reports judged deltas and confidence intervals as
an uploaded artifact and gates on nothing but a deterministic floor already proven elsewhere: contract
diff (git aware, `task contracts:diff`), guard interception (`atlas.domain.guard`'s own fail closed
functions, zero tolerance), SP7's registry anchored answer correctness rate (a disclosed, reasoned
bar, `ANSWER_CORRECTNESS_FLOOR = 0.5`, not a calibrated one, per SP10 Task 2's own decision 2), and the
sentinel probe's own binary GO or NO GO reading (deterministic by construction: every red probe class
fails the gate, never averaged against the other two passing). None of these five floors is a judged
verdict; the judge tiers and pass to the k power statistic sit alongside them, measured and reported,
never gating, until the noise floor measurement this document names as still missing actually happens.
That measurement is a named follow on, not a number SP10 invents today.

## 5. D18's weekly cadence cut, held explicitly against research 15's own recommendation

`research/15-ci-lanes.md`'s own worked example recommends a weekly scheduled full sweep in addition to
merge to main triggering, as a backstop against silent provider drift: "Run the full frontier judged
eval on merge to main plus weekly scheduled... Weekly scheduled run guards against silent provider
drift." D18's own binding text is stricter and this sub project holds that stricter line rather than
silently adopting the more permissive research recommendation: "standing cadences are otherwise cut...
the one earned cron is the weekly janitor." `full-sweep.yml` therefore triggers on `push` to `main`
only, with no `schedule:` block at all, and its own header names this exact conflict and the
affirmative choice to resolve it in D18's favor rather than pick a cadence silently.
`testing/tests/test_full_sweep_workflow.py`'s own `test_workflow_is_not_also_a_scheduled_cron` and
`test_workflow_header_names_the_d18_weekly_cron_declination` pin this as a hermetic fact, not prose
alone: a future edit that adds a `schedule:` trigger to `full-sweep.yml`, or removes the header's own
disclosure, fails the hermetic suite. The weekly `janitor.yml` (SP5) remains the single earned standing
cadence in this entire repository; nothing SP10 built adds a second one.

## 6. `judge-live.yml` and `staleness.yml`: a retained judge calibration utility pair, not a sixth lane

Neither `judge-live.yml` (SP8, `workflow_dispatch` only, runs `testing/harness/judge/
live_provisional.py`'s registry truth agreement and judge versus judge kappa probe, both explicitly
provisional, neither licensing a deployment decision) nor `staleness.yml` (SP8, weekly cron plus
`workflow_dispatch`, checks the newest dated calibration artifact under
`testing/harness/judge/artifacts/live_provisional/` is under 45 days old) is one of the five HLD
section 7.3 lanes this document's Section 1 table names. They predate SP10 entirely, and SP10 did not
touch, extend, or rewire either file. This document classifies them explicitly, the same discipline
the SP10 digest already recommended (Section 3h): a matched pair supporting the judge calibration
lifecycle specifically, a manual recalibration probe and its own freshness watchdog, not a Live PR or
Full sweep substitute. `staleness.yml` is documented, by design, to read red honestly from merge day
until a real calibration artifact is committed (`docs/measurements/sp8-judge-human-loop.md` Section
8); that red reading is an intentional, disclosed state, not a regression this document's own review
found or needed to fix.

Naming this pair here, rather than leaving their relationship to the five lanes unstated, closes the
one remaining classification gap the digest flagged: a future reviewer counting workflow files against
the HLD's own five row table should land on exactly five named lanes plus this one retained utility
pair, `ci.yml`, `live-pr.yml`, `full-sweep.yml`, `simulator.yml`, `burst-benchmark.yml`, and
`judge-live.yml`/`staleness.yml` alongside `janitor.yml`/`release.yml`/`codeql.yml` as pre existing,
already reviewed infrastructure, never a sixth or seventh lane this sub project quietly added.

## 7. Carries to SP11 and SP12: cancelled, accepted debt with no future owner

SP11 (portfolio surface) and SP12 (record and replay return) were cancelled by user directive on
2026-07-21, recorded in the ledger (`.superpowers/sdd/progress.md`) directly: "DO SP9 + SP10 THEN
STOP. SP11 + SP12 CANCELLED." SP10 is therefore the last sub project this repository's own dual plane
rewrite effort will complete. Every item this sub project's own planning digest and task reports
named as an SP11 or SP12 carry is stated here plainly as accepted debt: real, disclosed, and now
without a sub project left to close it, rather than quietly dropped or left implying a future task
will pick it up.

**Carried from SP9's own final review, originally assigned to SP11: both already paid at SP9's own
wrap up commit (45a6804), before this document was even first drafted, named here only to close the
loop rather than as live debt:**

- The `ARCHITECTURE.md` `load/` line's earlier overstatement of "no hermetic gate surface" was
  corrected at 45a6804: `testing/harness/ARCHITECTURE.md:178` now reads "hermetically gated by 46
  tests", naming the load package's three helper modules (`thresholds.py`, `prompt_corpus.py`,
  `phoenix_join.py`) and their real hermetic coverage. No debt remains here.
- `docs/measurements/sp9-variants-matrix.md`'s own framing around the variant comparison stage
  (`matrix.variants.run_variant_comparison`) was also corrected at 45a6804: the document now names
  the variant stage explicitly as pending item 5 (`docs/measurements/sp9-variants-matrix.md:50`),
  naive vs agentic vs graph, measured over the SAME cases every other stage uses. No debt remains
  here either.

**Carried from SP10's own five task reports, originally intended for a future sub project's own
portfolio or live capture work, now accepted debt with no future owner:**

- The noise floor measurement (5 to 10 repeat runs) that would license any judged numeric gate on the
  Live PR lane's cheap tier, the Full sweep lane's frontier tier, or the Simulator lane's pass to the
  k power statistic (Section 4): an open SP9 backlog item at the time SP10 started, still open at the
  time SP10 ends. No sub project remains to perform it; a future maintainer inherits the measurement
  itself, not a scheduled task to run it.
- A larger golden set reaching D16's 50 to 80 or 300 to 500 case targets (Section 3): SP7 dataset
  growth work, never SP10's, never assigned to any remaining sub project either.
- Three pre existing burst infrastructure gaps SP10 Task 5 named but did not close, all disclosed
  directly in `burst-benchmark.yml`'s own header: `task burst:up`'s own pre existing indexes restore
  stop (burst's retrieval index has no restore from R2 path onto a node or PVC yet, SP5's own
  documented gap); burst's own Postgres credentials Secret never rendering today
  (`infra/environments/burst/values.yaml` never sets `.Values.postgres`); and turning a real, live
  Phoenix span export into `phoenix_join.SpanRecord`'s own JSON shape, documented, not yet built, live
  wiring (SP9 Task 6's own carry). A real first run of `burst-benchmark.yml`, even with every secret
  correctly configured, would fail at the first of these three gaps today. All three are SP5 or SP9
  owned infrastructure, not a CI lane wiring question, and none has a sub project left to close it.
- SP9's own live matrix sweep finding (the BGE reranker v2 m3 cross encoder saturating a CPU backed
  TEI node's own 8 cores hard enough to make `/info` itself unreachable, `docs/measurements/
  sp9-variants-matrix.md` Section 2's own "measured CPU reranker saturation finding") named a real
  operational limit that a GPU backed reranker endpoint or a hard candidate pool size cap would need
  to address in production. This remains an unaddressed finding; no sub project remains to act on it.

**Carried from SP9's own boundary decisions, originally SP12's to close, now accepted debt:**

- SP9 Task 3 built the matrix's `EmbeddingClient` port with NO REPLAY mode by design, a deliberate,
  disclosed boundary at the time (embeddings are cheap and fingerprint cached, so a full REPLAY seam
  for that port was named as SP12's own scope to finish, not something SP9 or SP10 needed to build).
  SP12 is cancelled; this port's own REPLAY gap has no sub project left to close it, and stays exactly
  as narrow as SP9 left it.
- The SP10 digest's own stated boundary: SP10's five lanes use the model gateway's existing
  RECORD/REPLAY/LIVE modes and the matrix's own content hash cache exactly as SP9 built them; SP10
  deliberately did not extend record and replay to any new seam, since that extension was named as
  SP12's own territory. With SP12 cancelled, no sub project remains to perform that extension either;
  every live lane this document's Section 1 table names stays bound by the record/replay mechanics
  SP9 already shipped, permanently, not provisionally.

Every item above is named here once, plainly, as this sub project's own honest closing state, not
because SP10 owed a fix for any of it (none of it was SP10's task list), but because a document
claiming to be the honest measurement record for the last sub project in this effort must say clearly
that these items now have no assigned future owner, rather than let a reader assume cancellation
quietly erased them.

## 8. Contract versioning: unchanged by this task

This task touches no `contracts/*/schema.json` file and adds no new contract; `git diff --stat`
against this commit shows zero contract changes (the same check every SP10 task before it already ran
and recorded clean). No CHANGELOG entry is required for this task, the same "docs only, no contract
touched" reasoning `docs/measurements/sp7-datasets-metrics.md` Section 7, `docs/measurements/
sp8-judge-human-loop.md` Section 9, and `docs/measurements/sp9-variants-matrix.md` Section 8 all
already applied to their own respective final, docs only tasks.
