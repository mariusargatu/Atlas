# Runbook: behavioural drift and cassette re-recording

**Trigger.** A scheduled re-record falls due, a provider deprecation notice arrives, or the judge
recalibration runbook sent you here.

**What exists, and what does the diffing.** There is no separate committed drift baseline. The
committed cassettes are the baseline: `task record` re-records them against the live provider, and the
hermetic suite replaying them (`task test`) is the diff. A behavioural change (a moved tool call, a
flipped guard, a different outcome) fails a decision-level assertion; a pure rewording does not,
because the graders key on trace spans, not prose. `task drift` is a self-contained demo: it seeds its
own cassettes over three hardcoded snapshots and prints the prose-vs-behavioural classification you
apply when you read those failures. The live shadow re-record that would let drift fire on its own
against the provider is deferred (see the drift package docstring), so the re-record cadence is the
control, not the code.

**Cadence.** Re-record on a fixed cadence and always before adopting a new model snapshot. The
scheduled `staleness` workflow nags when calibration artifacts age past 45 days; GitHub disables cron
on public repos after 60 days without commits, so treat the workflow as a reminder with a known
failure mode, not a guarantee.

## Steps

1. Confirm the current suite is green before touching anything: `task test`.
2. Re-record the committed cassettes against the live provider (needs keys; never the PR lane):
   `task record`.
3. Replay them to surface the diff: `task test`. A green run means no behavioural drift; a failure
   names the decision that moved. Run `task drift` if you want to rehearse the severity call on the
   demo snapshots first.
4. Read each failure by severity:
   - **Prose-only drift** (wording moved, decisions identical): acceptable. The suite stays green
     because graders key on spans, not prose. Commit the re-recorded cassettes.
   - **Behavioural drift** (a decision changed, a test fails): stop. Each changed decision is either a
     provider-side regression (file it: reduce to a golden case via the promotion runbook, then decide
     whether to pin back or adapt) or an intended improvement (update the expectation explicitly, in
     its own commit, with the reason in the message).
5. Never edit an expected output to match what the model now produces without writing down why. That
   is fixing the test instead of the bug.
6. Commit the re-recorded cassettes on their own, so the diff history stays readable.

**Tolerance policy.** One behavioural drift is an investigation. Repeat behavioural drift on the same
decision across two re-records means the invariant is wrong or the provider is unstable: either move
the behaviour behind deterministic logic (design the risk away) or widen the oracle deliberately and
say so in the commit.

**Danger line.** Changing canonicalization (`testing/harness/determinism/canonical.py`) invalidates
every committed cassette. That is a migration, not a re-record.
