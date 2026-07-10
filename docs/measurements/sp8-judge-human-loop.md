# Measurement: SP8 judge and human loop, provisional calibration and the recompute path

Committed factual record for SP8 Task 8, the last task of the sub project. Follows
`docs/measurements/sp3-rag-spine.md` and `docs/measurements/sp7-datasets-metrics.md`'s own pattern:
every number below is cross referenced to the specific test or command that reproduces it, MECHANICS
are gated by the hermetic suite, PROBABILISTIC outcomes are measured and recorded, never silently
gated by a test that would either be flaky or launder a real finding behind a green check mark.

## LIVE CAPTURE DEFERRED, three lanes, none run by this docs only task

Three lanes in SP8 need a live call and are deferred here, exactly like SP7's own retrieval half:

1. **The judge live provisional sweep** (`task judge-live`, `testing/harness/judge/live_provisional.py`).
   Needs `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`. Both ARE present in this repository's `.env`
   (confirmed by presence check only, never by reading either value, the same convention
   `docs/measurements/sp7-datasets-metrics.md` Section 5 already used). This lane needs no Postgres,
   no TEI, and no fastlane node at all: `judge.provisional.manufactured_cases` cites the registry
   directly, so nothing about its own infrastructure blocks it. It is deferred here anyway, because
   Task 8 is a docs only task (per `.superpowers/sdd/sp8-task-7-report.md`'s own closing line, "Task 8
   ... is a docs only task with no further code lane to build") and running it would spend real
   provider money to produce a live measurement outside this task's own scope, not because of a
   missing dependency. See Section 2 for the exact recompute command.
2. **The metamorphic live lane** (`testing/tests/test_metamorphic_live.py`, SP8 Task 6). Needs
   `docker compose up postgres tei-embed tei-rerank`. The fastlane node (the native amd64 TEI
   endpoint) was deleted after SP7 (`.superpowers/sdd/sp8-hitl-report.md`'s own account of the batch
   label generation lane names the same deletion); the compose stack is the documented, keyless but
   slower, retrieval and embedding path. This environment currently has no Atlas compose services
   running (`docker compose ps` returns an empty service list at authoring time). See Section 4.
3. **The corpus mutation live lane** (`testing/harness/corpus_mutation/__main__.py` and
   `testing/tests/test_corpus_mutation_live.py`, SP8 Task 7). Needs the same compose stack plus a
   provider key for the real agent turn. Same reason, same absent stack. See Section 5.

A fourth thing is pending for a different reason, not a live infrastructure gap: the real, roughly
200 item human label set (Section 3.2, Section 7). The HITL adjudication page and its storage exist
and are hermetically tested; zero real labels have been collected in this repository as of this
writing (`var/labels/` is gitignored and absent), and the seed set that feeds the label generator
currently has 76 cases, not yet the target of roughly 200.

Everything else in this document is real today: hermetic facts already gated by `task test`, or a
number recomputed directly from a committed test fixture, needing no live call at all.

## 1. The judge, the rubric, and the trace boundary: hermetic facts (gated, `task test`)

`testing/harness/judge/rubric.py`'s `RUBRIC_GROUNDEDNESS` is the one binary rubric SP8 ships: every
factual claim in an answer must be entailed by the cited retrieved context or the registry's own
entity facts; an abstention passes; an unsupported claim fails. `judge.llm_judge.judge_label` runs
one answer past the judge through the record and replay gateway (REPLAY in the hermetic lane, a
missing cassette hard fails, never reaches the network) and returns a binary label; `translate_verdict`
crosses that label into the frozen trace contract's own wire vocabulary, `grounded` or `ungrounded`
(`contracts/trace/schema.json`'s pinned enum), never the judge's own PASS or FAIL prompt vocabulary.
`judge.contract.JudgeContract.fingerprint()` is the judge's versioned identity (model id, rubric
version, prompt template hash, one canonical digest).

The trace contract bumped MINOR, 1.1.0 to 1.2.0 (SP8 Task 1, commit `fcb1f8d`): `atlas.judge.id`,
`atlas.judge.rubric_version`, `atlas.judge.verdict`, and `atlas.subject.pseudonym` (an HMAC of
`customer_id`, never the raw id on the wire) all carry real emitters now, four attributes reserved
since v0.1.0 and narrowed at the v1.0.0 freeze until this task. `contract_tools.freeze_check` reads
clean: all 30 reserved attributes are emitted or narrowed, gated by
`testing/tests/test_freeze_check.py::test_the_real_committed_freeze_is_clean`. Fingerprint stability,
verdict translation both ways, the fail closed parse on an unparseable reply, and a seeded REPLAY
judge run producing one grounded and one separate ungrounded verdict are all pinned by
`testing/tests/test_judge_contract.py`, `test_judge_rubric.py`, `test_judge_llm_judge.py`, and
`test_judge_trace_integration.py`.

## 2. Provisional calibration: the manufactured set and its two labeled numbers

### 2.1 The manufactured set (n=4)

`judge.provisional.manufactured_cases()` walks the committed registry's two typed contradictions
(`corpus/registry/core.yaml`) and yields exactly one true and one false case per contradiction:

- `conflict-daniel-contract`: winning fact `contract_term-daniel-2025:contract_months` (value 12),
  losing fact `plan-fiber-100:contract_months`.
- `conflict-promo-price-north`: winning fact `region-north:equipment_rental_override_amount`, losing
  fact `promotion-fiber500-launch-north:equipment_rental_waived`.

Two contradictions times one true and one false case each is **n=4** manufactured cases total,
pinned by `testing/tests/test_judge_provisional.py::test_manufactured_cases_cover_every_registry_contradiction_exactly_twice`.
This is intentionally tiny (SP8 Task 3's own report names it a concern), and any agreement number
computed from it, real or fixture, is a correspondingly noisy statistic; Section 7 states this
plainly again so it is not lost among the numbers.

### 2.2 Registry truth agreement -- source: manufactured ground truth by construction

Source label (`RegistryTruthAgreement.SOURCE`): `registry_truth_manufactured_ground_truth_by_construction`.

**REAL, against an actual judge model: PENDING LIVE CAPTURE.** Recompute command (`task judge-live`,
needs `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`, deferred per the callout above):

```
task judge-live
```

Expands to `PYTHONPATH=backend:testing/harness:. uv run --all-groups --env-file .env python
testing/harness/judge/live_provisional.py`, which sweeps each provider's tiers cheapest first,
features the first tier that ran without error per provider, and overwrites
`testing/harness/judge/artifacts/live_provisional/latest.md` plus a dated snapshot with the real
reading, source labeled, the honesty statement attached (Section 2.4). This has NOT been run for
this document; no artifact under `live_provisional/` carries today's date.

**FIXTURE DERIVED (hermetic, gated by `task test` today): not a real distribution claim, only the
arithmetic.** `testing/tests/test_judge_provisional.py` computes `registry_truth_agreement` against
the real committed registry's own manufactured cases, using the cases' own construction guaranteed
ground truth AS the judge's labels (a stand in for "a judge that gets every case right" or "every
case wrong"), never a real model's output:

| reading | agreement | test |
|---|---|---|
| a judge that matches every manufactured case's ground truth | 1.0 | `test_registry_truth_agreement_is_one_when_the_judge_gets_every_case_right` |
| a judge that inverts every manufactured case's ground truth | 0.0 | `test_registry_truth_agreement_is_zero_when_the_judge_gets_every_case_wrong` |

### 2.3 Judge vs judge kappa -- source: two judges, no ground truth on either side

Source label (`JudgeVsJudgeAgreement.SOURCE`): `judge_vs_judge_no_ground_truth`.

**REAL, between two live judge contracts: PENDING LIVE CAPTURE**, the same `task judge-live` command
above; it needs BOTH `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` present (both are, per the callout),
since `judge.provisional.judge_vs_judge_kappa` compares a cross family (OpenAI) and a same family
(Anthropic) tier on the same manufactured set. If either provider's every tier fails, `live_provisional.py`'s
own `_run()` states plainly that both keys are needed and computes no kappa at all, rather than
guessing (see Section 6.1 for the one concrete way a same family tier can fail this way).

**FIXTURE DERIVED (hermetic, gated by `task test` today): not a real distribution claim, only the
arithmetic.** `testing/tests/test_judge_provisional.py`:

| reading | kappa | test |
|---|---|---|
| two judges given identical labels on the manufactured set | 1.0 | `test_judge_vs_judge_kappa_is_perfect_when_both_judges_agree_on_everything` |
| two judges given fully inverted labels on the manufactured set | negative (kappa < 0, the exact value is not pinned, only its sign) | `test_judge_vs_judge_kappa_reads_negative_when_the_two_judges_disagree_on_everything` |

### 2.4 Honesty statement (KAPPA HONESTY, binding)

Neither number above, whether PENDING LIVE CAPTURE or fixture derived, licenses a production
deployment of this judge. `judge.provisional.ProvisionalCalibrationArtifact.render()` closes with
this exact statement, reproduced here because the plan's own KAPPA HONESTY rule requires the source
label to travel with the number everywhere it is restated, never just once in the source module:

> This provisional artifact does not license a production deployment of this judge: neither the
> registry truth agreement above nor the judge vs judge kappa above may ever stand in for it. The
> production deployment gate is Cohen's kappa >= 0.6, its confidence interval's lower bound, against
> REAL human gold labels only (`judge.calibration.CalibrationReport.licensed`, D15). These two numbers
> are provisional signals collected before any human labeled set exists; read them as a sanity check
> on the judge's wiring, never as a calibration.

This repository has its own documented prior failure behind this rule, named directly in
`testing/tests/test_judge_provisional_honesty.py`'s own docstring: a kappa reported as 0.29 in prose
did not match the 0.21 actually sitting in the committed artifact, one number quietly standing in for
another. `judge.provisional`'s own source code never imports `AUTOMATION_BAR`, `gate_on_lower_bound`,
`GateVerdict`, or `GateDecision`, and neither `RegistryTruthAgreement` nor `JudgeVsJudgeAgreement` nor
`ProvisionalCalibrationArtifact` exposes a `licensed` property; `test_judge_provisional_honesty.py`
proves this by an AST walk over the module's own source, not by trusting a docstring, plus a spy on
`gate_on_lower_bound` proving it is never called even against a "perfect" provisional artifact where
both numbers would clear 0.6 if anyone (wrongly) compared them.

### 2.5 The deployment gate itself, illustrated only, never a real license

`judge.calibration.CalibrationReport.licensed` reads `AUTOMATION_BAR = 0.6` through
`quality.gate.gate_on_lower_bound`, the SAME rule a release gate uses, never a hand rolled
`kappa_ci[1] >= bar` comparison. Its own committed hermetic fixtures (`testing/tests/test_judge_calibration.py`,
every one hand typed, never real human gold, per the file's own docstring) illustrate the arithmetic
path a real human calibration would run through, once real labels exist:

| reading | n | kappa | gate verdict | test |
|---|---|---|---|---|
| a near perfect hand typed set | 50 | 0.80 | PASS (licensed) | `test_a_licensing_report_clears_the_bar_on_the_lower_bound_not_just_the_point` |
| a near chance hand typed set | 100 | 0.20 | FAIL (not licensed) | `test_a_rejected_report_misses_the_bar_on_the_lower_bound` |
| a tiny hand typed set with one real disagreement | 4 | (interval too wide to call) | QUARANTINE (not licensed either) | `test_a_quarantined_interval_is_not_licensed_either` |

These three rows are the arithmetic only, exercised against invented ids and invented labels; they
are not a reading of this judge, provisional or otherwise, and are cited here only to show the same
gate Section 3.2's recompute calls once real human labels land.

## 3. The one command recompute

### 3.1 Provisional recompute: `task judge-live`

Already documented in Section 2.2 and 2.3. Cadence: monthly, or on any rubric or judge model change
(`docs/runbooks/judge-recalibration.md` Step 1 and Step 3). This overwrites
`testing/harness/judge/artifacts/live_provisional/latest.md` and a dated snapshot; there is no bar to
clear here, only a provisional reading to record and commit.

### 3.2 Human calibrated recompute: the real deployment gate

**No dedicated Taskfile target exists for this today.** `task judge:calibrate` (the plan's own named
example) does not exist; `Taskfile.yml` today defines `judge-live` (Section 3.1), `label:generate`,
and `label:generate-live` (the batch answer generator, SP8 Task 4), but nothing wraps
`judge.calibration.calibrate()` end to end against a real label set. This is stated here plainly
rather than implied, per the plan's own "discover or confirm it" instruction and this repository's
KAPPA HONESTY posture of never asserting a capability that is not actually wired.

What exists and is already hermetically tested, composable into the real recompute the day labels
land (`docs/runbooks/judge-recalibration.md` Step 2 names the same composition in prose):

1. `backend.atlas.adapters.label_store.LabelStore(path, clock).read_all()` returns every collected
   `LabelRecord` (`trace_id`, `role`, `verdict` of `pass` or `fail`, `critique`, `created_at`);
   filter to `role == "adjudicator"`.
2. The label item set the HITL page served (`testing/harness/labeling/generate_label_set.py`'s
   output, `var/labels/label_items.real.jsonl` once generated via `task label:generate-live`) maps
   each `trace_id` to its own `question`, `answer`, and `retrieved_chunks`.
3. For each labeled `trace_id`, run `judge.llm_judge.judge_label(gateway, judge.rubric.RUBRIC_GROUNDEDNESS,
   question, answer, context)` (the context built from the item's own `retrieved_chunks` text) to get
   the judge's own current verdict for that same turn.
4. `judge.calibration.calibrate(contract, case_ids=trace_ids, human_labels=[...], judge_labels=[...],
   generated_at=<a real clock>)` builds the `CalibrationReport`; `report.render()` prints `n`, raw
   agreement, Cohen's kappa with its 95 percent interval, AC1, prevalence, and the gate's own verdict.

`judge.calibration.calibrate()` takes `human_labels` as a plain argument; it has no opinion on where
they came from (its own module docstring says so directly), so composing the four steps above into
one script is the entire remaining work, not a design question. The day real labels replace the
fixture rows in Section 2.5's table, this SAME arithmetic reads them instead, automatically: no
schema change, no code change to `calibrate()` itself, only real data flowing through an already
built and already tested path. Recorded here as a concrete concern for whoever next touches this
lane: add a `judge:calibrate` Taskfile target wrapping steps 1 through 4 above, so "the one command"
becomes literally one command rather than a documented composition; not built in this task, since
Task 8 is docs only.

## 4. Metamorphic lane: hermetic facts real today, live outcomes PENDING

Hermetic (SP8 Task 6, gated by `task test`): three frozen families seeded from
`conflict-daniel-contract`, `testing/harness/metamorphic/families.py`. `PARAPHRASE_FAMILY` (rank
overlap floor 0.5, five natural rewordings of "is my plan contract free"), `TYPO_NOISE_FAMILY` (floor
0.5, three character level typos), `QUERY_PERTURBATION_FAMILY` (floor 1.0, casing and whitespace
noise only, the strongest tier). All three families hold every deterministic, judge free invariant
(`id_based_retrieval_agreement`, `rank_overlap_floor_holds`, `registry_answer_equivalence_holds`) over
the real stub retriever, and four "has teeth" tests prove each invariant can genuinely fail (a member
missing the ground truth id, a scrambled ranking below its floor, a drifted answer failing
equivalence, a corpus with the winning chunk silently dropped), all in
`testing/tests/test_metamorphic.py`. The 0.5 floors were measured against the real stub retriever
during development, not guessed and then encoded (`.superpowers/sdd/sp8-task-6-report.md`'s own
account).

**Live lane: PENDING LIVE CAPTURE.** `testing/tests/test_metamorphic_live.py`, written, import
checked, collecting cleanly (deselected by the default `not live` marker), never run. Two things it
would measure: MECHANICS gated (the fused, pre rerank candidate pool contains the Daniel chunk for
every paraphrase family member), and QUALITY measured only, never asserted (rank overlap at the real
deployed k with reranking on, expected in the neighborhood of `docs/measurements/sp3-rag-spine.md`
Section 1's own already recorded finding, that a generic cross encoder reranker may demote the
customer specific override below generic "No contract" marketing pages). Recompute command:

```
docker compose up postgres tei-embed tei-rerank -d
uv run pytest -q -m live testing/tests/test_metamorphic_live.py
```

## 5. Corpus mutation lane: hermetic facts real today, live outcomes PENDING

Hermetic (SP8 Task 7, gated by `task test`): `testing/harness/corpus_mutation/selection.py`
deterministically picks the SAME `conflict-daniel-contract` winning fact
(`contract_term-daniel-2025:contract_months`, 12) the metamorphic lane and the manufactured set both
already seed from, and mutates it to 24 (`old_value + 12`, guaranteed different, zero invented
corruption logic), narrowing to exactly one affected document over the real registry. `scope.py`
derives a deterministic, content addressed `corpus-mutation-<hash>` name that can never collide with
a committed `corpus-X.Y.Z` version, and guarantees ephemeral directory cleanup on every exit path.
`tracking.py`'s `answer_tracks_mutated_truth` distinguishes an answer that tracks the new truth from
one that repeats the stale, pre mutation value (parametric or cached knowledge) from one that is
simply wrong, over SP7's own `quality.agent_metrics.is_fact_grounded`, never a second home rolled
grounding check.

**Live lane: PENDING LIVE CAPTURE.** The five step operator pipeline (mutate the real registry, re
render only the affected document under the ephemeral corpus_version, re index it via a real TEI
embed and a real Postgres load, drive the real Atlas graph for one turn as the customer the fact
belongs to, grade the real answer) has NOT been run: it needs the compose stack (Section 4's same
absent dependency) and a provider key for the real generation call. Recompute command:

```
docker compose up postgres tei-embed tei-rerank -d
PYTHONPATH=backend:testing/harness:. uv run python -m corpus_mutation
```

`testing/tests/test_corpus_mutation_live.py` covers the mechanics half only (a real retrieval query
returns the new mutated value and never the stale one, and the ephemeral directories are truly gone
once the run exits), also not run, collecting cleanly and deselected by the default marker.

## 6. Two accumulated carries, folded in here as measurements

### 6.1 The judge_label live content flattening gap: a named caveat for a live run

`judge.llm_judge.judge_label` reads `getattr(reply, "content", "") or ""` and hands it straight to
`_parse_label`, which calls `text.strip()`. Every seeded cassette in the hermetic suite returns a
plain string `content`, so this path is exercised only with strings today
(`.superpowers/sdd/sp8-task-3-report.md`'s own Concern 1 names this gap first). A real live reply from
an Anthropic model can return `content` as a LIST of content blocks rather than a bare string, the
exact shape the retired `evals/judge/live_calibration.py`'s own `_content_text` helper existed to
flatten. `judge_label` has no such flattening step, so a list shaped `content` makes `text.strip()`
raise `AttributeError`, not a silent bad parse.

This matters specifically for the same family (Anthropic) tier of Section 2.3's judge vs judge kappa.
`judge.live_provisional._sweep` wraps every tier's `judge_label` calls in a `try` and `except
Exception`; a raised `AttributeError` is caught, that tier is recorded as `FAILED: AttributeError:
...`, and the sweep moves to the next Anthropic tier rather than propagating a garbled label. If every
Anthropic tier fails this way, `_run()` states in the printed artifact that both `OPENAI_API_KEY` and
`ANTHROPIC_API_KEY` are needed and computes no judge vs judge kappa at all. The same family tier of
`task judge-live` therefore fails SAFE, either the next cheapest tier's plain string reply succeeds,
or the run reports no kappa, never a kappa quietly computed from a caught exception's fallback value.
This gap is disclosed, not fixed, here or in Task 3: fixing it means editing Task 1's already
committed `llm_judge.py`, and the fix cannot be exercised hermetically either way (every seeded
cassette uses a plain string), so it is carried forward as a known caveat for a live run rather than
silently repeated in a future report.

### 6.2 The registry versus catalog id space distinction, stated explicitly again

Two genuinely separate identity spaces exist in this repository, and neither ever shares an id with
the other:

- **The RAG knowledge registry** (`corpus/registry/core.yaml`), reached through the
  `search_knowledge` tool: entity ids like `plan-fiber-100` and `contract_term-daniel-2025`, the same
  two ids Section 2.1's manufactured contradiction cites.
- **The demo account and catalog seed** (`backend/atlas/domain/catalog.py`'s `CATALOG`), reached
  through the account tools: plan ids like `plan_legacy_value`.

`docs/measurements/sp7-datasets-metrics.md` Section 6 already names this distinction as an
informational carry from SP7's own T5/T6 reviews, which caught and fixed exactly this confusion in
two dataset contract examples (`gc-0001`/`gc-0002`, a case authoring a `catalog.get_plan` argument
with a registry id instead of a real catalog id). Restated here because SP8's own judge, promotion,
and label content all name plans and fees in prose (`judge.rubric.RUBRIC_GROUNDEDNESS`'s prompt text,
`contracts/dataset/taxonomy.yaml`'s `hallucinated_entity` code, `judge.promotion`'s promoted cases):
any future content authored against either surface must use the id space that surface actually reads,
never the other one, exactly the defect this repository has already made and fixed once.

## 7. Small n honesty, stated plainly

**The manufactured set is n=4** (Section 2.1: two registry contradictions, one true and one false
case each). Any registry truth agreement or judge vs judge kappa computed from it, real or fixture,
is a correspondingly noisy statistic; Section 2's own artifact and this document both say so on
purpose, matching SP8 Task 3's own report ("a reader skimming only a rendered percentage should not
mistake a small n for statistical weight").

**76, not 200.** The seed set that feeds the label generator (`testing/harness/dataset_tools/seed_cases.jsonl`)
has 76 cases today, short of the roughly 200 item session the plan's own Goal and RESEQUENCING text
size the HITL adjudication page for (D30 itself names a smaller 30 item pilot plus a 40 to 50 item
double labeled overlap subset within that larger session, not the roughly 200 total on its own). Zero
real human labels exist in this repository as of this writing (`var/labels/` is gitignored and absent
from this working tree). A full labeling pass over today's 76 cases would produce at most 76 real
labels, not the roughly 200 the SP11 portfolio surface should expect, unless SP7's own seed set growth
lands first. This is not a defect in the HITL machinery: the label store, the adjudication page, and
the batch answer generator are all built and hermetically tested (`.superpowers/sdd/sp8-hitl-report.md`),
and `task label:generate-live` makes no assumption about the seed set's own size, processing however
many cases exist at the moment it runs. It is a real, honestly stated gap between the dataset's
current size and the label target, carried forward explicitly to SP11 in Section 8.

## 8. Carries forward

**Immediate, from merge day:** the weekly staleness workflow (`.github/workflows/staleness.yml`,
`testing/harness/evals/staleness.py`) reads red every Monday until the first `task judge-live`
artifact is committed under `testing/harness/judge/artifacts/live_provisional/`, since none exists
yet (Section 2.2's callout). This is intentional, recorded honesty, not a broken check: an absent
calibration IS stale, and the workflow is notification only, never a gate on `task test`; it
resolves the first time Section 3.1's recompute lands and is committed.

**SP9 (variants and matrix):**

- **The jury or panel caller.** `judge.panel.panel_vote` (ties fail closed to label 0, absorbed
  verbatim in SP8 Task 2) is a real, hermetically tested mechanism with no caller anywhere in this
  repository today. SP9's benchmark matrix runner is named, in `judge.panel`'s own docstring and in
  the SP8 planning digest, as the eventual caller in a headline benchmark context (D15's three model
  cross provider jury); nothing here builds that caller, only the mechanism it will invoke.
- **The D28 local generator spot check seam.** "Judge behavior on local generator outputs is human
  spot checked before the kappa gate is trusted on that distribution"
  (`.superpowers/sdd/sp8-planning-digest.md` Section 1). This needs SP9's local generation and GPU
  burst arm to exist before there is anything to spot check; SP8 names the seam only (this document
  is that naming) and builds no machinery for it.

**SP11 (portfolio surface):**

- **The honest kappa number.** Only `judge.calibration.CalibrationReport.licensed`, computed against
  REAL human labels through Section 3.2's recompute, may ever appear on a portfolio surface as this
  judge's calibration. Neither Section 2.2's registry truth agreement nor Section 2.3's judge vs judge
  kappa, real or fixture, may stand in for it in any prose, dashboard, or artifact (Section 2.4).
- **The 76, not 200, label reality.** SP11 should not assume a completed roughly 200 item human
  calibration exists by the time it finalizes portfolio numbers. As of this document, real human
  labels number exactly 0, and the seed set that would feed them numbers 76, not yet roughly 200
  (Section 7). Recompute per Section 3.2 once real labels land, before any portfolio number reads on
  this judge's calibration.

## 9. Contract versioning: unchanged by this task

The trace contract's MINOR bump (1.1.0 to 1.2.0) already landed in SP8 Task 1 (`fcb1f8d`), committed,
ADR-029 amended, `freeze_narrowed.yaml` corrected, freeze clean at 30 of 30 reserved attributes
(Section 1). This task (8) touches no `contracts/*/schema.json` file and adds no new contract; the
dataset family stays wherever SP7 last left it. No CHANGELOG entry is required for this task, the same
"docs only, no contract touched" reasoning `docs/measurements/sp7-datasets-metrics.md` Section 7
already applied to its own last task.
