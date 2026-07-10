# Measurement: SP7 datasets, metrics, and the seed set's retrieval numbers

Committed factual record for SP7 Task 7, the last task of the sub project. Follows
`docs/measurements/sp3-rag-spine.md`'s own pattern: every number below is cross referenced to the
specific test or command that reproduces it, MECHANICS are gated by the hermetic suite,
PROBABILISTIC outcomes are measured and recorded, never silently gated by a test that would either
be flaky or launder a real finding behind a green check mark.

## LIVE CAPTURE DEFERRED (fastlane node OOM)

The seed set's retrieval half (recall@k/MRR/nDCG) and the flagship baseline's reranked position
reproduction both need a live call against real TEI. The fastlane node (`atlas-fastlane`,
`.env.fastlane`) went unreachable mid measurement at authoring time: an unplanned reboot left the
TEI process OOM thrashing (the TCP socket accepts a connection but neither SSH nor an HTTP request
against `/info` ever completes). A reboot is pending an operator decision and is deliberately not
run as part of this task.

Sections 2, 4, and 5 below are marked **PENDING LIVE CAPTURE** rather than filled with an invented
number: this project's own doctrine (`docs/measurements/sp3-rag-spine.md`, `quality/gate.py`'s own
docstring) is that a probabilistic outcome is measured and named, never fabricated to fill a table
cell. The exact, copy pasteable rerun command an operator runs once the node is healthy again:

```
docker compose up -d postgres
set -a; source .env.fastlane; set +a
PYTHONPATH=backend:testing/harness:. uv run pytest -q -m live -s \
  testing/tests/test_sp7_retrieval_metrics_live.py
```

`testing/tests/test_sp7_retrieval_metrics_live.py` is marked `live`, excluded from `task test`
(`addopts = "-m 'not live'"`, `pyproject.toml`), written, import checked, and collected clean
(3 tests) against the current tree; it has simply not yet completed a run against a healthy TEI
node. Once it runs, the operator fills Sections 2/4/5 below with the printed figures and replaces
this callout, exactly the same operator rerunnable pattern SP6's trace freeze evidence established
(`contracts/trace/freeze_evidence.json`, committed data plus a documented capture command; the
hermetic freeze check reads the committed file and never reruns the capture itself).

Everything else in this document (Sections 1, 3, 6, 7) is real today: hermetic facts already gated
by `task test`, or a power sizing analysis computable from `n` alone with a disclosed, deliberately
pessimistic variance bound, needing no live call at all.

## 1. The seed set, hermetic facts (gated, `task test`)

76 hand curated cases (`test_seed_dataset.py`, D16 sizing 50-150), covering every case class Task 1's
mechanical generator can derive plus three hand authored classes it cannot (action/write cases,
multi turn trajectories, the D33 fairness persona cohort):

| slice | count | dev | test |
|---|---|---|---|
| `factoid_one_hop` | 35 | 28 | 7 |
| `factoid_two_hop` | 10 | 8 | 2 |
| `grounded_not_true` | 10 | 8 | 2 |
| `hallucination_bait` | 10 | 8 | 2 |
| `other` (action, multi turn, persona overflow) | 11 | 8 | 3 |
| **total** | **76** | **60** | **16** |

Reproduced by `manifest.build_manifest(seed_cases)` (`testing/harness/dataset_tools/manifest.py`),
pinned by `test_manifest_reports_every_known_slice_and_every_case_is_placed` and
`test_both_splits_carry_adversarial_coverage_the_t4_hard_requirement`
(`testing/tests/test_seed_dataset.py`). `fact_overlap.count == 4`
(`contract_term-daniel-2025:contract_months`, `fee-equipment-rental:amount`,
`plan-fiber-500:name`, `region-north:name`): the small registry (21 entities, 2 contradictions)
forces some dev/test fact coverage overlap, declared in the manifest, never gated, never hidden
(`test_manifest_declares_fact_overlap_honestly_small_registry_visible`). `contamination_lint`
passes over all 76 cases (`test_manifest_build_over_the_seed_set_passes_the_contamination_lint`).

**The retrieval relevant slice for this document's retrieval measurement is 55 cases**
(`factoid_one_hop` + `factoid_two_hop` + `grounded_not_true`, every case whose `expected_doc_ids`
is nonempty): `hallucination_bait` is excluded by construction (`answerable: false`, nothing to
retrieve for an unanswerable case), and `other` carries no case level `expected_doc_ids` (action
and multi turn cases are graded by `quality.agent_metrics`, not `quality.ir_metrics`). This matches
the plan's own wording exactly ("the retrieval half of the seed set's factoid + adversarial
cases") and is pinned as a mechanics assertion in the live test itself
(`test_seed_set_retrieval_half_recall_mrr_ndcg`: `assert len(cases) == 55`), computed hermetically
here by walking the committed `seed_cases.jsonl` with `dataset_tools.manifest.case_slice`, no
network call needed to count case classes.

`expected_doc_ids` is named for schema continuity with the dataset contract's original field name,
but every value in it is a chunk id (`rag_tools.chunker.ChunkRecord.chunk_id`), not a raw `doc_id`
string: `dataset_tools.generator`'s own docstring states this directly ("`expected_doc_ids` holds
retrieval unit ids... not raw `doc_id` strings: retrieval happens at chunk granularity"). Retrieval
grading below is therefore always chunk level ID membership, exactly what `quality.ir_metrics`
computes.

**The deterministic metric harness itself is real and hermetically tested today**, independent of
the live capture: `testing/harness/quality/retrieval_report.py` (`evaluate`, `CaseRetrieval`,
`RetrievalReport`) aggregates a sequence of already computed `(retrieved, relevant)` pairs into
hit_rate@k (Wilson interval), recall@k/MRR/nDCG@k (percentile bootstrap), and
`detectable_effect_ndcg`, using `quality.ir_metrics` and `quality.stats` unmodified, no reinvented
formula. `testing/tests/test_retrieval_report.py` (14 tests) proves this over a hand computed
fixture (four cases, k=3, every per case recall/nDCG/reciprocal rank value hand worked and cross
checked against `quality.ir_metrics` directly), written red first (confirmed by temporarily
removing the implementation and observing `ModuleNotFoundError` before restoring it) then green.
This is the module the live test imports; only the DATA (real retrieved chunk ids from a live
`PgvectorRetriever`) is pending, never the arithmetic that turns that data into a report.

## 2. Retrieval half: recall@3, MRR, nDCG@3 -- PENDING LIVE CAPTURE

Config once run: `DEPLOYED_K=3` (`atlas.mcp_servers.knowledge_server.DEPLOYED_K`),
`RetrievalConfig()` defaults (`k_fused=50`, `k_final=5`, `rerank_enabled=True`) -- the exact call
shape `knowledge_server.search_knowledge` makes in production, not a special evaluation only
configuration. `n=55` (Section 1), seed `20260720`
(`test_sp7_retrieval_metrics_live.METRICS_SEED`, the seed for this report's own bootstrap
resampling, `n_resamples=2000`, so the recorded CI reproduces byte for byte on rerun against the
same retrieved data).

| metric | point | 95% CI | interval kind |
|---|---|---|---|
| hit_rate@3 | PENDING LIVE CAPTURE | PENDING LIVE CAPTURE | Wilson (`stats.wilson_interval`) |
| recall@3 | PENDING LIVE CAPTURE | PENDING LIVE CAPTURE | percentile bootstrap (`stats.bootstrap_ci`) |
| MRR | PENDING LIVE CAPTURE | PENDING LIVE CAPTURE | percentile bootstrap |
| nDCG@3 | PENDING LIVE CAPTURE | PENDING LIVE CAPTURE | percentile bootstrap |

MECHANICS, gated by the live test itself once it runs: the full 55 case retrieval relevant slice
must be used (no silent drop), and every one of the 55 cases must return at least one chunk (no
dead retrieval path). The four numbers above are QUALITY, to be measured and recorded, never gated
on a specific bar: this is a 45 chunk corpus behind a small, hand curated seed set, not a benchmark
scale retrieval quality claim (see Section 3 for exactly how much power this n has).

## 3. Honest interval width: what this n can actually detect (hermetic, real today)

`n=55` is the seed set's real, honest retrieval relevant sample size, not a rounded up or padded
number (D16 sizing was explicit that the registry's own coverage, not a target count, decides seed
set size); `n=76` is the full seed set. Neither needs a live call to size: `stats.required_n` and
`stats.detectable_effect` are pure functions of `n` and a per case score standard deviation, and a
defensible upper bound on that standard deviation needs no observed data at all, only the fact that
nDCG is bounded in `[0, 1]`. A value bounded in `[0, 1]` has maximum population variance `0.25`
(`p(1-p)` maximized at `p=0.5`, an even split between the two extremes), i.e. standard deviation
`0.5`: a deliberately pessimistic ceiling, since a genuinely mixed one hop/two hop/adversarial case
slice essentially never splits its scores entirely between 0 and 1. Computed directly
(`quality.stats.detectable_effect`/`required_n`, reproduced by
`python3 -c "from quality import stats; print(stats.detectable_effect(55, 0.5))"` with
`PYTHONPATH=backend:testing/harness:.`):

| n | `detectable_effect(n, sd=0.5)` | interpretation |
|---|---|---|
| 55 (retrieval relevant slice) | 0.1889 | the smallest paired nDCG delta this slice could reliably see (80% power, alpha 0.05), even under the WORST case variance bound |
| 76 (full seed set) | 0.1607 | the same bound if every hand authored case somehow carried a retrieval relevant score |

The SP7 planning digest's own sizing recommendation (`.superpowers/sdd/sp7-planning-digest.md`,
design question 4) named "roughly a 3 point nDCG delta, 250 to 400 queries" as the target a future
regression check should resolve. `stats.required_n(0.03, 0.5) = 2181`: even under the worst case
variance bound, resolving a 3 point (0.03) nDCG delta needs roughly 2181 cases, nowhere near this
seed set's 55 or 76. This is named here explicitly rather than claimed away: **the seed set, as it
stands, cannot reliably resolve a small nDCG regression; it can only reliably resolve a large one
(at least ~0.19 at n=55 under the worst case bound, and the true detectable effect will be smaller
once the live pass supplies the real, almost certainly tighter, observed variance).** The seed set
is sized for coverage of the registry's own fact space (D16: 50-150, honest count against a 21
entity, 2 contradiction registry), not for matrix scale statistical power -- that is SP9's job (see
Section 6). A future SP9 benchmark row drawing on this dataset must size its own case count against
`required_n`, never assume 55 or 76 is already enough for a small delta.

Once the live pass runs (Section 2), the ACTUAL observed per case nDCG standard deviation replaces
the `sd=0.5` worst case bound above in a follow up edit, and `RetrievalReport.detectable_effect_ndcg`
(already computed by `quality.retrieval_report.evaluate` on every run, live or hermetic fixture) is
the real, tighter number for the CI reported in Section 2 -- this section's bound stays correct and
citable regardless, since it is a mathematical ceiling, not an estimate that could be wrong.

## 4. The SP3 flagship baseline

`seed-flagship-daniel-contract-free` (query `"Is my plan contract free?"`, `expected_doc_ids:
["2514487e4633b47b"]`, the mechanically derived Daniel contract terms chunk) is the seed set's own
pinned encoding of `docs/measurements/sp3-rag-spine.md`'s flagship finding: the BGE reranker demotes
the customer specific override below generically worded "no contract" marketing pages.

**Hermetic today:** the case's own grading contract is fused set MEMBERSHIP only, never a rank or
position, pinned by `test_flagship_baseline_row_is_pinned_and_asserts_set_membership_only`
(`testing/tests/test_seed_dataset.py`) -- `expected_doc_ids` is a plain one item list, and the
dataset schema (`contracts/dataset/schema.json`) carries no rank or position field anywhere, so no
grader over this case (`quality.ir_metrics`, today, or SP8's judge, later) can ever assert a
specific reranked position by construction, independent of any live run.

**Already measured once, SP3, cited here (not re derived):** fused rank 5 of 45, reranked rank 14,
score 0.00136 (`docs/measurements/sp3-rag-spine.md` Section 1,
`testing/tests/test_pgvector_adapter_live.py::test_daniel_contract_free_query_mechanics_gate_and_conflict_measurement`).
Every chunk that outranked it after reranking literally contained the phrase "No contract. Cancel
any time."; the reranker is a generic cross encoder with no notion of a customer specific override.

**PENDING LIVE CAPTURE, this task's own reproduction:** SP7 Task 7's job is to reproduce this live,
once, against the seed set's own case (a slightly different query string, `"Is my plan contract
free?"` capitalized with a question mark, versus SP3's lowercase `"is my plan contract free"` --
both are expected to land on the same neighborhood of the embedding space, close enough that the
reproduction is a real check, not a foregone conclusion) and record the reranked position observed
this run:

- fused (pre rerank) rank: PENDING LIVE CAPTURE of 45 -- MECHANICS, will be gated once run
  (`assert DANIEL_CHUNK_ID in fused_ids`,
  `test_flagship_baseline_fused_membership_gated_reranked_position_measured`).
- reranked rank / score: PENDING LIVE CAPTURE -- QUALITY, measured only, expected in the
  neighborhood of SP3's rank 14 but recorded exactly as observed, never asserted on.

## 5. Generation half -- PENDING LIVE CAPTURE, contingent on Section 2/4's rerun

A provider key IS present in this repo's `.env` (`ANTHROPIC_API_KEY`/`OPENAI_API_KEY` both set,
confirmed by presence check only, never by reading either value). Per the plan's own instruction
("generation half only if a provider key is present... if absent, say so and skip, do not fail"),
`test_generation_half_grounded_in_real_local_retrieval` (same live test file) reuses
`rag_tools.smoke`'s own established generation half exactly (`_generation_half`, `_keyed_provider`):
a single, `MAX_TOKENS=64` bounded grounded completion over real locally retrieved passages, not a
new agentic generation benchmark (Task 7's own plan text scopes the live lane to retrieval; a full
generation benchmark is SP8/SP9 territory). This half needs the same live TEI the retrieval half
does (to retrieve the passages it grounds the completion in), so it is deferred together with
Sections 2 and 4, run by the same rerun command in the callout above.

## 6. Carries forward

**SP8 (judge + human loop):**
- Consumes the seed set and `quality.agent_metrics` directly: the reference based tool call,
  citation, intent confusion, and refusal metrics Task 3 built are the deterministic layer SP8's
  calibrated groundedness judge sits beside, never duplicates.
- `seed-flagship-daniel-contract-free` and the seed set's other `grounded_not_true` cases are the
  conflict slice SP8's judge calibration is seeded from (`docs/measurements/sp3-rag-spine.md`'s own
  "SP8: `conflict-daniel-contract` is the seed for a conflict slice" line, now backed by a real,
  pinned dataset row rather than only a live test finding).
- Human labels (the HITL adjudication page, D30) activate calibration against this same dataset;
  SP7 builds no judge, rubric, or label page itself (the 04/05 grader boundary,
  `CLAUDE.md`/ADR-029's corrected owner column).

**SP9 (variants + matrix):**
- Consumes `testing/harness/quality/stats.py` (relocated + Holm complete, Task 2) and the dataset
  contract/manifest machinery (Task 4/6) for its staged embedder then reranker then generator sweep.
- Must size its own sweep against `required_n`/`detectable_effect`, not assume the 55 case
  retrieval relevant slice already has matrix scale power (Section 3 above names the gap
  explicitly: roughly 2181 cases needed for a 3 point nDCG delta even under the worst case
  variance bound, against 55 available today).
- The flagship baseline row (Section 4) is the named comparison point for any future rerank model
  swap or query rewrite step: measured against this baseline, never a silently broken assertion.

**Informational carries from the T5/T6 reviews** (`.superpowers/sdd/sp7-task-5-review.md`,
`sp7-task-6-review.md`), restated here since T7 is SP7's last task and nothing later in this sub
project revisits them:

- **The RAG registry id space (`corpus/registry/core.yaml`, e.g. `plan-fiber-500`) and the backend
  account/catalog id space (`atlas.domain.catalog.CATALOG`, e.g. `plan_legacy_value`) are genuinely
  disjoint.** Any future case authoring `actions.change_plan`/`catalog.get_plan` arguments must use
  a real backend catalog id, never a registry entity id (the exact defect T6 found and fixed in the
  seed set's two corrected dataset contract examples, `gc-0001`/`gc-0002`).
- **The D33 persona region axis remains deferred, not added.** The dataset contract's `persona`
  block (v0.1.0) carries `name`/`style` only (`additionalProperties: false`); no consumer
  (`quality.agent_metrics.counterfactual_equivalent`, `dataset_tools.counterfactual`) reads a region
  key anywhere, pinned by `test_persona_field_never_carries_a_region_key_region_axis_deferred`. A
  region axis needs a MINOR schema bump plus the CHANGELOG gate plus golden regeneration before any
  future task can populate it; SP7 declines to pay that cost for a field nothing yet reads.

## 7. Dataset contract versioning: unchanged

This task touches no `contracts/*/schema.json` file (`git diff --stat` against this commit shows
zero contract changes); the dataset family stays at `0.1.0`. Per the plan's own binding decision
(digest design question 5, restated in the plan's Architecture section): the freeze is deliberately
not owed now, only the trace family froze after SP6 (ADR-029). No CHANGELOG entry is required for
this task (the fenced `contract-versions` block only moves when a schema file itself changes).
