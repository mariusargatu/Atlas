# Measurement: SP9 variants and matrix, honest intervals and the pareto point in prose

Committed factual record for SP9 Task 8, the last task of the sub project. Follows
`docs/measurements/sp3-rag-spine.md`, `docs/measurements/sp7-datasets-metrics.md`, and
`docs/measurements/sp8-judge-human-loop.md`'s own pattern: every number below is source labeled and
cross referenced to the specific test or command that reproduces it, MECHANICS are gated by the
hermetic suite, PROBABILISTIC outcomes are measured and recorded, never silently gated by a test
that would either be flaky or launder a real finding behind a green check mark.

## LIVE CAPTURE DEFERRED, five lanes, none swept to a real number by this docs only task

Five things in SP9 need a live call or a live sweep and are deferred here, matching SP8 Task 8's
own precedent (a docs only task with keys present in `.env` that deliberately does not spend real
provider money or real wall clock time on its own scope):

1. **The staged matrix sweep itself** (`testing/harness/matrix/`, Task 4 through 7): real embedder,
   reranker, and generator calls against TEI, OpenAI, Anthropic, and Ollama, spend gated. The
   `atlas-fastlane` Hetzner TEI node was recreated and verified healthy on 2026-07-21 (both `/info`
   revisions match `models.lock`: `bge-m3` at `5617a9f6`, the reranker at `953dc6f6`), specifically
   so this sweep and the rest of the DEFERRED LIVE BACKLOG (Section 9) could batch into one live
   capture session. That session has not happened yet: Tasks 4 through 7 built and hermetically
   tested every mechanic the sweep needs (staging, the content hash cache, the spend gate, the cost
   column, the Ollama generator cell) without ever calling `run_matrix` against a real embedder,
   reranker, or generator. A second, more specific gap compounds this: **no Taskfile target or CLI
   entry point for the full staged sweep exists yet at all** (`testing/harness/matrix/__main__.py`
   does not exist; SP9 Task 7's own report names this directly, "no live matrix CLI was built...
   this task stops at library code plus tests"). This is stated here plainly rather than implied,
   the same "discover or confirm it, do not assume it" instruction `docs/measurements/
   sp8-judge-human-loop.md` Section 3.2 already applied to `judge:calibrate`. See Section 2 for the
   composition an operator would run in its place.

   **Addendum, dated after the paragraph above (2026-07-21, this sub project's own live wrap up):**
   the CLI gap named above closed later the same day: `testing/harness/matrix/__main__.py` plus
   `live_driver.py`/`live_search.py` (commit `d4db0eb`) wire `task matrix:live` to real
   `TeiEmbeddingClient`/`OpenAiEmbeddingClient`, a real `Reranker.rerank` adapter, real Claude/GPT/
   Ollama generator builders, and the variant comparison stage (item 5 below) into `run_matrix`,
   hermetically tested keylessly (`test_matrix_live_driver.py`, `test_matrix_live_search.py`,
   `test_matrix_main.py`). A live capture session was then run against the recreated node named
   above and hit a genuine infrastructure wall rather than completing a sweep. See Section 2's own
   new closing subsection for the full, measured outcome.
2. **The real Neo4j LLM extraction run, PENDING LIVE CAPTURE** (`task graph-up`, SP9 Task 2). Needs
   the `graph` dependency group and the compose `neo4j` service; `registry_graph.EXTRACTED_*` is a
   hand authored fixture standing in for a genuine `SimpleKGPipeline` run today. See Section 1.
3. **The load lane's real k6 burst sweep** (`task load:k6` then `task load:join`, SP9 Task 6).
   Needs a k6 binary built with the `xk6-sse` extension on PATH and the burst tier's own ingress
   (D3/7.1: local numbers are never quoted for a saturation knee). See Section 6.
4. **The real Ollama daemon call and the human spot check sample** (SP9 Task 7). Needs a running
   Ollama daemon with `qwen2.5:7b` pulled, and, once a live matrix run exists, a real
   `matrix.generators.GenerationCell` to build the spot check items from. See Section 7.
5. **The variant comparison stage, PENDING LIVE CAPTURE** (`matrix.variants.run_variant_comparison`,
   wired into `run_matrix` as the optional `variants` argument, SP9's final review fix wave, commit
   `0283796`, landed after this document was first authored). Naive vs agentic vs graph, measured
   over the SAME cases with the SAME `quality.agent_metrics`/`quality.ir_metrics` every other stage
   uses, so any difference between the three rows is genuinely the variant's own mechanism, never a
   different corpus or a different model; two hermetic scenarios already prove the rows are
   genuinely distinct (graph traversal recovers a chunk naive's fixed truncation buries, and the
   agentic bounded rewrite recovers a chunk only a rewritten query returns). No live call has ever
   exercised it: every hermetic test injects a fixture retriever/reranker/graph/gateway, never
   `PgvectorRetriever`/`PgKnowledgeGraph`/a real generator. It rides the SAME batched live capture
   session as item 1 above: `task matrix:live` builds its own `VariantsConfig` from real components
   and threads it into `run_matrix` alongside the staged stages, pre checked against the SAME spend
   gate before any paid generator call it needs is ever made.

Everything else in this document is real today: hermetic facts already gated by `task test`, a
number recomputed directly from the real committed registry or a committed fixture, or a citation
of an already committed measurement from an earlier sub project's own document. No live call is
needed for any of it.

## 1. The three variants: hermetic facts today (naive, agentic, graph)

**Naive** has no standalone module: production retrieval lives embedded inside
`atlas_graph.py`'s `_knowledge_call`, exactly what `testing/tests/test_naive_variant_live.py`'s own
docstring states directly ("the chat pipeline's retrieved context via the one seam that is both
real and assertable: the exact MCP tool the running graph calls, not a hand rolled alternative").
SP9 Task 1's own report names this as the reason no extraction into a shared `naive_rag.py` module
happened: the `query in / chunks plus an answer out` contract (D6) is DEFINED by Task 1, for the
other two variants to match, rather than lifted from an existing naive module that does not exist
at that granularity.

**Agentic** (`backend/atlas/orchestration/agentic_rag.py`, SP9 Task 1, commit `2b5a3a1`): seven
nodes, every decision a named conditional edge. `route_query` reuses `atlas.domain.binding.
classify_intent` verbatim (never a second heuristic); `retrieve` and `rerank` call the same
`Retriever`/`Reranker` ports the naive path's own MCP tool call and this sub project's other
variants all share; `grade_documents` is CRAG style, per chunk relevance via registry `entity_ids`
overlap, cross checked directly against `quality.agent_metrics.citation_precision_recall` on a
shared fixture so the two formulas can never silently drift apart
(`test_grade_documents_overlap_matches_citation_precision_recall_on_a_shared_fixture`); a
conditional edge to `rewrite_query` is bounded to exactly one retry via a `_AGENTIC_BUDGET =
Budget(max_tool_calls=12, max_retrieval_rounds=2)` instance from `atlas.domain.budget`, never a new
counter; `generate` and `check_faithfulness` close the loop, a still unfaithful second answer
shipping with a disclosure suffix rather than a silent pass or a third attempt. The tamper case
(both the original and the rewritten query resolve to the same entity unlinked chunk, forcing
`grade_documents` to fail forever) still stops at exactly one retry
(`test_tamper_forcing_perpetual_low_grades_still_stops_at_the_budget`, `testing/tests/
test_agentic_rag.py`, 10 tests total, all passing hermetically against `InMemoryRetriever` plus
`GatewayChatModel` in REPLAY mode).

**Graph** (`backend/atlas/orchestration/graph_rag.py` plus `backend/atlas/adapters/
pg_knowledge_graph.py`, SP9 Task 2, commit `0788fb4`): six linear nodes (no loop, unlike the
agentic variant), `resolve_entities` into `traverse` (a fixed two hop breadth first loop over
`KnowledgeGraph.neighbors`) into `retrieve` into `collect_chunks` (`domain.graph_retrieval.
collect_chunks_by_entities`, the pure entity to chunk join) into the same `rerank`/`generate` nodes
the agentic variant uses. `PgKnowledgeGraph` is the GOLD graph, a `WITH RECURSIVE` Postgres query
over two new tables materialized from the real registry; its headline hermetic case
(`test_graph_rag_variant.py`) proves the graph's own multi hop value directly: entity linking pulls
in a chunk the query's own words never mention at all, not just plumbing that happens to pass.
`Neo4jKnowledgeGraph`/`neo4j-graphrag`'s `SimpleKGPipeline` is repositioned, not discarded, as the
LLM EXTRACTED comparison arm, scored against the same gold graph by `quality.graph_metrics`
(relocated from `testing/harness/evals/retrieval/graph_metrics.py` via `git mv`, the exact SP7
relocation pattern already applied to `stats.py`/`ir_metrics.py`). Neither graph is dead weight;
both are named explicitly in the adapter, port, `pyproject.toml`, and compose file docstrings.

**The extraction comparison numbers below are FIXTURE DERIVED, not a real distribution claim; the
real extraction run is PENDING LIVE CAPTURE** (Section LIVE CAPTURE DEFERRED item 2, rerun command
`task graph-up`). `registry_graph.EXTRACTED_*` is a hand authored fixture standing in for a real
`SimpleKGPipeline` run, with three documented failure modes (a missed relation,
a spurious relation, a fractured cluster entity merge), each justified in the module's own
docstring against a plausible real extraction failure, not a strawman. Computed against the real
registry's 19 edges (`available_in` x9, `applies_to` x2, `overrides_fee` x1, `compatible_with` x6,
`supersedes` x1, verified against the loaded registry directly):

| metric | value | test |
|---|---|---|
| `triple_prf` precision | 15/16 = 0.9375 | `test_triple_prf_scores_the_documented_miss_and_spurious_edge` |
| `triple_prf` recall | 15/19 = 0.7895 | same test |
| `triple_prf` F1 | exactly 6/7 = 0.8571 | same test |
| `bcubed_prf` precision | 0.75 (the fractured merge charges precision, recall stays 1.0) | `test_bcubed_charges_precision_not_recall_for_the_false_merge` |
| `bcubed_prf` F1 | 6/7 | same test |
| `pairwise_prf` | (0.0, 0.0, 0.0), degenerate by construction (every gold cluster is a singleton) | `test_pairwise_f1_is_zero_when_every_gold_cluster_is_a_singleton` |
| `path_recall` on `plan-fiber-100 -> region-north -> fee-equipment-rental` | 0.0 (the one missed relation removes every incoming edge into the fee) | `test_path_recall_collapses_to_zero_on_the_named_plan_region_fee_chain` |

All four numbers are pinned exactly (not approximately illustrated) in `testing/tests/
test_graph_extraction_comparison.py`; `task graph`'s own printed output matches them to three
decimal places, confirmed at authoring time.

## 2. The staged matrix runner: mechanics real and hermetically gated, the sweep PENDING LIVE CAPTURE

`testing/harness/matrix/` (SP9 Task 4, commit `3e79f25`; Task 5's spend gate, commit `8bb7d2c`;
Task 7's Ollama cell, commit `c2668ba`): a caller of SP7/SP8's already built substrate, never a
rebuild of any of it.

**Stage 1, embedders**, retrieval only metrics (`recall@k`, `nDCG@k`), no LLM call anywhere: the
two real axes `{bge-m3 local, text-embedding-3-small openai}`, the documented narrowness (no
Voyage key), plus two named baseline rows every table carries, `BM25_COMPONENT_ID =
"bm25-no-reranker"` and `EXACT_SCAN_COMPONENT_ID = "exact-scan"` (the recall ground truth row),
proven present by `test_bm25_and_exact_scan_baseline_rows_are_present`.

**Stage 2, rerankers**, over stage 1's own cached candidates at depths `{20, 50, 100}`
(`rerankers.DEPTHS`), axis `{BGE reranker v2 m3, none}` (the documented narrowness, no Voyage
`rerank-2.5-lite` key). Still no LLM call anywhere.

**Stage 3, generators**: the top one to two retrieval configs (`select.select_top_configs`, ranked
by nDCG point estimate, ties broken by `config_id` ascending, never rerun) times the generator axis
`{Claude, GPT, qwen2.5:7b}`, scored by `quality.agent_metrics.answer_correctness_rate` (primary,
reference based) and `judge.panel.panel_vote` (secondary, calibrated) --
**`panel_vote`'s first real caller anywhere in this repository**, proven by
`test_panel_vote_ran_in_stage_3_disagreement_and_labels_present` (`test_matrix_runner.py`) and
`test_panel_vote_is_really_invoked_majority_and_disagreement_flag_are_correct`
(`test_matrix_generators.py`). Plus one off diagonal validation cell (research 14's own
recommendation), recording, never asserting, whether the retrieval stage's own best ranking
predicts the generation stage's ranking too
(`test_off_diagonal_check_is_present_recorded_not_asserted`).

**Determinism and lineage**: two calls to `run_matrix` over identical inputs produce a byte
identical `manifest.json` (`test_two_runs_produce_a_byte_identical_manifest`, and again with two
independent fresh caches, `test_two_runs_with_independent_fresh_caches_still_produce_a_byte_
identical_manifest`); the content hash cache (`cell_key = hash(corpus_version, dataset_version,
component_id, params)`, the same digest the cassette key already uses) skips recompute on a rerun
(`test_content_hash_cache_skips_recompute_on_a_full_run_rerun`); every stage row's own D26 lineage
tuple validates directly against `contracts/manifest/schema.json`
(`test_every_stage_rows_lineage_validates_against_the_manifest_contract`). Every generator/judge
call in every one of these tests runs through `replay.gateway.GatewayChatModel` pinned to REPLAY
mode against seeded cassettes, keyless, zero egress.

**The spend gate and the cost column (SP9 Task 5)**: cumulative per provider dollar tracking,
ceilings `CEILINGS_USD = {"openai": 20.0, "anthropic": 10.0, "ollama": 0.0}` (Ollama in the
`ALWAYS_RUNS` frozenset, never rationed against a remaining balance at all). A cell over its
provider's remaining budget is skipped before it ever runs and logged into the manifest's own
`dropped_cells` list, never silently capped
(`test_a_cell_over_budget_is_skipped_and_logged_never_silently`,
`test_an_always_runs_provider_is_never_dropped_even_with_a_huge_estimate`). The cost column is
backward compatible: `Cassette.from_result` now also persists `usage_metadata` when a live
provider returns one; an OLD cassette (recorded before this task, carrying no `usage_metadata` key
at all) still replays green, cost reported as unavailable, never a silently wrong zero. This is a
real, already closed trace bump, not itself pending: `atlas.cost.input_tokens`/`output_tokens`/
`usd` moved from narrowed to emitted at the trace family's 1.2.0 to 1.3.0 MINOR bump, both the
hermetic translation AND a real live capture against a real OTel collector performed in the same
commit (`docker compose --profile observability up`, `contracts/trace/freeze_evidence.json`'s
`sp9_task5_addendum`, 5 new export lines), `freeze_check` reads clean today (all 30 reserved
attributes emitted or narrowed). What is PENDING is the real dollar figure a live matrix cell would
actually spend, not the mechanism that would capture it.

**PENDING LIVE CAPTURE: no real embedder, reranker, or generator has swept this dataset yet.**
Every recall@k, nDCG@k, paired delta, Holm corrected p value, panel agreement rate, and dropped cell
reason a real sweep would produce is unmeasured today; every hermetic test above proves the
MECHANICS (staging, caching, gating, lineage, determinism) against seeded fixture callables, never
against a real TEI, OpenAI, Anthropic, or Ollama response. Composition an operator runs in the
absence of a dedicated CLI (mirroring `docs/measurements/sp8-judge-human-loop.md` Section 3.2's own
"the composition, not a single command" disclosure for `judge:calibrate`); a dedicated CLI now
exists too (`task matrix:live`, see the closing subsection below), so this hand assembled composition
is kept here as the lower level, library only equivalent, not the only way to run this:

```
docker compose up -d postgres
set -a; source .env.fastlane; set +a
PYTHONPATH=backend:testing/harness:. uv run --all-groups --env-file .env python -c "
from matrix.runner import run_matrix, MatrixRunConfig
from matrix.spend_gate import SpendGate
from matrix.cache import MatrixCache
from matrix.ollama_generator import build_ollama_generator_component
# construct TeiEmbeddingClient / OpenAiEmbeddingClient, a real Reranker.rerank adapter,
# the Claude/GPT GeneratorComponent builders (not built by this sub project, see Task 7's
# own carry), plus build_ollama_generator_component for the free arm, then call run_matrix
# with a SpendGate() and a MatrixCache(<a real directory>).
"
```

**The live capture attempt (2026-07-21) and the measured CPU reranker saturation finding, honest
outcome not a bare placeholder.** The driver named in the addendum above (`task matrix:live`,
`testing/harness/matrix/__main__.py` + `live_driver.py` + `live_search.py`, commit `d4db0eb`) is
BUILT, HERMETICALLY TESTED, and READY today: it needs no further code, only a live TEI endpoint to
point at. A live capture session was run the same day against the recreated `atlas-fastlane` cpx41
node (the same node whose `/info` revisions were verified against `models.lock` above) and hit a
genuine infrastructure wall rather than completing a sweep: no live recall@k, nDCG@k, paired delta,
or dropped cell reason exists for this dataset yet, exactly as pending as the paragraph above states,
now with the measured reason recorded rather than a bare TODO.

The measured, reproducible finding: the BGE reranker v2 m3 cross encoder is CPU bound, not memory
bound, roughly 0.5 seconds per short candidate, about 14 seconds for a batch of 30 candidates, and 6
or more seconds per long corpus chunk. On realistic candidate pools, 20 to 100 long chunks, exactly
the range this matrix's own reranker depths `{20, 50, 100}` sweep over the 55 to 76 case seed set,
the reranker saturates all 8 cores on the node hard enough that TEI cannot even serve its own
`/info` endpoint concurrently; this presents as a hang rather than a crash, and the node recovers
within about 10 seconds once the load stops. The node wedged three separate times under this
workload (a rerank timeout, an embed timeout, and a connection refused, in that order): TCP kept
accepting connections while the server process itself stopped answering. Embedding alone stayed fast
and unaffected throughout, 0.12 to 0.56 seconds per call: the wall is squarely the cross encoder
reranker running on CPU, never the embedder axis, so stage 1 above is exactly as ready to run live as
the driver claims; it is stage 2's own reranker depth sweep a CPU serving tier cannot carry at these
candidate pool sizes.

Honest conclusion, stated plainly rather than papered over with a placeholder: the full live matrix
sweep at reranker depths `{20, 50, 100}` over this seed set is impractical on a CPU backed TEI tier
and needs either a GPU backed reranker endpoint or a strictly bounded candidate pool in production,
never an unbounded one. This is not a failure of the driver, the plan, or this task; it is the same
class of finding Section 4's own flagship conflict slice already put a number on, a cross encoder
reranker changing which chunk answers a query, now extended into an operational cost finding: the
reason a real deployment's reranker stage needs GPU serving or a hard pool size cap is that CPU
serving cannot carry it at the depths this matrix's own design already calls for, which is exactly
why the SP3 flagship rerank finding matters for more than one query in isolation. The node was
deleted immediately after (billing stopped, per this sub project's own "no infrastructure survives
past its own use" discipline); rerunning this sweep needs no new code at all, only a healthy, GPU
backed TEI endpoint pointed at by `.env.fastlane`, then the exact `task matrix:live` command above,
run against that endpoint.

## 3. Honest interval width: what this n can actually detect (hermetic, real today, cited not recomputed)

`docs/measurements/sp7-datasets-metrics.md` Section 3 already computed and disclosed this; SP9
cites the SAME numbers rather than recomputing a rosier version, per this sub project's own
"HONEST STATISTICS" global constraint. `quality.stats.detectable_effect`/`required_n` are pure
functions of `n` and a per case score standard deviation; a defensible upper bound on that
deviation needs no observed data at all, only that a bounded `[0, 1]` metric has maximum population
variance `0.25` (standard deviation `0.5`), a deliberately pessimistic ceiling:

| n | `detectable_effect(n, sd=0.5)` | interpretation |
|---|---|---|
| 55 (SP7's retrieval relevant slice) | 0.1889 | the smallest paired nDCG delta that slice could reliably see, even under the worst case variance bound |
| 76 (the full seed set) | **0.1607** | the same bound if every hand authored case somehow carried a retrieval relevant score |

`required_n(0.03, 0.5) = 2181`: resolving the roughly 3 point nDCG delta research 14 names as a
target needs roughly 2181 cases, nowhere near this seed set's 55 or 76. **The staged matrix's own
per query result files (Section 2) draw on this SAME 76 case seed set** (`matrix.cases`, reusing
`dataset_tools.manifest.case_slice`'s own established buckets, the same 55 of 76 cases `test_sp7_
retrieval_metrics_live.py` already runs live retrieval over); every paired 95% CI the matrix's
`quality.stats.paired_bootstrap_diff` computes in Section 2 is therefore subject to this SAME
honest small n reality. A matrix cell reporting a small delta with a wide CI is not a bug in the
matrix; it is this exact, already disclosed statistical ceiling, restated here rather than
forgotten because a new sub project is doing the measuring.

## 4. The flagship SP3 conflict slice: cited, not re announced

`seed-flagship-daniel-contract-free` (`docs/measurements/sp3-rag-spine.md` Section 1,
`docs/measurements/sp7-datasets-metrics.md` Section 4): the BGE reranker demotes the Daniel contract
terms chunk (the customer specific override), fused rank 5 of 45, reranked rank 14, score 0.00136;
every chunk that outranked it after reranking literally contained the phrase "No contract.
Cancel any time." **Already measured once, cited here, never re derived or re announced as a fresh
SP9 finding.** `docs/measurements/sp7-datasets-metrics.md` Section 4's own live reproduction of this
same finding (a slightly different, capitalized query string against the seed set's own pinned
case) is itself still marked PENDING LIVE CAPTURE in that document; SP9 does not attempt a third,
independent reproduction here.

This slice lands as a named cell in the matrix's own rerank and generation stages once the live
sweep (Section 2) runs: the reranker axis `{BGE reranker v2 m3, none}` over the seed set's
`seed-flagship-daniel-contract-free` case is exactly the reranker on versus off comparison this
finding is about, and the matrix's own paired bootstrap CI plus Holm correction (Section 2, Section
3) is what actually says, with an honest interval, whether this one hand identified case
generalizes past itself. Nothing in this section fabricates a matrix specific number; the finding
stays exactly what SP3 and SP7 already recorded until a live sweep adds a third, comparable
reading.

## 5. The load lane: hermetic helpers real, the burst sweep PENDING LIVE CAPTURE

`testing/harness/load/` (SP9 Task 6, commit `b81ab07`): a k6 plus `xk6-sse` script (182 lines) against
the real `/chat/stream` SSE endpoint, stepped concurrency 1 to 32 VUs, custom Trend metrics
(`ttft_ms`, `tokens_per_sec`, `e2e_ms`) and a Rate metric (`goodput`). Thresholds as code, one
committed file (`thresholds.json`) read by both the Python parser and the k6 script itself, so the
two can never silently disagree: `ttft_ms` p95 under 2000ms, `tokens_per_sec` p50 over 5, `e2e_ms`
p95 under 8000ms, `goodput` rate over 0.95, honestly documented as placeholder targets a live
burst run's own first pass would recalibrate, never measured numbers themselves
(`testing/harness/load/thresholds.py`, `test_load_thresholds.py`). The Phoenix join
(`phoenix_join.py`) over `atlas.turn.seq` (SP6's fix wave repaired join key) is verified against a
REAL, in process `OtelTracer` (`test_the_join_key_constant_matches_what_a_real_otel_tracer_
actually_stamps`), not assumed from prose alone. All 46 of this task's own new tests
(`test_load_thresholds.py`, `test_load_prompt_corpus.py`, `test_load_phoenix_join.py`) pass
hermetically, keyless, zero egress.

**PENDING LIVE CAPTURE: the real burst sweep has not run.** No TTFT, tokens per second, e2e
percentile, goodput rate, or per stage latency by concurrency step exists yet for any real request
against the real `/chat/stream` endpoint. Per D3/7.1's own rule (local numbers are never quoted for
a saturation knee), this sweep only ever runs on the Hetzner burst tier, never localhost. The
expected shape, predicted not measured: the saturation knee lands first at the reranker (a cross
encoder scoring k candidates per query on CPU, an order of magnitude more compute than one embedding
call), per D31's own framing; pgvector's own concurrent behavior under real load is itself an open
question this run is meant to answer, not the published serial single connection numbers research 24
already names as a literature gap. Rerun command, copy pasteable, real Taskfile targets:

```
k6 run testing/harness/load/k6/chat_sse_load.js
# then, once a Phoenix span export exists as a local JSON file in phoenix_join.SpanRecord's shape:
PYTHONPATH=backend:testing/harness:. uv run python -m load --iterations <path> --spans <path>
```

(`task load:k6` then `task load:join`, both real, committed Taskfile targets, burst tier only,
never gated by `task test`.)

## 6. D28's local generator arm: ADR-030 cited, wiring real, the live call PENDING

`ADR-030` (SP9 Task 7, commit `c2668ba`) documents
the substitution explicitly, cited here rather than re argued: D28's own text names one GPU burst
arm, vLLM plus Qwen3-8B on a rented spot GPU; SP9 has no RunPod account, key, or GPU tier
provisioned (a separate, not yet authorized spend and infrastructure decision), so `qwen2.5:7b` on
Ollama CPU decode substitutes for the local generator cell. What is preserved: D28's own
qualitative claim (small models can be more faithful than frontier ones under imperfect retrieval,
and their failure mode is hallucination rather than abstention) stays testable, scored by the same
judge panel and reference based metric every other generator cell uses, human spot checked before
its verdicts are trusted. What is not preserved, named plainly in the ADR: `qwen2.5:7b` is not
Qwen3-8B (the Vectara HHEM leaderboard numbers ADR-030 cites, 4.8% for Qwen3-8B against 9.6% for
GPT-4o and 10.3% for Claude Sonnet 4, describe the substituted away model, not the substitute); CPU
decode is not GPU serving. This document does not restate or re derive those numbers a second time;
see ADR-030 itself.

**`testing/harness/matrix/ollama_generator.py`**: `build_ollama_generator_component` is the first
real code anywhere in this repository turning `models.lock`'s pinned `ollama`/`qwen2.5:7b` entry
into an actual `GeneratorComponent`, routed through `matrix.spend_gate.build_generator_gateway`
(RECORD mode, `estimated_usd` always `0.0`, the `ALWAYS_RUNS` provider). Keyless in every hermetic
test (11 tests, `test_matrix_ollama_generator.py`): an injected stub `BaseChatModel` stands in for
the real Ollama daemon everywhere; `_live_ollama_model`, the only place `replay.providers.
build_chat_model("ollama", ...)` is reached, is a function body import never exercised hermetically.

**`testing/harness/matrix/ollama_spot_check.py`**: `build_ollama_spot_check_items` reuses SP8's
existing HITL machinery end to end, per the plan's own explicit instruction not to build a second
labeling surface. `build_ollama_spot_check_items` produces the identical label item shape `labeling.
generate_label_set.generate_label_items` already produces, loading unmodified through
`atlas.label_routes.build_label_router`'s existing `ATLAS_LABEL_ITEMS_PATH`/`task label:generate-
live` seam. Proven for real, not just asserted: the capstone test of the 13 in `test_matrix_ollama_
spot_check.py` builds a small sample, writes it, and loads it through a real FastAPI app built from
`build_label_router` plus a real `LabelStore`, `GET /labels/items` and `POST /labels` both round
tripping through the same store `test_label_routes.py`'s own tests exercise.

**PENDING LIVE CAPTURE: neither the real Ollama daemon call nor a real spot check sample exists
yet.** `_live_ollama_model` has never been exercised against a running daemon; `build_ollama_spot_
check_items` has never been called against a real matrix run's own `GenerationCell` (only a hand
built fixture in its own tests). Both need the batched live capture session (Section 2); no
dedicated Taskfile target wraps either step today, matching Section 2's own disclosed CLI gap. Once
a live matrix run produces a real Ollama `GenerationCell`, the composition is: call `build_ollama_
spot_check_items(cell, config, run_id=..., limit=<a small sample size>)`, write it with `labeling.
generate_label_set.write_label_items`, then point `ATLAS_LABEL_ITEMS_PATH` at the output file
before starting the backend, exactly the existing `/adjudicate` page's own documented flow.

## 7. The pareto point, in prose only

SP11 builds the actual rendered chart; this section sketches, in prose only, the shape a pareto
frontier across this sub project's own three axes (retrieval or answer quality, dollar cost,
latency or round trips) is expected to take once the live sweep (Section 2) supplies real points to
plot. No chart, no table of fabricated numbers, and no invented point estimate appears below.

**The cheap, fast corner.** The naive path (no rewrite loop, no graph traversal, a single generation
call) plus the `bm25-no-reranker` baseline sits at the cheapest, fastest corner of the frontier by
construction: one retrieval call, no reranker compute, one generation call. Its quality position is
the floor every other cell is measured against, not a claim that it is a bad choice; a real
workload whose questions never need multi hop reasoning or self correction may find this corner is
already good enough, exactly the kind of question the matrix's own paired CIs (Section 2, Section
3) are built to answer honestly rather than assume.

**The reranker axis is a real quality lever, but not a uniform win.** Section 4's own flagship
finding is the concrete reason this frontier is not a simple staircase: a generic cross encoder
reranker can demote a customer specific override below generically worded pages that happen to
share more surface vocabulary with the query. The expected shape is therefore a frontier with a
real bend, not a straight line, once enough cells exist to see it: the reranker axis likely
improves quality on average across the seed set's broader case mix while genuinely costing the one
named conflict slice cell something, and depth `{20, 50, 100}` trades added cross encoder compute
for a chance to recover a chunk a shallower depth would have truncated away before the reranker
ever saw it.

**The agentic and graph variants both trade round trips for a specific capability, not for
capability in general.** Agentic's bounded rewrite plus its bounded regenerate together can cost up
to two extra LLM calls beyond naive's single pass (one rewritten retrieval round, one regenerate
pass), purchasing a self correction property naive has none of; whether that cost is worth paying
depends entirely on how often the CRAG grade actually fires low on a real workload, an empirical
question this document does not answer by assertion. Graph's own headline hermetic finding
(Section 1: entity linking pulls in a chunk the query's own words never mention at all) is a real
recall capability naive's single embedding call structurally cannot have, purchased at the cost of
one Postgres recursive CTE round trip per query; its value is concentrated on genuinely multi hop
questions and is expected to look like pure overhead on a single hop question that naive retrieval
already answers correctly.

**The local generator sits at the zero cost corner with an unmeasured quality position.** `qwen2.5:
7b` on Ollama is free by construction (Section 6, `ALWAYS_RUNS`), the one axis point whose dollar
cost is not a projection but a fact; its position on the quality axis is genuinely unmeasured today,
not merely unmeasured for lack of a live run, but unmeasured BECAUSE ADR-030's own numbers describe
a different model (Qwen3-8B, not the CPU substitute actually wired here). Its faithfulness or
hallucination behavior under this repository's own imperfect retrieval cases is exactly what the
matrix's judge panel scoring plus the human spot check (Section 6) exist to measure honestly, not
assume from a leaderboard number that does not describe this exact model.

**What the eventual rendered chart (SP11) needs from this sub project, stated as a because
clause:** because every cell's own paired CI (Section 3) can be wide at this dataset's honest n,
SP11's frontier should render interval bars alongside every point, never a bare point estimate that
would visually overstate confidence the underlying statistics do not have.

## 8. Contract versioning: unchanged by this task

The trace contract's MINOR bump (1.2.0 to 1.3.0) already landed in SP9 Task 5 (`8bb7d2c`),
committed, `ADR-029` amended with a new Amendment
section, `contracts/trace/freeze_narrowed.yaml` corrected, freeze clean at 30 of 30 reserved
attributes (Section 2). This task (8) touches no `contracts/*/schema.json` file and adds no new
contract; `git diff --stat` against this commit shows zero contract changes. No CHANGELOG entry is
required for this task, the same "docs only, no contract touched" reasoning `docs/measurements/
sp7-datasets-metrics.md` Section 7 and `docs/measurements/sp8-judge-human-loop.md` Section 9 both
already applied to their own last tasks.

## 9. Carries forward

**Immediate, informational, folded in here rather than silently repeated:**

- **The judge_label live content flattening caveat** (`docs/measurements/sp8-judge-human-loop.md`
  Section 6.1) applies directly to this sub project's own first real `panel_vote` caller (Section
  2): a live Anthropic reply whose `content` returns as a list of content blocks rather than a bare
  string makes `judge.llm_judge.judge_label`'s `text.strip()` raise `AttributeError`, not a silent
  bad parse. `judge.live_provisional._sweep`'s own per tier `try`/`except` already fails this SAFE
  (a caught exception, never a wrong label); the same protection applies wherever the matrix's own
  panel calls Anthropic through the identical `judge_label` function. Not fixed here or in Task 4;
  carried forward as a known caveat for the live sweep, exactly as SP8's own document already named
  it.
- **The Neo4j LLM extraction arm and the KEDA autoscaling seam** are both documented, deferred
  boundaries, never silently dropped: `task graph-up` is the real, existing operator command for
  the former (Section LIVE CAPTURE DEFERRED item 2); KEDA on `te_queue_size` is named in three
  places in the load package (`testing/harness/load/__init__.py`'s own docstring, the k6 script's
  header comment, the `load:k6` Taskfile description) and built nowhere, per the plan's own explicit
  instruction.

**SP10 (the burst benchmark CI lane):**

- SP10 is the sub project that actually RUNS this matrix on a schedule: the trigger, the protected
  environment, the OIDC backed `tofu apply` against the burst tier, and the janitor that tears the
  tier back down afterward are all SP10's own territory, not built here. Everything SP10 needs
  already exists and is hermetically proven today: the staged runner (Section 2), the spend gate
  (Section 2), the content hash cache (Section 2, so a scheduled rerun only recomputes what actually
  changed), and the manifest's own D26 lineage rows (so a CI run's numbers trace back exactly two
  hops, the same discipline this document itself follows). SP10 supplies the schedule and the
  infrastructure; it should not need to rebuild any of the runner's own mechanics.

**SP11 (the rendered report pages):**

- SP11 renders the actual pareto chart Section 7 only sketches in prose, reading real matrix
  manifests (Section 2) once the live sweep has produced them, with interval bars on every point
  (Section 7's own because clause).
- SP11 should not assume the live matrix sweep, the real burst load numbers, or the real Ollama spot
  check sample already exist by the time it finalizes portfolio pages; as of this document, none of
  the four LIVE CAPTURE DEFERRED lanes (Section LIVE CAPTURE DEFERRED) have run. Recompute per
  Section 2's composition (or, once SP10 lands, per its own scheduled run) before any portfolio page
  reads on a real matrix number.
- The flagship conflict slice (Section 4) stays a citation, never a number SP11 re derives on its
  own; if a later live sweep produces a genuinely new reading for that same case, that reading
  belongs in a future document's own Section 2, not silently substituted into this one.
