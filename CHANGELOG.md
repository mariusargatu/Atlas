# Changelog

Sub project history for Atlas, the reference system for the "Evals Are Checks, Not Tests" series.
One block per sub project, factual, drawn from the planning ledger (`.superpowers/sdd/progress.md`)
and the sub project plans. This is a project history record, not a
semantic versioned package: sub projects are numbered SP1 onward, most recent first, and a new
block lands at sub project completion or whenever a `contracts/*/schema.json` family changes
version.

The fenced block below is machine parsed by `contract_tools.changelog_gate` (SP6 task 6, D37): it
must equal `contract_tools.loader.contract_versions()` exactly, family for family, updated in the
same commit as any `contracts/*/schema.json` version bump. This file consistency check is the
hermetic half, folded into `task test`. The git aware half, a contract touching change landing with
no CHANGELOG entry at all, lives only in the push activated CI workflow
(`.github/workflows/release.yml`), never in the hermetic lane.

```contract-versions
trace: 1.3.0
dataset: 0.1.0
manifest: 0.1.0
sse: 0.1.0
```

## SP9 task 5: hard spend gate and a real cost column via backward compatible usage capture (2026-07-21)

The matrix's spend gate (`testing/harness/matrix/spend_gate.py`, pure): cumulative per provider
dollar tracking against a hard ceiling (OpenAI $20, Anthropic $10, Ollama always $0 and always
runs, never rationed against a remaining balance at all), a small pricing table of its own
(deliberately not `judge/live_provisional.py`'s `_PROVIDER_TIERS`, that module's own docstring
already disclaims reuse here). A cell that would exceed its provider's remaining budget is SKIPPED
and logged (`DroppedCell`/`dropped_cell_for`), never silently capped. Generator calls route through
`replay/gateway.py`'s existing RECORD mode (`build_generator_gateway`), so an unchanged cell rerun
replays for free; no new record/replay mechanism.

The cost column, backward compatible: `replay/cassette.py`'s `Cassette.from_result` now also
persists a RECORD mode call's `usage_metadata` (token counts) when the live provider actually
returned one; `to_chat_result` reads it back with a `None` default. An OLD cassette (recorded
before this task, no `usage_metadata` key at all) still replays green, cost reported as
UNAVAILABLE (`matrix.spend_gate.cost_from_usage` returning `None`); a NEW RECORD mode capture
carries a real number. `CASSETTE_VERSION` is unchanged (still `1`): the addition is optional and
additive by construction, never a breaking shape change.

Trace bumps **1.2.0 to 1.3.0** (additive MINOR): the usage accounting trio
(`atlas.cost.input_tokens`/`output_tokens`/`usd`), narrowed since the v1.0.0 freeze, now carries a
real emitter, `testing/harness/matrix/cost_emission.py`'s `emit_cost` (a new `kind="llm"` span,
`generator_cost`, the same D29 batch, report time disposition the judge's own emitter already
established -- never wired into the live graph). `ADR-029` and `contracts/trace/freeze_narrowed.yaml` are updated in this same commit (the ADR's
own rule): the three attributes are removed from the narrowed set, three remain (all RAG
observability). `contract_tools.freeze_check` reports all 30 reserved attributes emitted or
narrowed, never missing, backed by both a hermetic translation run (the regenerated
`contracts/trace/span_inventory.json`) and a real `docker compose --profile observability` capture
appended to `contracts/trace/freeze_evidence.json`.

## SP8 task 1: groundedness judge with versioned identity and trace emission (2026-07-20)

The judge (`testing/harness/judge/`): `JudgeContract`'s versioned identity triple (model id, rubric
version, prompt template hash) absorbed verbatim from the pre rewrite `evals/judge/contract.py`
(D15's own rule word for word); a fresh binary groundedness rubric (every claim in the answer
entailed by the cited retrieved context; an abstention passes; any unsupported claim fails),
replacing the pre rewrite lane's helpfulness and account truth pair; the parsing mechanics
(`judge_label`/`order_swap`/first-standalone-token-wins, fail closed on anything unparseable)
absorbed unchanged; the model call routed through the record/replay gateway exactly like the pre
rewrite judge, hermetic and keyless under REPLAY. The judge's own PASS/FAIL prompt vocabulary stays
independent of the trace contract's wire vocabulary; `translate_verdict` is the one function that
crosses that boundary.

Trace bumps **1.1.0 to 1.2.0** (additive MINOR): four attributes reserved since v0.1.0 and narrowed
at the v1.0.0 freeze now carry a real emitter. `atlas.judge.id`/`atlas.judge.rubric_version`/
`atlas.judge.verdict` are stamped by a new `kind="judge"` span (`judge.emission.emit_verdict`, span
kind `EVALUATOR`); `atlas.subject.pseudonym` is an HMAC-SHA256 of `customer_id` (never a plain
digest, never the raw id), threaded through `atlas_graph.py`'s "turn" `tracer.open` call site as a
new kwarg, sourced from the session/bearer identity exactly like every other per-customer decision
in that function, never the model. `ADR-029` and
`contracts/trace/freeze_narrowed.yaml` are updated in this same commit (the ADR's own rule): the four
attributes are removed from the narrowed set, six remain (three RAG observability, three usage
accounting). `contract_tools.freeze_check` reports all 30 reserved attributes emitted or narrowed,
never missing, backed by both a hermetic translation run (the regenerated
`contracts/trace/span_inventory.json`) and a real `docker compose --profile observability` capture
appended to `contracts/trace/freeze_evidence.json`.

## SP6 final review fix wave: observability deploy and trace correlation (2026-07-20)

The whole branch review before merge to local main found the SP6 kubernetes observability tier
undeployable from a fresh cluster: three independent blockers, each of the "declared and render
tested, never actually run from scratch" class. `task k3d:up` never synced phoenix/otel-collector/
atlas-monitoring (the sync list was never extended when SP6 task 5 added them); the phoenix
Deployment could not start at all (`command: ["sh", "-c"]` against an image with no shell); and
atlas-monitoring could not install on a cluster with no PrometheusRule CRD (helm rejects the whole
release on one unknown kind). All three are fixed: `k3d-up.sh` now syncs the three releases with an
explicit rollout wait each; phoenix assembles its database URL via Kubernetes' native dependent
environment variable expansion instead of a shell wrapper; and the one CRD type the PrometheusRule
custom resource needs is vendored into the chart's own `crds/` directory, installed once on a fresh
cluster, still with no controller behind it.

The review also found the envelope/log trace id and the exported span's real OTel trace id were two
disjoint worlds: neither artifact carried anything that could locate the other. The trace contract
bumps to **1.1.0** (additive, MINOR) for the fix: a new reserved attribute, `atlas.turn.seq`, is
stamped by `OtelTracer.open()` on every exported span, the process local sequence number the
response envelope and the JSON logs already carry for a turn's own root -- a documented join key,
proven in both directions by a live shaped hermetic test (find the span from the envelope id via
the attribute; find the envelope id from any span in the trace). The stream error path's JSON log
no longer names a `span_id` for the ttft stage span, which never closes -- and so never exports --
on that path; a log now only ever names a span it can prove actually left the process.
`OtelTracer._spans`/`_pending_stage` no longer retain every span for the life of the process: both
maps are bounded FIFO caches (oldest entries evicted at a fixed cap). A turn boundary sweep was
tried first and rejected by its own regression test, which proved it corrupts parent linkage when
turns interleave on the shared instance; the bounded cap has no turn ordering dependency.

## SP6: observability (2026-07-19)

A typed settings module (`AtlasSettings`, `config_hash`) became the one place every scattered env
read resolves through. An OTel backed tracer adapter translates the graph's informal span
vocabulary into the 29 reserved `atlas.*` attributes plus `gen_ai.*`, failing closed on any
uninventoried key; five stage duration attributes land through a minimal `close(seq)` protocol
extension, and trace id handoff closes the SP4 error correlation carry. A redacting OTel collector
fans out to Phoenix, the only deployed backend, with LangSmith and LangWatch kept as commented
exporter blocks proving pluggability. Structured JSON logs correlate by trace id and span id with
the redaction policy applied at the logger; log aggregation is refused by design. Prometheus rules
implement the deterministic paging set (probe failure, staleness gauge, breaker open, error rate)
behind one Alertmanager webhook receiver, and a sentinel probe CronJob drives the deployed service
over three query classes. Task 6 added the release identity surface: `/version`, the CHANGELOG
gate, and the CI pipeline that computes `helmfile.lock` and joins image digests into run manifests.

Task 7, the sub project's capstone, froze the trace contract at **1.0.0**. The emitter checklist
(`contract_tools.freeze_check`) walked all 29 reserved attributes against a hermetic translation run
and a live capture (`docker compose --profile observability`, the fastlane TEI node, one healthy and
one degraded turn, `contracts/trace/freeze_evidence.json`): 19 carry a real emitter (nine newly added
this task -- `atlas.semconv.version`, `atlas.variant`, `atlas.corpus.version`, `atlas.index.build_id`,
the three `atlas.contract.*_version` fields, `atlas.privacy.content_captured`, and
`atlas.privacy.redaction_policy_version`), and 10 are explicitly narrowed
(`ADR-029`, `contracts/trace/freeze_narrowed.yaml`):
three await a future RAG observability follow up (retrieval doc ids, pre/post rerank scores), three
await SP7's judge, three await a future usage accounting follow up (token counts, USD cost), and one
(the subject pseudonym) awaits SP7's HITL surface. The golden examples now show real, single
captured spans rather than the pre freeze aspirational composite. The diff engine's pre freeze
relaxation ("while MAJOR is 0, breaking requires at least a MINOR bump") is pinned to stop applying
now that trace's MAJOR is 1 (`test_pre_freeze_relaxation_stops_applying_once_major_is_1`), so the
ordinary SchemaVer floor governs every contract touching change from here on.

## SP5: infra (2026-07-19)

One helmfile spans two environments: `local`, a free k3d cluster, and `burst`, a credential gated
Hetzner tier built with OpenTofu and kube hetzner. CNPG Postgres runs a custom layered pgvector
image; TEI embed and rerank serving are pinned by digest. The backend and web Deployments sit
behind the k3s built in Traefik ingress with a rate limit middleware. Every image reference is
digest shaped in both compose and helm values. A weekly janitor workflow guards the Hetzner account
against orphaned resources once a burst tier exists, and the teardown script backs the wildcard
certificate up before destroying.

## SP4: agent core and reliability (2026-07-19)

LangGraph moved to the 1.2 line with a Postgres backed checkpointer: Alembic migrations plus an
async saver swapped in once the event loop is running. A circuit breaker and jittered retry ladder
wrap the TEI and Postgres calls, with a provider fallback rung for live and record mode. MCP tool
binding went live, the account, actions, and catalog dispatchers gained an isError fail closed
backstop, and the frozen SSE streaming contract gained a real producer. A fault injection lane
covers eight failure cases end to end.

## SP3: RAG spine (2026-07-19)

The corpus is chunked and embedded with BAAI bge m3, served by Hugging Face TEI, and indexed into
pgvector with hybrid tsvector plus RRF fused search and BAAI bge reranker v2 m3 reranking. The
flagship finding: the reranker demotes a correct answer chunk below the fused top five because a
competing chunk literally contains the phrase the correct chunk deliberately avoids, a designed
conflict measured live and pinned as a golden retrieval case. A compose acceptance check exercises
real retrieval and keyed generation together.

## SP2: registry and corpus (2026-07-19)

A fact registry of entities and edges, contradictions included, renders into a templated corpus of
customer facing and internal documents across nine plan variants. Corpus 0.1.1 ships forty
documents, built atomically (stage then rename) behind a data integrity gate that recomputes every
rendered fact from the registry, plus a pairwise distinctness gate across sibling documents.

## SP1: contracts v0.x (2026-07-18)

Four versioned JSON Schema families (trace, dataset, manifest, sse) with a SchemaVer diff engine
that classifies a schema change as patch, minor, or major and checks the required version bump. The
trace schema reserves the 29 `atlas.*` attributes this sub project's later work emits against; the
manifest schema reserves the twelve field lineage tuple. `task contracts:diff` compares any two
schemas, or a schema against a git ref.
