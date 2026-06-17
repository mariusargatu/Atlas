# Atlas contracts

The dual plane join: four versioned JSON Schema families. Data only; the code lives in
`testing/harness/contract_tools/` and the tests in `testing/tests/test_contract_*.py`.

| Family   | Governs                                            |
|----------|----------------------------------------------------|
| trace    | one exported span: atlas.* attributes and events   |
| dataset  | one golden case (turns array, slicing metadata)    |
| manifest | one run manifest (12 field lineage tuple)          |
| sse      | the /chat streaming event vocabulary               |

## Versioning (SchemaVer semantics, HLD D25)

MAJOR.MINOR.PATCH in `x-contract-version`. MAJOR means historical artifacts are no longer
comparable and forces a trend re baseline. Evolution rules, enforced by the diff engine
(`contract_tools/diff.py`):

| Change                          | Required bump |
|---------------------------------|---------------|
| remove a property               | MAJOR         |
| change a property type          | MAJOR         |
| add to required                 | MAJOR         |
| narrow an enum                  | MAJOR         |
| add an optional property        | MINOR         |
| widen an enum                   | MINOR         |
| metadata and description only   | PATCH         |

While MAJOR is 0 (pre freeze), breaking changes require at least a MINOR bump; this relaxation
stops applying once MAJOR reaches 1 and the ordinary floor above governs every later change
(`test_pre_freeze_relaxation_stops_applying_once_major_is_1`, `testing/tests/test_contract_diff.py`).

The trace family froze at **1.0.0** in sub project 6 task 7: every one of the 29 reserved
`atlas.*` attributes (`contract_tools.loader.RESERVED_TRACE_ATTRIBUTES`) either has a real emitter
(evidenced by both a hermetic translation run and a live capture,
`contract_tools.freeze_check`) or is explicitly narrowed with a named owner
(`ADR-029`,
`contracts/trace/freeze_narrowed.yaml`) -- never a silent gap. Emitting a narrowed attribute for
real later is a MINOR bump, per that ADR's own rule.

`RESERVED_TRACE_ATTRIBUTES` grew to 30 names post freeze, at **1.1.0** (the SP6 final branch
review, I1 fix): `atlas.turn.seq`, the join key between the response envelope/log trace id and the
real exported span it names (`OtelTracer.open()` stamps it on every span,
`backend/atlas/adapters/otel_tracer.py`), evidenced the SAME way, emitted from the moment it was
reserved -- never narrowed. An additive property with a real emitter is exactly the MINOR bump this
table already governs, the freeze's own post 1.0.0 discipline working as designed, not an exception
to it.

Four of the ten narrowed attributes closed at **1.2.0** (SP8 task 1): `atlas.judge.id`,
`atlas.judge.verdict`, `atlas.judge.rubric_version` (the groundedness judge, `testing/harness/
judge/`) and `atlas.subject.pseudonym` (an HMAC of `customer_id`, threaded through `atlas_graph.py`'s
"turn" call site). Same evidence discipline, same MINOR bump rule
(`ADR-029`'s own Amendment).

The remaining usage accounting trio closed at **1.3.0** (SP9 task 5): `atlas.cost.input_tokens`,
`atlas.cost.output_tokens`, `atlas.cost.usd`, emitted by `testing/harness/matrix/cost_emission.py`'s
`emit_cost` (a NEW `kind="llm"` span, `generator_cost`, never wired into the live graph -- the same
D29 batch, report time disposition the judge's own emitter already established), backed by a
backward compatible `replay/cassette.py` change (a RECORD mode call now also persists the
provider's own `usage_metadata` when it returned one; an OLD cassette with no `usage_metadata` key
still replays green, cost reported as unavailable) and `testing/harness/matrix/spend_gate.py`'s own
small pricing table plus hard per provider spend ceilings. Three remain narrowed, all RAG
observability fields (`atlas.retrieval.doc_ids`, `atlas.rerank.scores_pre`,
`atlas.rerank.scores_post`).

The freeze's own scope is attribute names only, never event shapes. `schema.json`'s `events` array
(the `user_feedback` shape `contracts/trace/examples/chat_turn.json` illustrates) is contract
modeled but has no producer anywhere in this codebase yet; no call site emits one today, and the 29
attribute checklist the v1.0.0 freeze walked never walked events at all (`ADR-029`'s own Consequences section says so explicitly).

Diff two schema files: `uv run python -m contract_tools.diff OLD.json NEW.json`
Diff against a git ref: `uv run python -m contract_tools.diff --git-ref main contracts/trace/schema.json`

## `mcp_snapshots/` (D11, a DIFFERENT mechanism)

`contracts/mcp_snapshots/<server>.json` (account, actions, catalog, knowledge) are golden byte
diffs of what each MCP server currently advertises, not a versioned, evolvable schema family: no
`x-contract-version`, no SchemaVer bump table, no diff engine. `testing/tests/test_mcp_snapshots.py`
asserts exact byte equality against a fresh dump; regenerate intentional changes with
`uv run python -m contract_tools.mcp_snapshot --write` (see that file's own header) and review the
resulting diff in the same change.

Each snapshot captures a tool's INPUT schema only (`parameters`, the `inputSchema` a client sees
before calling); a tool's OUTPUT shape (the JSON its result actually carries, e.g. knowledge
passages gaining `chunk_id`/`score`) is not advertised as a schema anywhere today and so does not
surface as a diff here.
