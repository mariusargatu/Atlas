# Atlas: the system under test

A small, **runnable reference system** for the blog series *"Evals Are Checks, Not Tests."*
Atlas is an authenticated broadband support agent built on **LangGraph + MCP**: it answers help
questions from documents, reads the signed in customer's account, and makes changes to it, and it
is built so that every claim about it can be **tested** rather than asserted.

The point of the repo is the **harness**, not the agent. Clone it, run `task test`, and watch a
grounded but false answer get caught the moment it would have shipped, with **no API keys and no
network**.

```bash
task install   # uv sync
task test      # hermetic, deterministic, zero egress: the PR lane CI runs
```

Targets live in [`Taskfile.yml`](Taskfile.yml) ([Task](https://taskfile.dev)); `task --list` shows them all.

## The cold open

Two customers ask the same agent a question. Both answers are fluent and grounded in a real
retrieved document. One is right; one is wrong:

- **Sarah** asks to move to a faster plan → the agent confirms price + start date, she says yes,
  the change is applied behind a typed confirmation. Every fact correct.
- **Daniel** asks if his plan is contract free → the agent quotes the *current* plan page and says
  "yes, cancel any time." But Daniel is on last year's plan, which carries a 12 month term, a fact
  that lives in his **account**, not on the page. The answer is *grounded* and *false*.

`grounded ≠ true` is the hardest thing in the system to get right, and the harness is built to
prove whether the agent gets it right, deterministically, a thousand times, before a customer sees
it. (This mirrors *Moffatt v. Air Canada*, 2024 BCCRT 149.)

## Architecture

```
chat front door ──▶ LangGraph core ──▶ MCP servers          guard (fail closed)
 (identity from        agent decides     • knowledge  (RAG)   • before any action
  the session,         · reads           • account    (read)  • before any render
  never the model)     · acts            • catalog    (read)
                       · confirms         • actions    (write)
                                          ▲
        harness: gateway (record/replay) · faked backends · tracing · per customer cache
```

Each part is named by **how it fails** (mapped to the OWASP LLM Top 10 and Agentic Top 10), because
the failure is what you test. The full system map is the *"System Under Test"* chapter of the
series; the design chapters and ADRs live in the companion series workspace, not in this code repo.

### The four invariants

1. **Nothing external is real in CI.** Account / catalog / actions run against seeded in memory
   fakes. A thousand adversarial plan changes move no real money.
2. **The model is recorded and replayed.** One non deterministic node runs through a gateway
   (`record` / `replay` / `live`); replay needs no provider SDK and a cassette miss hard fails.
3. **Every turn is traced** from the first commit: you cannot grade a path you did not record.
   Tracing instruments the test lane; the shipped container deliberately runs the no-op tracer
   (see *Scope & status*).
4. **Identity comes from the session, never the model.** `customer_id` rides in the OAuth 2.1
   bearer token / MCP call context; it is never a tool argument the model can fill in.

## Layout

```
backend/atlas/   the agent (the product, hexagonal: domain · ports · adapters · orchestration · mcp_servers)
frontend/        the Vite SPA (typed client generated from the OpenAPI contract)
testing/         everything that proves the product works
  harness/       the test rig, grouped by role (read the folders in this order):
    determinism/   pin every non-reproducible source + the canonical digest everything is keyed by
    replay/        record the model once, replay forever (gateway · cassette · store · providers)
    tracing/       the span tree each turn emits — the thing the graders read
    evals/         grade the agent: stats · evalkit · drift · inference_oracle (backend must NOT import this)
    recording/     operator scripts that capture new cassettes against a live model (need keys)
    cassettes/     committed recorded model responses the lanes replay (data, not code)
  tests/         the suite that gates every merge
```

> The series' design docs and ADRs (system map, principles, test architecture, oracles) live in the
> companion workspace, kept outside this code repo on purpose.

## Testing

The hermetic PR lane (`task test`) is deterministic and byte stable across x86_64 and arm64
(see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)). It uses:

- **Rule table tests**: each parametrized row is a spec (guard, intent binding, value bounds).
- **Property + metamorphic tests** for the pure cores (Wilson interval, Cohen's κ, the canonical
  cassette key digest).
- **Coverage** is risk scoped (`task cov`, ~97%); the omit list is an explicit "do not test" choice.
- **Mutation tested** (mutmut) on the domain logic: behavior bearing mutants are killed; survivors
  are equivalent (error message strings, frozen dataclass flags, opaque ids).

## Scope & status

This is a **reference system for a testing series**, not a deployable product:

- Account state is an **in memory store** (seeded, write through within a run), the correct
  substrate for the hermetic lane. A real `PostgresAccountStore` behind the existing account port
  is deliberately **deferred** (dev/prod only); `task up` needs no external infra.
- The intent classifier is a deterministic keyword heuristic (no model call, so the lane stays
  reproducible); a production system would classify all intents or let the model propose intent
  under review.
- The render-time truth guard (`guard.check_render_truth`) is likewise a cue-based heuristic: it
  catches the demonstrated "contract-free / cancel any time" phrasings against the account oracle,
  not an arbitrary paraphrase or a wrong fee amount. Grading a numeric claim needs structured-claim
  extraction, deferred to the metrics article.
- Auth is first party and demo grade (local signing key, password less sign in as a seeded
  customer); step up authorization is modelled but not gated on real re auth/MFA.
- The deployed container is untraced (`server.py` wires the no-op tracer) and the **online lane**
  — production trace capture, sampled online judge scoring, model assisted triage — is
  deliberately deferred until real infrastructure exists. The deterministic halves that *can*
  gate are built and gated: call budgets (enforced in the running graph and asserted over
  recorded trajectories), the review queue sampler, and the PII scrub → promote loop that turns a
  production failure into a golden case (`testing/harness/evals/monitor/`).

These simplifications are intentional and documented inline where they matter. Operational
procedures (cassette re-recording, judge recalibration, promoting a production failure) live in
[`docs/runbooks/`](docs/runbooks/).

## License

See [`LICENSE`](LICENSE).
