# Atlas: guide for an AI reviewer

This repo is the **runnable reference system** for the blog series *"Evals Are Checks, Not Tests."*
It is a LangGraph + MCP broadband support agent ("Atlas") built so its behavior can be **tested
deterministically**. The harness is the point; the agent is the specimen.

## Read first (in this order)

1. [`README.md`](README.md): what it is, quickstart, the four invariants.
2. The system map, the *"System Under Test"* chapter of the series; every part named by how it
   fails (OWASP LLM + Agentic Top 10). The design chapters and ADRs live in the companion series
   workspace, not in this code repo.
3. The code itself: start at `backend/atlas/orchestration/atlas_graph.py` (the runtime the map
   draws) and `testing/harness/` (the determinism machinery). ADRs are referenced inline as `ADR-0xx`.

## How to run

```bash
task install   # uv sync
task test      # hermetic, deterministic, NO keys, NO network: the PR lane
task cov       # same suite + coverage (risk scoped omit list in pyproject)
```

(Targets are defined in `Taskfile.yml`; run `task --list` to see them all.)

The PR lane never makes a live model call: the gateway runs in `replay` and the `record`
dependency group is not installed, so a cassette miss hard fails instead of going to the network.

## Conventions that matter when reviewing

- **Hexagonal layers are enforced by a test.** `backend/atlas/domain` and `ports` are pure: no
  framework/client imports, no dependency on outer rings (`testing/tests/test_import_lint.py`). Don't
  suggest importing LangGraph/MCP into the domain.
- **Determinism is a contract.** No wall clock, no `random`, no unordered iteration in runtime
  paths; time/ids come from injected factories (`testing/harness/determinism/sources.py`). The cassette key
  is a canonical digest (`testing/harness/determinism/canonical.py`). Changing canonicalization invalidates
  cassettes.
- **Identity never comes from the model.** `customer_id` is from the session/bearer token, never a
  tool argument. Flag any change that puts it in a tool schema.
- **The guard is fail closed**; binding intercepts an unauthorized tool and fails closed before it
  runs (a dev/prod build also withholds it from the model, so the capability is absent).
- **Immutability**: domain objects are frozen dataclasses updated via `replace`; the single mutable
  boundary is the in memory account store (`domain/accounts.py:_STATE`).

## Known, intentional simplifications (not bugs)

- Account persistence is **in memory**, write through within a run; a real Postgres adapter behind
  the account port is deliberately deferred (dev/prod only).
- Intent classification is a deterministic keyword heuristic (keeps the lane reproducible).
- Auth is first party/demo grade (local signing key, password less sign in); step up is modelled,
  not gated on real re auth.

These are documented inline where they occur. A good review distinguishes *"this is wrong"* from
*"this is a scoped simplification of a reference system."*

## Where to be skeptical

The interesting bugs in agentic systems live on the **write surface** and the **guard/cache/trace
wiring**, places where a component can be *declared and unit tested but not actually wired into the
running graph*. When reviewing `backend/atlas/orchestration/atlas_graph.py`, check that what the
docs claim is enforced is actually enforced at runtime, not just available as a function.
