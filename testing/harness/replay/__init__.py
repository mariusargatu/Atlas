"""Replay: record the model once, replay it forever (the determinism seam, ADR-007).

The agent has exactly one nondeterministic step, the call to the model. The gateway
wraps it and runs in one of three modes: REPLAY (cassette only, zero egress, a miss is
a hard fail, the PR lane), RECORD (call live and persist), or LIVE (call live, persist
nothing, the eval lane).

  gateway.py         The seam: a LangChain chat model the graph already calls, so
                     nothing upstream knows it is being recorded.
  cassette.py        The typed, content addressed on disk shape of one recorded call.
  cassette_store.py  Where cassettes live (file store for CI, in memory for tests),
                     behind a small port. Policy free: a miss returns None, and the gateway
                     decides that, in REPLAY, None is a hard failure.
  providers.py       Builds the live provider model (Ollama / Anthropic / OpenAI) that
                     RECORD and LIVE wrap. REPLAY needs none of these.

The committed recordings themselves live in ../cassettes/ (data, not code).
"""
