"""Atlas test harness — the rig that makes a nondeterministic LLM agent testable.

The agent ("Atlas") is the specimen; THIS is the product. It turns a stochastic
model into something you can run a thousand times and get the same bytes, then
grade. Read the folders in this order — they tell the story:

  determinism/  Pin every source of non-reproducibility (clock, RNG, ids, span
                order) plus the canonical digest everything is keyed by. Nothing
                downstream is reproducible without this.
  replay/       The one nondeterministic step is the model call. Record it once,
                replay it forever from a content-addressed cassette — so CI needs
                no API key and no network, and a miss is a hard failure.
  tracing/      Every turn emits a span tree (the model call, each tool, each
                guard). This is the thing the graders read; you cannot grade a
                path you did not record.
  evals/        The OTHER machine: grade the agent. Drives seeded cases, scores
                each run, reports a RATE not a single verdict. The product code
                (backend/) must NEVER import this — enforced by test_import_lint.
  recording/    Operator scripts that capture new cassettes against a live model.
                They need API keys, so they never run in the PR lane.
  cassettes/    Committed recorded model responses the lanes replay (data, not code).

Two machines, one seam: the rig (determinism + replay + tracing) makes the agent
reproducible; evals/ measures it. The dependency points one way — evals reads the
agent through its ports; the agent never imports the evals.
"""
