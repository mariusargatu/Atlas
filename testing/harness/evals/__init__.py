"""Evals: grade the agent (the second of the two machines).

The regression lane asks "did a pinned behaviour change?" (binary, gating). This half
asks "how good is the agent, and is it getting better or worse?" It drives the agent
over seeded cases and reports a RATE, never a single verdict, because the live model is
stochastic and one sample lies.

  stats.py           Honest numbers: intervals (Wilson/Wald/bootstrap/BCa/cluster), paired
                     tests, power sizing, variance split, judge<->human agreement.
                     A score without an interval is an anecdote.
  gate.py            Release gating on the interval's lower bound, never the point
                     (variance budget + quarantine, fails closed).
  evalkit/           The scored lane: planner (designs cases) / generator (the agent) /
                     evaluator (the grader stack), run over many trials.
  drift/             Catches the model moving behind a stable version string by diffing
                     DECISIONS (not wording) against a committed baseline.
  inference_oracle/  Grades answers whose truth was never stored, by deriving the truth
                     in code and comparing it to the model's claim.

The product (backend/) must never import this package, the two harness seam, enforced
by testing/tests/test_import_lint.py.
"""
