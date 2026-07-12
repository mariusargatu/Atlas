"""Semantic mutation testing: mutate the metric like a human would get it wrong, not like an
operator flip.

Classical mutation testing (mutmut) flips syntax: `>` to `>=`, `+` to `-`. Cheap, but many are
equivalent mutants and none resemble a *real* bug. A semantic mutant is a plausible human mistake, a
recall that divides by k, a reciprocal rank that takes the last hit, an NDCG on exponential gain. The
signal is the SURVIVORS: a realistic mutant no test kills is a bug the suite would ship.

- ``mutants`` holds a frozen registry of realistic IR-metric mutants, each with a WITNESS input that
  a Phase-1 test actually asserts on, so "killed" means "the real suite kills it" (``test_mutation``).
  This is the deterministic, gate-safe core.
- ``__main__`` (operator lane) re-reports the frozen kill result and, if an LLM is available, generates
  NEW semantic mutants of a source function and reports which survive, exactly the red-team loop:
  every survivor is minimised and promoted into the frozen registry as a permanent regression.
"""
