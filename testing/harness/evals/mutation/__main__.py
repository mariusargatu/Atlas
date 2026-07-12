"""`task mutation`: report the frozen semantic-mutant kill result (deterministic, always runs), then
hand off to an LLM to generate NEW realistic mutants of a metric and report survivors to promote.
Non-gating; the gate proof is ``test_mutation``. Prints guidance instead of crashing without an LLM.
"""
from __future__ import annotations

import importlib.util

from evals.mutation.mutants import IR_METRIC_MUTANTS, kills


def main() -> None:
    print("semantic mutation of the IR metrics (frozen registry — realistic human bugs):")
    for mutant in IR_METRIC_MUTANTS:
        status = "KILLED " if kills(mutant) else "SURVIVED"
        print(f"  [{status}] {mutant.name:30} — {mutant.realistic_bug}")
    killed = sum(kills(m) for m in IR_METRIC_MUTANTS)
    print(f"kill rate: {killed}/{len(IR_METRIC_MUTANTS)} realistic bugs caught by the Phase-1 suite")

    if importlib.util.find_spec("ollama") is None:
        print(
            "\nNo local LLM client; ran the frozen registry only. For LIVE semantic mutation:\n"
            "  pip/uv add an Ollama client, then this lane asks a model to introduce a realistic bug\n"
            "  into evals/retrieval/ir_metrics.py, runs each mutant against the Phase-1 witnesses, and\n"
            "  reports SURVIVORS. Every survivor is minimised and promoted into mutants.py as a\n"
            "  permanent regression — the same loop as red-team -> poison corpus."
        )
        return
    print(
        "\nOllama available: prompt it to mutate a target metric semantically, run the witnesses via\n"
        "evals.mutation.mutants.kills against the new fn, and promote any survivor into the registry."
    )


if __name__ == "__main__":
    main()
