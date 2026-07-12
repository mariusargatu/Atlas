"""`task metamorphic`: show the frozen paraphrase family and the invariant the gate enforces, then
hand off to an LLM to generate NEW paraphrases of the seed to be reviewed, frozen, and replayed. The
metamorphic relation (paraphrase invariance of the guard verdict) is what the gate checks
deterministically; this lane only proposes new members. Non-gating; prints guidance without an LLM.
"""
from __future__ import annotations

import importlib.util

from evals.datasets.metamorphic_golden import PARAPHRASE_FAMILY

SEED_QUESTION = PARAPHRASE_FAMILY[0][0]


def main() -> None:
    print(f"seed golden case: {SEED_QUESTION!r}")
    print("frozen paraphrase family (the derived set the gate replays):")
    for question, answer in PARAPHRASE_FAMILY:
        print(f"  Q {question}")
        print(f"    A {answer}")
    print(
        "\ninvariant (checked deterministically in test_metamorphic through the real atlas_graph):\n"
        "  the render guard holds every false 'no-contract' answer for a customer WITH a term, and\n"
        "  renders it for one WITHOUT — the oracle decides, not the wording."
    )
    if importlib.util.find_spec("ollama") is None:
        print(
            "\nNo local LLM client; showed the frozen family only. For LIVE augmentation:\n"
            "  add an Ollama client, then this lane asks a model to paraphrase the seed question and\n"
            "  its false claim, you review (is the transform truly meaning-preserving?), record a\n"
            "  cassette per new member, and freeze into metamorphic_golden.py."
        )
        return
    print("\nOllama available: generate paraphrases, hand-review the transforms, record + freeze.")


if __name__ == "__main__":
    main()
