"""`task ragset`: generate synthetic goldens from the corpus with DeepEval's Synthesizer, freeze to a
versioned JSONL artifact for human review. Non-gating; needs the ``rageval`` group + an Ollama daemon.
Prints how to run rather than crashing if deepeval is absent.
"""
from __future__ import annotations

import json
from pathlib import Path

from atlas.domain.corpus import CORPUS

ARTIFACT = Path(__file__).parent / "artifacts" / "synthetic_goldens.jsonl"


def main() -> None:
    try:
        from deepeval.synthesizer import Synthesizer
    except ImportError:
        print(
            "deepeval is not installed (operator lane). Run:\n"
            "  task ragset\n"
            "or:  uv run --group rageval python -m evals.ragset\n"
            "and start an Ollama daemon for local generation."
        )
        return

    from evals.rageval.judge import build_ollama_judge

    synthesizer = Synthesizer(model=build_ollama_judge())
    # one context per corpus chunk; reweight toward multi-hop shapes by hand-reviewing the output.
    contexts = [[chunk.text] for chunk in CORPUS]
    goldens = synthesizer.generate_goldens_from_contexts(contexts=contexts)

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT.open("w") as fh:
        for golden in goldens:
            row = {"input": golden.input, "expected_output": getattr(golden, "expected_output", None),
                   "context": getattr(golden, "context", None)}
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {len(goldens)} candidate goldens to {ARTIFACT}")
    print("Review and prune by hand before committing: synthetic is a starting point, the frozen")
    print("human-reviewed set is the fixture the gate replays.")


if __name__ == "__main__":
    main()
