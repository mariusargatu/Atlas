"""`task metamorphic`: print the deterministic metamorphic report (always runs, hermetic, replays
the frozen families against the stub retrieval fixture), then hand off to an LLM to propose NEW
paraphrase/typo/perturbation members to review, freeze, and replay. This lane never freezes a
member on its own; reviewing and committing a proposal into `families.py` is a manual, human
curated step, exactly `evals/metamorphic/__main__.py`'s own precedent. Non-gating: this file is on
`pyproject.toml`'s coverage omit list, the same "operator entrypoint, not gated" treatment as
`judge/live_provisional.py` and `labeling/__main__.py`; the functions it calls (`report.
run_all_families`, every family and invariant function) ARE gated, by `testing/tests/
test_metamorphic.py`.

The live/full lane against the real pgvector/TEI stack (`task test:live`, covering
`testing/tests/test_metamorphic_live.py`) is a SEPARATE operator step, documented in its own
Taskfile entry and in this sub project's report; it is deferred, like SP7's own live measurements,
and is not invoked from here.
"""
from __future__ import annotations

import importlib.util

from metamorphic.report import run_all_families


def main() -> None:
    report = run_all_families()
    print(report.render())
    if not report.all_hold:
        print("one or more families failed an invariant -- see above")

    if importlib.util.find_spec("ollama") is None:
        print(
            "No local LLM client; ran the frozen families only. For a LIVE proposal round:\n"
            "  uv sync --extra ollama, then this lane asks a local model to propose NEW natural\n"
            "  paraphrasings, typos, or surface perturbations of the same conflict-daniel-contract\n"
            "  question. Every proposal is reviewed by a human curator and, if accepted, hand added\n"
            "  to families.py as a new member -- never auto frozen, never auto gated."
        )
        return
    print(
        "\nOllama available: prompt it to propose new family members for review; accepted proposals\n"
        "are hand added to families.py (a new family_id if the relation itself changes, a new member\n"
        "otherwise), never auto committed."
    )


if __name__ == "__main__":
    main()
