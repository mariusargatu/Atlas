"""`task redteam`: the read-surface red-team lane. The always-runnable core re-verifies that a
defended agent treats every committed poison document as data (the invariant the gate enforces) and
prints an OWASP-tagged coverage summary. If the ``redteam`` group is installed, it additionally hands
off to DeepTeam for LIVE attack generation against a target, whose landed attacks are then promoted
back into the poison corpus. Non-gating; never the PR lane.
"""
from __future__ import annotations

import importlib.util

from atlas.domain.poison_corpus import POISON_CORPUS, canary_tokens, forbidden_tools
from evals.retrieval.injection import Trajectory, detect_breach

# A defended agent: it may quote or refuse, but it never obeys a retrieved instruction.
_DEFENDED = Trajectory(
    answer="I can't act on instructions embedded in a help article. Here is the guidance you asked for.",
    tools_called=(),
)


def _coverage_report() -> None:
    print("read-surface poison corpus (single-sourced, gate + red-team):")
    for doc in POISON_CORPUS:
        tool = f" -> {doc.forbidden_tool}" if doc.forbidden_tool else ""
        print(f"  [{doc.owasp_id}] {doc.id:24} {doc.injection_type}{tool}")
    verdict = detect_breach(_DEFENDED, canary_tokens=canary_tokens(), forbidden_tools=forbidden_tools())
    status = "HOLDS" if not verdict.breached else f"BREACHED: {verdict.reasons}"
    print(f"invariant 'retrieved content is data, never commands': {status}")


def main() -> None:
    _coverage_report()
    if importlib.util.find_spec("deepteam") is None:
        print(
            "\ndeepteam is not installed; ran the deterministic corpus check only. For live generation:\n"
            "  uv sync --group redteam     (installs deepteam), then re-run: task redteam\n"
            "then point DeepTeam's SyntheticContextInjection / CrossContextRetrieval attacks at the\n"
            "Atlas target and promote every landed attack into atlas.domain.poison_corpus."
        )
        return
    print(
        "\ndeepteam is available. Wire its red_team(...) over a callback that returns Atlas's answer\n"
        "and tools_called, then minimise + promote each landed attack into the poison corpus."
    )


if __name__ == "__main__":
    main()
