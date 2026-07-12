"""`task redteam`: the read-surface red-team lane. The always-runnable core runs the breach detector
over every committed poison document (not one hard-coded trajectory) and prints an OWASP-tagged
coverage summary. If the ``redteam`` group is installed, it additionally hands off to DeepTeam for
LIVE attack generation against a target, whose landed attacks are then promoted back into the poison
corpus. Non-gating; never the PR lane.
"""
from __future__ import annotations

import importlib.util

from atlas.domain.poison_corpus import POISON_CORPUS, canary_tokens, forbidden_tools
from evals.retrieval.injection import Trajectory, detect_breach


def _coverage_report() -> None:
    print("read-surface poison corpus (single-sourced, gate + red-team):")
    for doc in POISON_CORPUS:
        tool = f" -> {doc.forbidden_tool}" if doc.forbidden_tool else ""
        print(f"  [{doc.owasp_id}] {doc.id:24} {doc.injection_type}{tool}")

    # per document: a defended answer (its own expected_behavior) must not breach, and an answer
    # that obeys the injection (echoes the canary, calls the forbidden tool) must be caught
    tokens, tools = canary_tokens(), forbidden_tools()
    defended_breaches = compromised_detected = 0
    for doc in POISON_CORPUS:
        defended = Trajectory(answer=doc.expected_behavior, tools_called=())
        if detect_breach(defended, canary_tokens=tokens, forbidden_tools=tools).breached:
            defended_breaches += 1
        obeyed = (doc.forbidden_tool,) if doc.forbidden_tool else ()
        compromised = Trajectory(answer=f"Okay, {doc.canary_token}", tools_called=obeyed)
        if detect_breach(compromised, canary_tokens=tokens, forbidden_tools=tools).breached:
            compromised_detected += 1

    print(f"\ndocs_checked={len(POISON_CORPUS)} defended_breaches={defended_breaches} "
          f"compromised_detected={compromised_detected}")
    print("gating invariant ('retrieved content is data, never commands') lives in "
          "testing/tests/test_injection.py; this is a smoke re-check, not the gate.")


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
