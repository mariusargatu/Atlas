"""`task simulation`: the persona-driven simulation lane. The deterministic core (always runs) shows
the roster and proves the multi-turn 'confirm the settled intent' check discriminates the cold-open
failure from the correct one. The live loop is the three-agent harness: a persona model drives the
agent, a separate calibrated evaluator (the judge lane) scores, each scenario runs many trials and
reports a pass rate with a confidence interval, and conversations that expose a failure are frozen as
new fixtures (ADR-019). Non-gating; the gate proof is ``test_simulation``. Prints guidance without an LLM.
"""
from __future__ import annotations

import importlib.util

from evals.datasets.simulation_golden import MIND_CHANGER
from evals.simulation.grade import grade_conversation
from evals.simulation.model import ConversationOutcome
from evals.simulation.personas import PERSONAS


def main() -> None:
    print("persona roster (the simulator plays these against the agent):")
    for persona in PERSONAS:
        print(f"  {persona.name:20}  {persona.disposition}")

    print(f"\ncold-open scenario: the {MIND_CHANGER.persona}, {len(MIND_CHANGER.turns)} turns, "
          f"settled intent = {MIND_CHANGER.settled_plan_id}")
    settled = ConversationOutcome(actions=(("change_plan", {"plan_id": MIND_CHANGER.settled_plan_id}),))
    walked = ConversationOutcome(actions=(("change_plan", {"plan_id": MIND_CHANGER.walked_back_plan_id}),))
    print(f"  confirms the settled plan -> sound={grade_conversation(settled, MIND_CHANGER).sound}")
    print(f"  confirms the walked-back  -> sound={grade_conversation(walked, MIND_CHANGER).sound}  "
          "(the cold-open failure, caught between turns)")

    print("\nThe gate (test_simulation) drives this conversation through the real atlas_graph and asserts")
    print("the agent confirms the plan she landed on. This lane generates new conversations live:")
    if importlib.util.find_spec("ollama") is None:
        print(
            "  no Ollama; ran the deterministic grade only. For live generation:\n"
            "  a persona model drives the agent, a separate calibrated evaluator (the judge lane) scores\n"
            "  (never the agent grading its own conversation), each scenario runs many trials and reports\n"
            "  a pass rate with a Wilson interval (quality.stats), and every conversation that exposes a\n"
            "  failure is recorded and frozen into simulation_golden.py as a permanent regression."
        )
        return
    print("  Ollama available: play each persona, score with the calibrated evaluator over many trials,")
    print("  watch the variance, and freeze the failures into fixtures.")


if __name__ == "__main__":
    main()
