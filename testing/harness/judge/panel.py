"""The judge panel: more than one judge, where the disagreement is the useful signal.

A panel of judges from disjoint model families (Verga et al., "Replacing Judges with Juries", 2024)
is steadier than any single grader and carries less same family bias. The aggregate is not the point;
the cases where the panel splits are the ambiguous, high information ones worth a human's attention.

Deliberately tiny: a majority vote plus a disagreement flag. Split cases route to a human, agreed
cases are trusted; the heavier panel mechanics (a model by judge matrix separating "the system
regressed" from "this judge reads differently") are out of scope here.

Absorbed verbatim from the pre rewrite `evals/judge/panel.py` (SP8 task 2, per the planning digest's
own disposition: "keep verbatim... matches D15's 3 model cross provider jury"). D15's headline jury
context (a 3 model cross provider vote in a headline benchmark burst) is SP9's benchmark matrix
runner: this module is the mechanism, SP9's matrix runner is the caller that invokes `panel_vote` in
that context. `matrix.generators.run_generation_cell` (SP9 task 4, `3e79f25`) IS that caller, for
real: every (retrieval config, generator) cell's answer is judged by the full panel there, proven by
tamper (`test_matrix_runner.py::test_panel_vote_ran_in_stage_3_disagreement_and_labels_present` and
the whole branch review's own independent invocation count check).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PanelVote:
    """The panel's reading of one case: the majority label, and whether the judges split."""

    label: int          # the majority verdict (1 PASS / 0 FAIL). Ties fail closed to 0
    disagreed: bool     # did the judges split? if so, route this case to a human
    votes: tuple[int, ...]


def panel_vote(labels: list[int]) -> PanelVote:
    """Aggregate a panel's per judge labels. Ties fail closed (label 0), because a panel that cannot
    agree an answer is good is not evidence it was. ``disagreed`` is the signal that matters: it is
    the flag that sends a borderline case to the human who can actually adjudicate it."""
    if not labels:
        raise ValueError("a panel needs at least one judge")
    passes = sum(labels)
    fails = len(labels) - passes
    label = 1 if passes > fails else 0
    disagreed = passes > 0 and fails > 0
    return PanelVote(label=label, disagreed=disagreed, votes=tuple(labels))


__all__ = ["PanelVote", "panel_vote"]
