"""The judge panel: more than one judge, and the disagreement is the prize.

A panel of judges from disjoint model families (Verga et al., "Replacing Judges with Juries", 2024)
is steadier than any single grader and carries less same-family bias. But the aggregate is not the
point. The cases where the panel SPLITS are, almost by definition, the ambiguous, high-information
ones, and they are exactly the cases worth a human's scarce attention. A panel does not just vote;
it tells you where to look.

This is deliberately tiny: a majority vote plus a disagreement flag. The routing it enables (split
cases go to a human, agreed cases are trusted) is the policy; the heavier panel mechanics and the
GLMM model-by-judge matrix that separates "the system regressed" from "this judge is simply
different" belong to the statistics article.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PanelVote:
    """The panel's reading of one case: the majority label, and whether the judges split."""

    label: int          # the majority verdict (1 PASS / 0 FAIL); ties fail closed to 0
    disagreed: bool     # did the judges split? if so, route this case to a human
    votes: tuple[int, ...]


def panel_vote(labels: list[int]) -> PanelVote:
    """Aggregate a panel's per-judge labels. Ties fail closed (label 0), because a panel that cannot
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
