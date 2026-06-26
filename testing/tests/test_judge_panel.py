"""`judge.panel`, hermetic (SP8 task 2): the D15 jury mechanism, ties fail closed.

Absorbed verbatim from the pre rewrite `evals/judge/panel.py`; nothing in this repo calls
`panel_vote` in a headline benchmark context yet (SP9's benchmark matrix runner is the caller, the
plan's own boundary note). These tests exercise the mechanism alone.
"""
from __future__ import annotations

import dataclasses

import pytest

from judge.panel import PanelVote, panel_vote


def test_panel_reads_a_unanimous_pass():
    vote = panel_vote([1, 1, 1])
    assert vote.label == 1
    assert vote.disagreed is False
    assert vote.votes == (1, 1, 1)


def test_panel_reads_a_unanimous_fail():
    vote = panel_vote([0, 0])
    assert vote.label == 0
    assert vote.disagreed is False


def test_panel_majority_wins_and_flags_the_disagreement():
    split = panel_vote([1, 0, 1])
    assert split.label == 1 and split.disagreed is True  # majority pass, flagged for a human


def test_panel_tie_fails_closed_to_zero_and_flags_the_disagreement():
    tie = panel_vote([1, 0])
    assert tie.label == 0            # a split panel is not evidence the answer was good
    assert tie.disagreed is True     # and it is flagged, never silently resolved


def test_panel_four_way_tie_fails_closed():
    tie = panel_vote([1, 1, 0, 0])
    assert tie.label == 0
    assert tie.disagreed is True


def test_panel_needs_at_least_one_judge():
    with pytest.raises(ValueError):
        panel_vote([])


def test_panel_vote_is_immutable():
    vote = panel_vote([1, 1])
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(vote, "label", 0)


def test_panel_vote_is_a_frozen_dataclass_instance():
    vote = panel_vote([1])
    assert isinstance(vote, PanelVote)
