"""Drift detection: catches a model that moved while its cassette still replays green.

Replay pins the model to a cassette and never re-checks it, so a live model change behind an
identical request goes unnoticed. This lane re-runs the pinned agent against a new snapshot and
diffs the decisions (intent, tools, guards, outcome), read from the trace and never from the
prose, against the committed cassette.

Tested hermetically by mutating a cassette. The live shadow re-record against the provider needs
keys and the `record` dependency group and is deferred, like the eval live lane.
"""
from __future__ import annotations

from evals.drift.compare import DriftReport, compare
from evals.drift.record import DecisionRecord, extract

__all__ = ["DecisionRecord", "DriftReport", "compare", "extract"]
