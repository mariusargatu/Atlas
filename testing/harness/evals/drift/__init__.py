"""The drift lane: the fourth gateway reading (a spike).

REPLAY pins the model to a cassette and never re-checks it. RECORD captures once. LIVE measures
quality. None of them notices when the live model has silently moved while the request bytes stay
identical, so replay returns last quarter's response forever and the suite stays green on a stale
proxy. The drift lane re-runs the pinned agent against a new model snapshot and diffs the DECISIONS
(intent, tools, guards, outcome) against the committed cassette, separating behavioural drift from
mere prose drift.

This package is the comparison core, tested hermetically by mutating a cassette. The live shadow
re-record (RECORD against the provider on a cadence) is the real trigger and is deferred, exactly
like the eval LIVE lane: it needs keys and the `record` dependency group.
"""
from __future__ import annotations

from evals.drift.compare import DriftReport, compare
from evals.drift.record import DecisionRecord, extract

__all__ = ["DecisionRecord", "DriftReport", "compare", "extract"]
