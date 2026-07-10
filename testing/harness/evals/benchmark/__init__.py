"""The honest benchmark worked example: the regression that wasn't.

Two Atlas model versions over the answer golden set, run through the real interval and
paired test machinery in `evals.stats`. The statistics are computed live and reproduce
byte for byte from a stamped seed. The per item outcome vectors are a committed fixture
(`dataset.py`), the same way the judge calibration study pins a committed set of 14 cases.
"""
from __future__ import annotations

from evals.benchmark.dataset import N, SEED, paired_vectors
from evals.benchmark.study import RESAMPLES, render, run

__all__ = ["N", "RESAMPLES", "SEED", "paired_vectors", "render", "run"]
