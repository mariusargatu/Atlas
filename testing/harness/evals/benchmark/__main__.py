"""Runnable honest benchmark study on the hermetic lane: zero keys, zero egress (`task benchmark`).

The regression that wasn't, as a committed artifact. Two Atlas model versions over the answer
golden set, v_a 84/100 and v_b 81/100, a three point gap a teammate read as a regression. The
study puts a Wilson interval on each score (they overlap), bootstraps the paired per item
difference (its 95% CI straddles zero), and runs the paired tests (permutation + exact McNemar)
that belong on paired data. The verdict comes back: the gap sits inside the noise.

Every number is computed live from `quality.stats`. The only fixed inputs are the seed and the
committed paired outcome set in `dataset.py`. The seed is stamped into the artifact so the
interval reproduces byte for byte, which is what makes a benchmark a measurement and not weather.

Declaring a small delta real is the most common eval lie. Declaring this one not real, in a
committed artifact a reviewer can rerun, is the honesty signal.
"""
from __future__ import annotations

import json
from pathlib import Path

from evals.artifacts import write_artifacts
from evals.benchmark.study import render, run

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MD = ARTIFACT_DIR / "benchmark_study.md"
JSON = ARTIFACT_DIR / "benchmark_study.json"


def main() -> None:
    r = run()
    study = render(r)
    write_artifacts([(MD, study), (JSON, json.dumps(r, indent=2) + "\n")], echo=study)


if __name__ == "__main__":
    main()
