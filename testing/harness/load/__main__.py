"""Operator entrypoint: `uv run python -m load --iterations <path> --spans <path>` (`task
load:join`).

Reads a k6 NDJSON iteration capture (`k6 run chat_sse_load.js | tee run.log`, the console lines
`k6/chat_sse_load.js` emits with a LOAD_ITER prefix) and a Phoenix span export (a local JSON file,
already in `phoenix_join.SpanRecord`'s own shape; turning a REAL live Phoenix export into that
shape is the documented, live burst wiring step this task does not build; see `phoenix_join.py`'s
own module docstring), joins them, and prints the latency summary (per concurrency step, per
stage) plus any join misses, never silently dropped.

Not part of the hermetic gate (`pyproject.toml`'s coverage omit list names this file the same
"operator entrypoint, not gated" way it already names `labeling/__main__.py`); the functions it
calls (`load_iteration_records`, `load_span_export`, `join_iterations_to_spans`,
`summarize_by_concurrency`) ARE gated, by `testing/tests/test_load_phoenix_join.py`.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from load.phoenix_join import (
    join_iterations_to_spans,
    load_iteration_records,
    load_span_export,
    summarize_by_concurrency,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=Path, required=True, help="k6 stdout capture (LOAD_ITER lines)")
    parser.add_argument(
        "--spans", type=Path, required=True,
        help="a Phoenix span export, already in phoenix_join.SpanRecord's own JSON shape",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    iterations = load_iteration_records(args.iterations)
    spans = load_span_export(args.spans)
    result = join_iterations_to_spans(iterations, spans)
    summary = summarize_by_concurrency(result)
    report = {
        "per_stage_latency_ms": summary,
        "join_misses": [asdict(miss) for miss in result.misses],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
