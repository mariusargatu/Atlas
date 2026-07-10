"""Thresholds as code (SP9 task 6, D31): one small, committed JSON file
(`thresholds.json`, next to this module) is the single source of truth for the load lane's own
pass/fail gates -- `k6/chat_sse_load.js` reads the SAME file via `open()` to build its own
`options.thresholds`, and this module parses it for the hermetic tests and any report run after
the fact. Two independent readers of one file, never two independently maintained literal tables
that could silently drift apart on the same metric's own number.

`MetricThreshold.as_k6_expr()` renders the EXACT string k6's own threshold DSL expects (e.g.
`"p(95)<2000"`, `"rate>0.95"`) -- a plain aggregation stat plus a comparison operator plus a value,
never interpreted beyond syntactic validation here (this module never talks to a k6 process; it
only proves the JSON parses into something both sides can use).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.json"

# k6's own threshold comparison operators (its docs' own grammar); no others parse there either.
_VALID_OPS = ("<=", ">=", "<", ">")
# k6's own recognized aggregation stats for a Trend (p(N[.N]), avg/min/max/med) or a Rate ("rate");
# "count" covers a Counter metric, included for completeness even though this lane defines none.
_STAT_PATTERN = re.compile(r"^(p\(\d{1,3}(\.\d+)?\)|avg|min|max|med|rate|count)$")


@dataclass(frozen=True)
class MetricThreshold:
    """One metric's own gate: `f"{stat}{op}{value}"` is k6's literal threshold expression syntax."""

    metric: str
    stat: str
    op: str
    value: float

    def as_k6_expr(self) -> str:
        rendered_value = str(int(self.value)) if self.value == int(self.value) else str(self.value)
        return f"{self.stat}{self.op}{rendered_value}"


def parse_threshold(metric: str, spec: Mapping[str, object]) -> MetricThreshold:
    """Validate one `thresholds.json` entry. Fails closed on anything k6's own threshold DSL could
    not parse -- a typo here should be caught by `task test`, not discovered mid live burst run."""
    if not metric:
        raise ValueError("a threshold's metric name cannot be empty")
    stat = spec.get("stat")
    op = spec.get("op")
    value = spec.get("value")
    if not isinstance(stat, str) or not _STAT_PATTERN.match(stat):
        raise ValueError(f"{metric!r}: unrecognized k6 stat aggregation {stat!r}")
    if op not in _VALID_OPS:
        raise ValueError(f"{metric!r}: unrecognized threshold operator {op!r}, expected one of {_VALID_OPS}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{metric!r}: threshold value must be numeric, got {value!r}")
    return MetricThreshold(metric=metric, stat=stat, op=op, value=float(value))


def load_thresholds(path: Path = DEFAULT_THRESHOLDS_PATH) -> dict[str, MetricThreshold]:
    """Sorted by metric name (never dict/file insertion order) so two loads of the same file agree
    byte for byte on iteration order, the same determinism discipline the matrix runner applies to
    its own manifest lists."""
    path = Path(path)
    raw = json.loads(path.read_text())
    if not raw:
        raise ValueError(f"{path}: no thresholds declared; the load lane needs at least one gate")
    return {metric: parse_threshold(metric, spec) for metric, spec in sorted(raw.items())}


def render_k6_thresholds(thresholds: Mapping[str, MetricThreshold]) -> dict[str, list[str]]:
    """k6's own `options.thresholds` shape: `{metric: [expr, ...]}`, one expression per metric here
    (this lane names no metric with more than one gate). JSON serializable by construction, so the
    k6 script's own companion `thresholds.json` read needs no translation at all -- this function
    exists for the Python side report and the hermetic test that pins the exact rendered shape."""
    return {metric: [t.as_k6_expr()] for metric, t in sorted(thresholds.items())}


__all__ = [
    "DEFAULT_THRESHOLDS_PATH",
    "MetricThreshold",
    "load_thresholds",
    "parse_threshold",
    "render_k6_thresholds",
]
