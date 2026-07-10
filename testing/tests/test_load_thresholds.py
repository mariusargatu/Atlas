"""`load.thresholds`, hermetic (SP9 task 6): the load lane's own "thresholds as code" -- one small
JSON file (`testing/harness/load/thresholds.json`) is the single source of truth for both the k6
script's `options.thresholds` (JS reads the same file via `open()`) and this pure Python parser, so
the two can never silently drift onto different numbers. Pure, no network, no k6 binary needed to
exercise this module at all.
"""
from __future__ import annotations

import json

import pytest

from load.thresholds import (
    DEFAULT_THRESHOLDS_PATH,
    MetricThreshold,
    load_thresholds,
    parse_threshold,
    render_k6_thresholds,
)


def test_parse_threshold_accepts_a_percentile_stat_with_a_less_than_operator():
    t = parse_threshold("ttft_ms", {"stat": "p(95)", "op": "<", "value": 2000})
    assert t == MetricThreshold(metric="ttft_ms", stat="p(95)", op="<", value=2000.0)


def test_parse_threshold_accepts_a_rate_stat():
    t = parse_threshold("goodput", {"stat": "rate", "op": ">", "value": 0.95})
    assert t.stat == "rate"
    assert t.value == pytest.approx(0.95)


def test_parse_threshold_rejects_an_unrecognized_operator():
    with pytest.raises(ValueError, match="operator"):
        parse_threshold("ttft_ms", {"stat": "p(95)", "op": "!=", "value": 2000})


def test_parse_threshold_rejects_an_unrecognized_stat():
    with pytest.raises(ValueError, match="stat"):
        parse_threshold("ttft_ms", {"stat": "banana", "op": "<", "value": 2000})


def test_parse_threshold_rejects_a_non_numeric_value():
    with pytest.raises(ValueError, match="numeric"):
        parse_threshold("ttft_ms", {"stat": "p(95)", "op": "<", "value": "fast"})


def test_parse_threshold_rejects_an_empty_metric_name():
    with pytest.raises(ValueError, match="metric"):
        parse_threshold("", {"stat": "p(95)", "op": "<", "value": 2000})


def test_metric_threshold_renders_its_own_k6_expression():
    t = MetricThreshold(metric="ttft_ms", stat="p(95)", op="<", value=2000.0)
    assert t.as_k6_expr() == "p(95)<2000"


def test_metric_threshold_renders_a_fractional_value_without_a_trailing_zero_pad():
    t = MetricThreshold(metric="goodput", stat="rate", op=">", value=0.95)
    assert t.as_k6_expr() == "rate>0.95"


def test_load_thresholds_reads_the_committed_canonical_file():
    thresholds = load_thresholds()
    assert set(thresholds) == {"ttft_ms", "tokens_per_sec", "e2e_ms", "goodput"}
    assert all(isinstance(t, MetricThreshold) for t in thresholds.values())


def test_load_thresholds_rejects_an_empty_file(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="no thresholds"):
        load_thresholds(path)


def test_default_thresholds_path_points_at_the_committed_file_next_to_this_module():
    assert DEFAULT_THRESHOLDS_PATH.name == "thresholds.json"
    assert DEFAULT_THRESHOLDS_PATH.exists()


def test_render_k6_thresholds_produces_the_exact_options_thresholds_shape():
    thresholds = load_thresholds()
    rendered = render_k6_thresholds(thresholds)
    assert rendered["ttft_ms"] == ["p(95)<2000"]
    assert rendered["goodput"] == ["rate>0.95"]
    # every value is a list (k6's own shape: one or more expressions per metric), never a bare string
    assert all(isinstance(v, list) for v in rendered.values())


def test_render_k6_thresholds_is_json_serializable_for_the_k6_script_to_consume():
    thresholds = load_thresholds()
    rendered = render_k6_thresholds(thresholds)
    json.dumps(rendered)  # must not raise
