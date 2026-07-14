"""The honest benchmark worked example: the committed study reproduces and reads as 'no regression'.

The committed artifact embeds these exact numbers, so the study is pinned here: a seeded run is a
measurement only if it recomputes identically, and the verdict must stay 'you cannot tell yet'.
"""
from __future__ import annotations

import json

import pytest

from evals.benchmark.dataset import N, paired_vectors
from evals.benchmark.study import render, run


@pytest.fixture(scope="module")
def study_result():
    return run()


def test_fixture_marginals_match_the_cold_open():
    a, b = paired_vectors()
    assert len(a) == len(b) == N == 100
    assert sum(a) == 84
    assert sum(b) == 81


def test_study_is_reproducible_under_the_stamped_seed(study_result):
    assert run() == study_result


def test_study_reports_overlapping_intervals_and_no_significant_gap(study_result):
    r = study_result
    assert r["marginal_intervals_overlap"] is True
    assert r["paired"]["diff_ci_excludes_zero"] is False
    assert r["paired"]["permutation_p"] > 0.05
    assert r["significant"] is False


def test_study_numbers_match_the_committed_artifact(study_result):
    # The exact values the committed artifact is kept in sync with. A drift here is a drift in the report.
    r = study_result
    assert round(r["v_a"]["ci95"][0], 3) == 0.756
    assert round(r["v_a"]["ci95"][1], 3) == 0.899
    assert round(r["v_b"]["ci95"][0], 3) == 0.722
    assert round(r["v_b"]["ci95"][1], 3) == 0.875
    assert round(r["paired"]["diff"], 3) == 0.030
    assert round(r["paired"]["mcnemar_p"], 3) == 0.549


def test_render_includes_the_verdict_and_marginal_scores(study_result):
    text = render(study_result)
    assert study_result["verdict"] in text
    assert "84/100" in text
    assert "81/100" in text


def test_release_gate_fails_closed_on_the_candidate(study_result):
    # The candidate's point (0.81) sits above the 0.80 bar, but its Wilson floor (0.722) does
    # not. Gating on the floor, never the point, turns "we cannot tell yet" into a hold.
    g = study_result["release_gate"]
    assert g["threshold"] == 0.80
    assert g["verdict"] == "fail"
    assert round(g["lower_bound"], 3) == 0.722
    assert g["lower_bound"] < g["threshold"] < study_result["v_b"]["rate"]
    assert g["width"] <= g["variance_budget"]  # within budget, so a fail, not a quarantine


def test_render_includes_the_release_gate(study_result):
    text = render(study_result)
    assert "release gate" in text.lower()
    assert study_result["release_gate"]["reason"] in text


def test_committed_artifacts_match_a_fresh_run_byte_for_byte(study_result):
    # A stamped-seed regeneration must match the committed artifact's bytes exactly, or the
    # artifact is a snapshot, not a measurement.
    from evals.benchmark.__main__ import JSON, MD

    text = render(study_result)
    assert MD.read_text() == text
    assert JSON.read_text() == json.dumps(study_result, indent=2) + "\n"
