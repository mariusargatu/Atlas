"""The golden dataset layer (04): access and metadata slicing.

These pin the dataset-article machinery that surrounds the seed: the intermediary CSV and the
canonical seed describe the same set, the metadata is sliceable, and the coverage adds up. Scoring
the cases is the metrics article's job (05); silver-to-gold promotion and decontamination land with
the production loop (12). This stops at the dataset boundary.
"""
from __future__ import annotations

from pathlib import Path

from evals.evalkit.case import EvalCase
from evals.evalkit.golden_loader import load_golden_drafts
from evals.evalkit.golden_set import (
    as_eval_cases,
    coverage,
    golden_set,
    slice_by,
)

_CSV = Path(__file__).resolve().parents[1] / "harness/evals/datasets/atlas_golden.csv"


def test_intermediary_csv_describes_the_same_set_as_the_canonical_seed():
    # The CSV is the explanation; seed.py is canonical. If their ids drift, the explanation lies.
    csv_ids = {d.id for d in load_golden_drafts(_CSV)}
    seed_ids = {c.id for c in golden_set()}
    assert csv_ids == seed_ids


def test_golden_set_is_the_canonical_ten():
    assert len(golden_set()) == 10


def test_slice_by_facet():
    actions = slice_by("category", "action", golden_set())
    assert {c.category for c in actions} == {"action"}
    assert len(actions) == 6


def test_slice_by_unknown_facet_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown facet"):
        slice_by("nonsense", "x")


def test_as_eval_cases_projects_for_the_runner():
    cases = as_eval_cases()
    assert len(cases) == 10
    assert all(isinstance(c, EvalCase) for c in cases)


def test_coverage_counts_add_up_per_facet():
    cov = coverage()
    assert set(cov) == {"category", "consequence", "risk", "source", "author_role", "tier"}
    for facet, counts in cov.items():
        assert sum(counts.values()) == 10, facet                # every case counted once per facet
    assert cov["consequence"]["high"] >= 5                      # stratified toward consequence
