"""The unified golden set: synthetic, SME authored, and promoted cases in one suite."""
from __future__ import annotations

import pytest

from evals.evalkit.golden_set import (
    FACETS,
    UNION_FACETS,
    coverage,
    excluded_silver,
    gold_only,
    golden_set,
    unified_eval_cases,
    unified_set,
)
from evals.evalkit.provenance import provenance_of


def test_the_union_contains_both_systems():
    origins = {provenance_of(r).origin for r in unified_set()}
    assert "synthetic" in origins, "registry generated cases are missing from the union"
    assert "authored" in origins, "SME authored cases are missing from the union"


def test_case_ids_are_unique_across_the_union():
    ids = [provenance_of(r).id for r in unified_set()]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f"case_id collision across the two systems: {duplicates}"


def test_gold_only_excludes_silver_and_reports_what_it_dropped():
    records = unified_set()
    kept = gold_only(records)
    assert all(provenance_of(r).tier == "gold" for r in kept)
    dropped = excluded_silver(records)
    assert len(kept) + len(dropped) == len(records)


def test_gold_only_actually_filters_a_union_that_contains_silver():
    """The real corpus is 86 gold and 0 silver today, so the assertion above (`all(tier == "gold")`)
    and the partition-count check hold identically whether `gold_only` filters at all or is a bare
    pass-through (`return records`) -- an unchecked gate is worse than an absent one. This builds a
    synthetic union with one gold and one silver record so the filter is exercised for real, without
    touching `seed_cases.jsonl` or `evals/datasets/seed.py`."""
    gold = {"case_id": "fx-gold-1", "origin": "authored", "intent": "troubleshooting", "turns": [{"user": "hi"}]}
    silver = {"case_id": "fx-silver-1", "origin": "promoted", "intent": "troubleshooting", "turns": [{"user": "hi"}]}
    union = (gold, silver)

    kept = gold_only(union)
    assert [provenance_of(r).id for r in kept] == ["fx-gold-1"]

    dropped = excluded_silver(union)
    assert dropped == ("fx-silver-1",)


def test_every_record_projects_to_an_eval_case():
    cases = unified_eval_cases(unified_set())
    assert len(cases) == len(unified_set())
    assert all(c.customer_id for c in cases), "every projected case must name a session"


def test_every_declared_grader_resolves_in_the_registry():
    from evals.evalkit.metric_graders import GOLDEN_GRADERS

    for case in unified_eval_cases(unified_set()):
        for name in case.graders:
            assert name in GOLDEN_GRADERS, f"{case.id} declares unregistered grader {name!r}"


def test_an_unregistered_grader_name_raises_at_set_build():
    """A case that runs ungraded reports green while checking nothing."""
    from evals.evalkit.case import EvalCase
    from evals.evalkit.golden_set import unified_eval_cases as _uec

    class _Rec:
        id = "bogus"
        source = "hand_written"
        tier = "gold"
        risk = "r"

        def to_eval_case(self):
            return EvalCase(id="bogus", turns=("hi",), customer_id="cust_current", graders=("no-such-grader",))

    with pytest.raises(ValueError, match="absent from GOLDEN_GRADERS"):
        _uec((_Rec(),))


def test_coverage_slices_the_union_by_origin():
    counts = coverage(unified_set())
    assert counts["origin"]["synthetic"] > 0
    assert counts["origin"]["authored"] > 0


def test_a_duplicate_case_id_across_the_union_raises_at_set_build(monkeypatch):
    """The two id namespaces are disjoint today, so this collision can only be manufactured, not
    found in the real corpus. Monkeypatch `unified_set`'s own inputs (`dataset_cases`/`GOLDEN`)
    rather than mutate the seed, and assert the message names the colliding id, distinguishing this
    guard from the two in `unified_eval_cases` below."""
    import evals.evalkit.golden_set as gs

    monkeypatch.setattr(
        gs, "dataset_cases", lambda: ({"case_id": "dup-case", "origin": "synthetic"},)
    )
    monkeypatch.setattr(gs, "GOLDEN", ({"case_id": "dup-case", "origin": "authored"},))

    with pytest.raises(ValueError, match="case_id collision") as exc:
        gs.unified_set()
    assert "dup-case" in str(exc.value)


def test_a_case_projecting_to_zero_graders_raises_at_set_build():
    """An empty grader tuple passes trivially: the case runs and reports green while checking
    nothing, so `unified_eval_cases` must catch it at set build. The message must name the
    offending id and read differently from the "absent from GOLDEN_GRADERS" guard below, so a
    refactor that collapsed the two into one generic error would be caught."""
    ungraded_record = {"case_id": "ungraded-case", "origin": "synthetic", "turns": [{"user": "hi"}]}

    with pytest.raises(ValueError, match="zero graders") as exc:
        unified_eval_cases((ungraded_record,))
    assert "ungraded-case" in str(exc.value)
    assert "absent from GOLDEN_GRADERS" not in str(exc.value)


def test_coverage_defaults_to_facets_inferred_from_content():
    """Documents the heuristic Finding 2 flags: with no Mapping present, the default infers
    GoldenCase-only FACETS and has no "origin" key, exactly the surprise an explicit override
    (below) exists to avoid."""
    counts = coverage(golden_set())
    assert "origin" not in counts
    assert set(counts) == set(FACETS)


def test_coverage_accepts_an_explicit_facets_override():
    """A GoldenCase-only tuple (zero Mapping records) would otherwise infer FACETS and drop
    "origin", so `coverage(...)["origin"]` would KeyError for a caller who wanted the union
    vocabulary. `facets=UNION_FACETS` states that intent explicitly instead of relying on content."""
    counts = coverage(golden_set(), facets=UNION_FACETS)
    assert "origin" in counts
    assert set(counts) == set(UNION_FACETS)
