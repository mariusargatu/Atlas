"""Access the canonical golden set, and slice it by the metadata that makes it debuggable.

Rich metadata is the difference between a dataset you can interrogate and one you can only stare at.
Without it a suite reports one number. With it you slice by surface, by risk, by source, so the day a
release goes red you know whether it broke the FAQ lookups or the plan changes, which are not the
same emergency. These are the dataset level accessors that payoff. Scoring the slices is deliberately
deferred to structured-claim extraction, and this module hands it a set it can already cut by any
facet.

The seed is all gold today. Silver to gold promotion (and the ``gold_only`` gate it needs) lands with
the production loop, where sampled real traffic enters as silver and is promoted on
review. Building the gate before any silver exists would be unused machinery.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from evals.datasets.seed import GOLDEN
from evals.evalkit.case import EvalCase
from evals.evalkit.dataset_case import dataset_case_to_eval_case
from evals.evalkit.golden_case import GoldenCase
from evals.evalkit.provenance import provenance_of

SEED_CASES_PATH = Path("testing/harness/dataset_tools/seed_cases.jsonl")

# The facets a regression is sliced by. ``risk`` is the CTO's rollup, ``category`` the surface,
# ``consequence`` the weight, and ``source``/``author_role``/``tier`` the provenance.
FACETS = ("category", "consequence", "risk", "source", "author_role", "tier")

#: The facets the UNION is sliced by. `FACETS` above stays as the GoldenCase-only vocabulary so the
#: existing callers are untouched.
UNION_FACETS = ("origin", "source", "tier", "risk")


def golden_set() -> tuple[GoldenCase, ...]:
    """The canonical, validated set (``datasets/seed.py``)."""
    return GOLDEN


def as_eval_cases(cases: tuple[GoldenCase, ...] = GOLDEN) -> tuple[EvalCase, ...]:
    """Project the set to the runner's input. This is the seam structured-claim extraction runs
    against: every later scoring pass grades these cases, and this is how they reach the runner."""
    return tuple(c.to_eval_case() for c in cases)


def dataset_cases(path: Path = SEED_CASES_PATH) -> tuple[dict, ...]:
    """Every dataset contract case, as parsed rows."""
    lines = path.read_text().splitlines()
    return tuple(json.loads(line) for line in lines if line.strip())


def unified_set() -> tuple[object, ...]:
    """Every golden record from BOTH systems: registry generated dataset cases and GoldenCase
    records (SME authored, plus promoted production traffic). Raises on a case_id collision, since
    the two id namespaces merge here and a silent last-write-wins would drop a case."""
    records: tuple[object, ...] = (*dataset_cases(), *GOLDEN)
    ids = [provenance_of(r).id for r in records]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"case_id collision across the golden systems: {duplicates}")
    return records


def gold_only(records: tuple[object, ...] | None = None) -> tuple[object, ...]:
    """The ratified cases: what the hermetic lane gates on."""
    records = unified_set() if records is None else records
    return tuple(r for r in records if provenance_of(r).tier == "gold")


def excluded_silver(records: tuple[object, ...] | None = None) -> tuple[str, ...]:
    """The ids `gold_only` dropped, sorted. Excluded, never silently: a suite that quietly shrinks
    reports green over fewer cases, the same reason the matrix runner surfaces its dropped cells."""
    records = unified_set() if records is None else records
    return tuple(sorted(provenance_of(r).id for r in records if provenance_of(r).tier != "gold"))


def unified_eval_cases(records: tuple[object, ...] | None = None) -> tuple[EvalCase, ...]:
    """Project the union to the runner's input, whichever system each record came from.

    Raises if any case declares a grader name the registry cannot resolve, OR if any case projects
    to zero graders. Both are the same defect from two directions: a case that runs with a grader
    name the registry cannot resolve, or with no grader at all, reports green while checking
    nothing, which is strictly worse than failing, so this is caught at set build rather than
    discovered as a suspiciously clean report.
    """
    from evals.evalkit.metric_graders import GOLDEN_GRADERS

    records = unified_set() if records is None else records
    cases = tuple(
        dataset_case_to_eval_case(r) if isinstance(r, Mapping) else r.to_eval_case() for r in records
    )
    unknown = sorted(
        {(c.id, name) for c in cases for name in c.graders if name not in GOLDEN_GRADERS}
    )
    if unknown:
        raise ValueError(f"cases declare graders absent from GOLDEN_GRADERS: {unknown}")
    ungraded = sorted(c.id for c in cases if not c.graders)
    if ungraded:
        raise ValueError(f"cases project to zero graders (checks nothing, would report green): {ungraded}")
    return cases


def _facet_value(record, facet: str) -> str:
    """Read a facet off either record type: GoldenCase attributes first, normalised provenance
    otherwise, so one call site slices a mixed set."""
    if not isinstance(record, Mapping) and hasattr(record, facet):
        return str(getattr(record, facet))
    return str(getattr(provenance_of(record), facet))


def slice_by(facet: str, value: str, cases: tuple[object, ...] = GOLDEN) -> tuple[object, ...]:
    """The cases whose ``facet`` equals ``value`` (e.g. ``slice_by("consequence", "high")``)."""
    if facet not in FACETS and facet not in UNION_FACETS:
        raise ValueError(f"unknown facet {facet!r}; known: {sorted(set(FACETS) | set(UNION_FACETS))}")
    return tuple(c for c in cases if _facet_value(c, facet) == value)


def coverage(
    cases: tuple[object, ...] = GOLDEN,
    *,
    facets: tuple[str, ...] | None = None,
) -> dict[str, dict[str, int]]:
    """Counts per value per facet, sorted for a canonical (determinism safe) view. Facets a record
    type does not carry are skipped for that record, so a mixed set still reports.

    ``facets`` lets a caller STATE the vocabulary instead of leaving it inferred. Left ``None``
    (the default, kept for the existing ``GoldenCase``-only call sites), the vocabulary is guessed
    from what happens to be in ``cases``: any ``Mapping`` present selects ``UNION_FACETS``,
    otherwise ``FACETS``. That guess is a heuristic on CONTENT, not on the caller's intent, and it
    is wrong exactly when a subset of the union happens to contain zero ``Mapping`` records (for
    instance a further filter over ``gold_only()`` that lands only on ``GoldenCase`` rows): the
    fallback to ``FACETS`` then silently drops ``"origin"``, and a caller who correctly expected the
    union vocabulary gets a ``KeyError`` on ``coverage(...)["origin"]`` instead of a wrong count.
    Pass ``facets=UNION_FACETS`` (or ``FACETS``) explicitly whenever the tuple you are handing in is
    not guaranteed, by how it was constructed, to still contain a marker of the record type you
    mean.
    """
    if facets is None:
        facets = UNION_FACETS if any(isinstance(c, Mapping) for c in cases) else FACETS
    return {
        facet: dict(sorted(Counter(_facet_value(c, facet) for c in cases).items()))
        for facet in facets
    }


__all__ = [
    "FACETS", "UNION_FACETS", "as_eval_cases", "coverage", "dataset_cases", "excluded_silver",
    "gold_only", "golden_set", "slice_by", "unified_eval_cases", "unified_set",
]
