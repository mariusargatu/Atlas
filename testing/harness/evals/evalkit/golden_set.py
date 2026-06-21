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

from collections import Counter

from evals.datasets.seed import GOLDEN
from evals.evalkit.case import EvalCase
from evals.evalkit.golden_case import GoldenCase

# The facets a regression is sliced by. ``risk`` is the CTO's rollup, ``category`` the surface,
# ``consequence`` the weight, and ``source``/``author_role``/``tier`` the provenance.
FACETS = ("category", "consequence", "risk", "source", "author_role", "tier")


def golden_set() -> tuple[GoldenCase, ...]:
    """The canonical, validated set (``datasets/seed.py``)."""
    return GOLDEN


def slice_by(facet: str, value: str, cases: tuple[GoldenCase, ...] = GOLDEN) -> tuple[GoldenCase, ...]:
    """The cases whose ``facet`` equals ``value`` (e.g. ``slice_by("consequence", "high")``)."""
    if facet not in FACETS:
        raise ValueError(f"unknown facet {facet!r}; known: {FACETS}")
    return tuple(c for c in cases if getattr(c, facet) == value)


def as_eval_cases(cases: tuple[GoldenCase, ...] = GOLDEN) -> tuple[EvalCase, ...]:
    """Project the set to the runner's input. This is the seam structured-claim extraction runs
    against: every later scoring pass grades these cases, and this is how they reach the runner."""
    return tuple(c.to_eval_case() for c in cases)


def coverage(cases: tuple[GoldenCase, ...] = GOLDEN) -> dict[str, dict[str, int]]:
    """Counts per value per facet, sorted for a canonical (determinism safe) view. The one glance
    answer to 'what does this set actually cover, and where is it thin?'"""
    return {
        facet: dict(sorted(Counter(getattr(c, facet) for c in cases).items()))
        for facet in FACETS
    }


__all__ = ["FACETS", "golden_set", "slice_by", "as_eval_cases", "coverage"]
