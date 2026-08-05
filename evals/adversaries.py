"""The deliberately broken systems, as a table the tests walk through. Each system scores
perfectly on one measure while failing a real one; the healthy control is the one no
detector may fire on. Adding a broken system or a measurement without wiring it to the
other fails the build.

A judge cannot be broken inside `System`: it carries a retrieval and an answer, not a
verdict. See the exemption reasons in evals/__init__.py, which name the tests that assert
judge and reranker failure modes directly instead.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, replace

from atlas.contracts import ChunkId, Question
from atlas.corpus.gold import ANSWERABLE_KINDS
from evals.injection import injected_instruction
from evals.ir_metrics import ndcg_at_k, precision_at_k, recall_at_k, success_at_k
from evals.systems import System, build_broken, build_healthy
from evals.validity import Benchmark, CheckOutcome, fixed_list_check

# Fixed rather than `system.benchmark.limit`: the thresholds below are measured at this
# depth, and a threshold means nothing detached from the depth it was measured at.
_LIMIT = 10
# Geometric mean of precision@10 for search_returns_everything (0.0876, must fire) and
# same_ten_passages (0.2165, must stay silent).
_PRECISION_THRESHOLD = 0.14
# retrieves_nothing scores nDCG 0.0000 at any depth; search_returns_everything's first ten
# score 0.0917. Only zero falls below this band, so this catches an empty ranking only.
_RANKING_QUALITY_THRESHOLD = 0.02
# retrieves_nothing scores 0.0 here; the fixed list that ignores the question clears
# 0.732. Anything in between separates them.
_SUCCESS_THRESHOLD = 0.05


__all__ = [
    "ALL_DETECTORS",
    "BROKEN_SYSTEMS",
    "BROKEN_SYSTEMS_BY_NAME",
    "BrokenSystemCase",
    "System",
    "build_broken",
    "build_healthy",
    "fixed_list_outcome",
    "resolve_detector",
    "run_detector",
]


def _injection_check(system: System) -> bool:
    """Fires when anything retrieved for any question carries text that reads as instruction.

    Scoped to what was retrieved, not the whole corpus: a poisoned document nothing ever
    retrieves has not reached a prompt.
    """
    by_name = {chunk.name: chunk.text for chunk in system.benchmark.chunks}
    return any(
        injected_instruction(by_name.get(name, "")) is not None
        for ranked in system.retrieved.values()
        for name in ranked
    )


def _citation_check(system: System) -> bool:
    return any(a.violations for a in system.answers.values())


def _reference_correctness(system: System) -> bool:
    """Fires when the correct passage was shown and the system still failed to answer.

    Only when shown: a system that honestly refuses because its own retrieval missed the
    fact is not what this detector is about.
    """
    for question in system.benchmark.questions:
        if question.kind not in ANSWERABLE_KINDS:
            continue
        answer = system.answers[question.name]
        correct = system.benchmark.gold.correct(question).chunks
        if not (correct & set(answer.shown)):
            continue
        if answer.outcome != "answered":
            return True
    return False


def _mean_below_at_limit(
    system: System,
    measurement: Callable[[tuple[ChunkId, ...], frozenset[ChunkId], int], float],
    threshold: float,
) -> bool:
    """Averages one measurement over what a system retrieved, scored at the shared depth
    the pipeline runs at, and fires when the mean falls under a threshold.

    Scored at `_LIMIT` for every system rather than at each system's own returned length,
    so means are comparable across systems (a system returning more junk would otherwise
    dilute its own score and buy immunity).

    Non-finite scores are dropped: `precision_at_k` returns NaN for an empty ranking, and
    letting that reach `fmean` would silently make every comparison false.
    """
    scores = []
    for question in system.benchmark.questions:
        correct = system.benchmark.gold.correct(question).chunks
        if not correct:
            continue
        score = measurement(system.retrieved[question.name], correct, _LIMIT)
        if math.isfinite(score):
            scores.append(score)
    return bool(scores) and statistics.fmean(scores) < threshold


def _success_at_k(system: System) -> bool:
    return _mean_below_at_limit(system, success_at_k, _SUCCESS_THRESHOLD)


def _precision_at_k(system: System) -> bool:
    return _mean_below_at_limit(system, precision_at_k, _PRECISION_THRESHOLD)


def _ranking_quality_at_k(system: System) -> bool:
    return _mean_below_at_limit(system, ndcg_at_k, _RANKING_QUALITY_THRESHOLD)


def _retrieved_recall(system: System) -> Callable[[Question, Benchmark], float]:
    """A `real_scorer` that scores the passages this system actually returned, at the
    benchmark's own limit. Recall, matching `fixed_list_scorer`, so `fixed_list_check`
    compares the same quantity on both arms.
    """

    def score(question: Question, benchmark: Benchmark) -> float:
        correct = benchmark.gold.correct(question).chunks
        if not correct:
            return math.nan
        return recall_at_k(system.retrieved[question.name], correct, benchmark.limit)

    return score


def fixed_list_outcome(system: System) -> CheckOutcome:
    """The paired comparison behind `_fixed_list_validity_check`, interval and all.

    Public because the detector returns only a bool, and tests need the interval itself.
    """
    return fixed_list_check(replace(system.benchmark, real_scorer=_retrieved_recall(system)))


def _fixed_list_validity_check(system: System) -> bool:
    """Fires when what the system retrieved does not beat the question-ignoring fixed list.

    Scores `system.retrieved` rather than `system.benchmark`: scoring the benchmark alone
    measures every system identically regardless of what it actually retrieved.
    """
    return not fixed_list_outcome(system).passed


_DETECTORS: dict[str, Callable[[System], bool]] = {
    "citation_check": _citation_check,
    "success_at_k": _success_at_k,
    "reference_correctness": _reference_correctness,
    "precision_at_k": _precision_at_k,
    "ndcg_at_k": _ranking_quality_at_k,
    "fixed_list_validity_check": _fixed_list_validity_check,
    "injection_check": _injection_check,
}

ALL_DETECTORS = tuple(_DETECTORS)


def resolve_detector(name: str) -> Callable[[System], bool] | None:
    return _DETECTORS.get(name)


def run_detector(name: str, system: System) -> bool:
    detector = resolve_detector(name)
    if detector is None:
        raise ValueError(f"no detector named {name!r}")
    return detector(system)


@dataclass(frozen=True, slots=True)
class BrokenSystemCase:
    name: str
    caught_by: tuple[str, ...]


# `judge_always_approves` and `pass_through_reranker` are not represented here: `System`
# carries a retrieval and an answer, not a verdict, so a judge cannot be broken inside this
# type. Those properties are asserted directly in tests/measurement/test_calibration.py and
# tests/measurement/test_report.py instead.
BROKEN_SYSTEMS = (
    BrokenSystemCase("empty_answer_writer", ("citation_check",)),
    BrokenSystemCase("search_returns_everything", ("precision_at_k", "fixed_list_validity_check")),
    BrokenSystemCase("ignores_account_records", ("reference_correctness",)),
    BrokenSystemCase("same_ten_passages", ("fixed_list_validity_check",)),
    BrokenSystemCase("fluent_wrong_citations", ("citation_check",)),
    # precision_at_k is absent here: precision over an empty ranking is NaN, not zero, so
    # `_mean_below_at_limit` skips it rather than counting it as a failure.
    BrokenSystemCase("retrieves_nothing", ("success_at_k", "ndcg_at_k", "fixed_list_validity_check")),
    BrokenSystemCase("poisoned_corpus", ("injection_check",)),
)

BROKEN_SYSTEMS_BY_NAME = {c.name: c for c in BROKEN_SYSTEMS}
