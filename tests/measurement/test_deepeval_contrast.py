from __future__ import annotations

import os
import tomllib
from pathlib import Path

import yaml

from atlas.contracts import QuestionName
from evals.deepeval_suite import _metric_name, contrast_row, contrast_scorers

REPO_ROOT = Path(__file__).resolve().parents[2]   # never the working directory


def test_the_library_version_is_pinned_to_exactly_one_release():
    deps = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    pins = [d.replace(" ", "") for d in deps if d.replace(" ", "").startswith("deepeval")]
    assert len(pins) == 1, f"exactly one dependency entry names the library, found {pins}"
    name, separator, version = pins[0].partition("==")
    assert (name, separator) == ("deepeval", "==") and version[:1].isdigit()
    scorers = contrast_scorers()
    assert [_metric_name(s) for s in scorers] == [
        "Answer Relevancy", "Faithfulness", "Contextual Relevancy", "refusal_correctness [GEval]",
    ]


def test_the_two_context_metrics_that_need_a_reference_answer_are_deliberately_absent():
    # Both require expected_output, and tau2's gold set names required documents rather than
    # the answer, so neither can run on this corpus. Asserted so that adding one is a
    # deliberate act rather than an import that quietly fails at measure time.
    from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
    from deepeval.test_case import SingleTurnParams

    for metric in (ContextualPrecisionMetric, ContextualRecallMetric):
        assert SingleTurnParams.EXPECTED_OUTPUT in metric._required_params, (
            f"{metric.__name__} no longer needs a reference answer, so it may now be "
            "computable on this corpus; see evals/deepeval_suite.py"
        )
    assert all(
        _metric_name(s) not in {"Contextual Precision", "Contextual Recall"}
        for s in contrast_scorers()
    )


# Twelve of the ninety-seven, spread across both verdicts so neither column is constant.
_SCORER_DISAGREES_ON = frozenset({1, 5, 7, 13, 19, 20, 25, 31, 40, 55, 70, 85})


def _sample_row_inputs():
    """Ninety-seven questions, with a scorer that agrees with the judge on eighty-five of
    them rather than on all of them: a scorer that agrees everywhere gives every bootstrap
    resample kappa 1.000, so the interval is [+1.000, +1.000] and no interval assertion
    can fail against it.
    """
    names = [QuestionName(f"q{i:04d}") for i in range(97)]
    verdicts = {name: ("fail" if i % 5 == 0 else "pass") for i, name in enumerate(names)}
    scores = {
        name: (0.9 if (verdicts[name] == "pass") is (i not in _SCORER_DISAGREES_ON) else 0.2)
        for i, name in enumerate(names)
    }
    return scores, verdicts


def test_contrast_row_carries_no_noise_aware_interval_without_a_flip_floor():
    scores, verdicts = _sample_row_inputs()
    row = contrast_row("sample", scores, verdicts, threshold=0.5, seed=0)
    assert row is not None
    assert row.low_with_judge_noise is None and row.high_with_judge_noise is None


def test_contrast_row_carries_a_noise_aware_interval_pulled_toward_zero():
    # Toward zero, not merely wider: flipping a reference verdict biases kappa down, so the
    # noise-aware interval sits lower than the plain one and can be narrower too. The floor
    # has to stay below 0.5, where the resampled reference is a fair coin whatever the judge
    # said and every scorer comes back with the same interval around zero.
    scores, verdicts = _sample_row_inputs()
    row = contrast_row("sample", scores, verdicts, threshold=0.5, seed=0, judge_flip_floor=0.247)
    assert row is not None
    assert row.low_with_judge_noise is not None and row.high_with_judge_noise is not None
    assert row.high > row.low > 0.0, "the fixture's plain interval has to be able to move"
    assert row.low_with_judge_noise < row.low
    assert row.high_with_judge_noise < row.high
    assert row.high_with_judge_noise < row.chance_corrected


def test_the_cache_is_disabled_in_the_committed_configuration_and_in_the_running_process():
    # Both, not either: reading only the process environment passes on a machine where
    # somebody exported the variable while the committed configuration says nothing. The
    # library's cache does not notice a judge model change, so a cached score from a
    # different model would read as a fresh measurement.
    tool = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["tool"]
    assert any("DEEPEVAL_DISABLE_CACHE=YES" in str(table) for table in tool.values())
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text())
    assert workflow["jobs"]["evaluation_numbers"]["env"]["DEEPEVAL_DISABLE_CACHE"] == "YES"
    assert os.environ["DEEPEVAL_DISABLE_CACHE"] == "YES"
