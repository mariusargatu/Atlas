"""Runnable RAG eval lane (`task rageval`): DeepEval's LLM-judged RAG metrics with a pinned local
Ollama judge, NON-gating (``evaluate``, not ``assert_test``). Needs the ``rageval`` group installed
and an Ollama daemon; it is never the PR lane. If deepeval is missing it prints how to run it rather
than crashing with an ImportError, the same courtesy the record lane extends.
"""
from __future__ import annotations


def main() -> None:
    try:
        from deepeval import evaluate
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
    except ImportError:
        print(
            "deepeval is not installed (operator lane). Run:\n"
            "  task rageval\n"
            "or:  uv run --group rageval python -m evals.rageval\n"
            "and start an Ollama daemon for the local judge."
        )
        return

    from evals.rageval.cases import to_test_cases
    from evals.rageval.judge import build_ollama_judge

    judge = build_ollama_judge()  # pinned local Ollama judge, temperature 0
    metrics = [
        FaithfulnessMetric(model=judge),
        AnswerRelevancyMetric(model=judge),
        ContextualPrecisionMetric(model=judge),
        ContextualRecallMetric(model=judge),
    ]
    # evaluate() reports; it does not gate. Under Option A these judged metrics never block a build.
    evaluate(test_cases=to_test_cases(), metrics=metrics)


if __name__ == "__main__":
    main()
