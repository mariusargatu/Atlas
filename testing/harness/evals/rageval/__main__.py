"""Runnable RAG eval lane (`task rageval`): DeepEval's LLM-judged RAG metrics with the default operator
judge, NON-gating (``evaluate``, not ``assert_test``). The judge is the deployable cross-family OpenAI
judge (gpt-5.4-nano, structured-output-capable with this key) when ``OPENAI_API_KEY`` is set, else the
local Ollama judge. The plain-completion calibration winner (gpt-5.6-luna) 401s on structured outputs
here; see ``rageval.judge`` for the reconciliation.
Needs the ``rageval`` group installed; it is never the PR lane. If deepeval is missing it prints how to
run it rather than crashing with an ImportError, the same courtesy the record lane extends.
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
            "or:  uv run --group rageval --env-file .env python -m evals.rageval\n"
            "The judge is gpt-5.4-nano when OPENAI_API_KEY is set (.env), else a local Ollama daemon."
        )
        return

    from evals.rageval.cases import to_test_cases
    from evals.rageval.judge import build_default_judge

    judge, label = build_default_judge()  # gpt-5.4-nano when a key is set, else local Ollama
    print(f"judge: {label}")
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
