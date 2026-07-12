"""DeepEval test cases for the RAG eval lane, built lazily from committed data so the corpus grows
independently of the runner. The headline case is the cold open: an answer that is perfectly
faithful to the document the retriever found and yet false against the oracle. DeepEval's faithfulness
metric will (correctly) score it high, faithfulness keeps the model honest about its sources; only
Atlas's own deterministic correctness-vs-oracle check (``domain.metrics.is_correct_vs_truth``) catches
that the source was the wrong one.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagExample:
    """Plain data; converted to a DeepEval ``LLMTestCase`` at run time (lazy import)."""

    input: str
    actual_output: str
    retrieval_context: tuple[str, ...]
    expected_output: str


RAG_EXAMPLES: list[RagExample] = [
    RagExample(
        input="Does my plan have a data cap?",
        actual_output="No, your plan is unlimited with no data cap and no contract.",
        retrieval_context=("The current Fast plan is unlimited with no data cap and no contract.",),
        # the oracle truth for this (legacy) customer; the faithful answer above contradicts it
        expected_output="The legacy Saver plan has a monthly data cap and is throttled when exceeded.",
    ),
    RagExample(
        input="Are late fees waived during an outage?",
        actual_output="Yes. During a confirmed network outage, late fees are waived for affected customers.",
        retrieval_context=("During a confirmed network outage late fees are waived for affected customers.",),
        expected_output="Yes, late fees are waived during a confirmed outage.",
    ),
]


def to_test_cases(examples: list[RagExample] | None = None):
    """Convert to DeepEval ``LLMTestCase`` objects. deepeval imported lazily (rageval group only)."""
    from deepeval.test_case import LLMTestCase

    return [
        LLMTestCase(
            input=e.input,
            actual_output=e.actual_output,
            retrieval_context=list(e.retrieval_context),
            expected_output=e.expected_output,
        )
        for e in (examples if examples is not None else RAG_EXAMPLES)
    ]
