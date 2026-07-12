"""Synthetic RAG test-set generation (operator lane): generate with a real model, review, FREEZE,
then the hermetic gate replays the frozen fixture (doc 07).

Generation is non-deterministic (LLM sampling is not replayable even at temperature 0), so it can
never live in the PR gate. This lane generates candidate query/context/answer goldens from the corpus
with DeepEval's Synthesizer and writes them to a versioned JSONL artifact. A human reviews and prunes
that artifact; from then on the frozen set replays deterministically in the gate, retriever tuning is
exactly the use case synthetic data is reliable for. Never gate model/judge choice on synthetic-only
correctness. deepeval imported lazily; run: ``task ragset``.
"""
