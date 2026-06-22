"""Synthetic RAG test-set generation (operator lane): generate with a real model, review, freeze, then
the hermetic gate replays the frozen fixture.

Generation is non-deterministic (LLM sampling is not replayable even at temperature 0), so it can
never live in the PR gate. This lane generates candidate query/context/answer goldens from the corpus
with DeepEval's Synthesizer and writes them to a versioned JSONL artifact; a human reviews and prunes
it, and from then on the frozen set replays deterministically in the gate. Never gate model/judge
choice on synthetic-only correctness. deepeval imported lazily; run: ``task ragset``.
"""
