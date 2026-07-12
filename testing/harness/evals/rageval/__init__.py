"""The RAG eval lane (Option A): DeepEval's LLM-judged RAG metrics, operator-run, NON-gating.

DeepEval's contextual precision/recall, faithfulness, and answer relevancy are LLM-as-judge, so they
cannot enter the hermetic PR gate (the deterministic IR/graph metrics in ``evals.retrieval`` do that).
Here they run with a pinned local judge (Ollama) as a second opinion and a calibration study, exactly
the lifecycle the tools want: RAGAS/DeepEval to explore and calibrate, the deterministic gate to gate.

deepeval is imported lazily, inside the entrypoints, so the hermetic lane never needs it: like the
``record`` provider SDKs, the ``rageval`` dependency group is not installed by ``task test``. Run:

    task rageval          # uv run --group rageval python -m evals.rageval
"""
