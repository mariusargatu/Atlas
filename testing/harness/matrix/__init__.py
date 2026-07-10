"""The staged benchmark matrix runner (SP9 task 4): Atlas stops being one RAG pipeline and becomes a
measured comparison of retrieval/rerank/generation components, wired to substrate SP7 and SP8 already
built and shipped unwired: `quality.stats` (interval + significance math), `quality.ir_metrics` (the
per query retrieval arithmetic), `quality.gate` (the release gate), `quality.agent_metrics` (reference
based answer correctness), and `judge.panel.panel_vote` (the D15 headline jury, ITS FIRST REAL CALLER
anywhere in this repo). This package is a CALLER of that substrate, not a second implementation of it.

Staged, per D17, explicitly NOT a cross product of every axis:

  Stage 1 (`embedders.py`)  embedders on retrieval only metrics (recall@k, nDCG@k), no LLM anywhere.
                            The two real embedder axes (`bge-m3` local, `text-embedding-3-small`
                            openai, the documented narrowness: no Voyage key) PLUS the two named
                            baseline rows every table carries (BM25 + no reranker, exact_scan the
                            recall ground truth row).
  Stage 2 (`rerankers.py`)  rerankers over STAGE 1's cached candidate lists, at depths {20, 50, 100}
                            (research 14: reranker quality can degrade past a depth). Axis
                            {BGE reranker v2 m3, none} -- thin on purpose, no Voyage rerank key,
                            named rather than padded.
  Stage 3 (`generators.py`) the top 1 to 2 retrieval configs (`select.py` picks by nDCG) times the
                            generator axis {Claude, GPT, qwen2.5:7b}, scored by
                            `quality.agent_metrics.answer_correctness_rate` (primary) and
                            `judge.panel.panel_vote` (secondary) -- plus ONE off diagonal validation
                            cell checking (never asserting) the staged design's own independence
                            assumption.

`cache.py` is the content hash cache (D17's "compute missing cells" pattern: a rerun recomputes only
what changed), keyed `hash(corpus_version, dataset_version, component_id, params)` via the SAME
canonical digest the cassette key already uses. `lineage.py` builds every cell's row in EXACTLY
`contracts/manifest/schema.json`'s shape (D26's 12 field attribution tuple) -- this package is a
caller of that frozen contract too, never a second lineage shape. `runner.py` assembles the staged
stages into one run manifest plus HELM style per query result files.

Fully hermetic by construction: every stage's own tests run against seeded REPLAY fixtures (the
gateway's REPLAY mode for every generator/judge call, a small committed fixture table standing in for
a live embedder/retriever call), keyless and networkless. The live sweep against real
TEI/OpenAI/Anthropic/Ollama is deferred to a batched live capture session (this package's own
mechanics unchanged either way); the spend gate and the real cost column are SP9 task 5's, not built
here.
"""
from __future__ import annotations

__all__: list[str] = []
