# Measurement: the SP3 RAG spine, two flagship findings

Committed factual record for two measurements taken during the SP3 final review, against the live
compose stack (real Postgres, real TEI embed and rerank, the committed `corpus-0.1.1` index build).
Both are recorded, not asserted on: this project's own doctrine is that a probabilistic outcome
(what a cross-encoder ranks first, whether an ONNX backend sums floats in the same order twice) is
measured and named, not silently gated by a test that would either be flaky or would hide the
finding behind a green check mark. See `testing/tests/test_pgvector_adapter_live.py` and
`testing/tests/test_ingest_live.py` for the live tests that reproduce these numbers; the exact
figures below are copied into those tests' own docstrings too, so the number and its source stay
next to each other.

## 1. The reranker demotes the planted conflict chunk

Query: `"is my plan contract free"`, against Daniel's account (`conflict-daniel-contract`, SP2's
deliberately planted grounding conflict: Daniel's individual, 12 month contract terms doc says the
opposite of the generic marketing plan pages).

- Fused, pre-rerank candidate pool (RRF over the HNSW and tsvector arms, `k_fused=50` on a 45 chunk
  corpus, so this is effectively the whole corpus): the Daniel chunk lands at **rank 5 of 45**.
- With reranking on (`BAAI/bge-reranker-v2-m3`, live, `k_final=50`): the Daniel chunk drops to
  **rank 14**, rerank score **0.00136**.
- Every chunk that outranks it after reranking literally contains the phrase `"No contract. Cancel
  any time."` The reranker is a generic cross-encoder with no notion of "this customer has an
  override"; it rewards lexical and semantic closeness to the query's surface form, and the
  marketing pages that repeat "no contract" verbatim are closer to that surface form than Daniel's
  own contract terms chunk is.

**Doctrine: measured, not gated.** The retrieval MECHANICS are gated (the Daniel chunk is present
in the fused candidate pool at all; reranking runs and attaches real, distinguishable scores) because
those are deterministic, assertable facts about this adapter. Which chunk a generic reranker ranks
first on a conflict query is not a defect in `PgvectorRetriever`; asserting a specific reranker
outcome here would either be a flaky test (reranker updates, corpus edits) or would silently
launder a real finding into "just another passing test." The finding is real and belongs to the
quality plane, not the mechanics plane.

**Carries forward to:**
- **SP7a**: this becomes a named baseline row in the golden retrieval/quality set, not a fixed
  correctness assertion. A future rerank model swap or a query rewrite step should be measured
  against this baseline, not silently break an assertion nobody reads.
- **SP8**: `conflict-daniel-contract` is the seed for a conflict slice, a set of queries where the
  right answer depends on resolving a customer-specific override against generic content. This
  measurement is evidence the resolution has to happen at the agent/tool level (the account tool
  answering from live customer data), not by hoping retrieval ranks the override chunk first.

## 2. Ingest is not byte identical across rebuilds

Rebuilding the `corpus-0.1.1` index (same corpus_version, same pinned `BAAI/bge-m3` revision, same
chunker config, i.e. an identical `index_build_id`) does not reproduce the committed
`chunks.parquet` byte for byte.

- A first live probe rebuilding the corpus found **9 of the 45** embedding vectors differing from
  the committed build's corresponding chunk_id (by exact float equality); minimum cosine similarity
  across all 45 chunks was **0.999999999998** (twelve nines).
- Re-verified while wiring the live cosine gate test below: a second, independent rebuild found
  **25 of the 45** vectors differing, minimum cosine similarity **0.9999999999993494** (also twelve
  nines). The exact count of differing vectors is NOT stable run to run (the underlying cause, below,
  is thread scheduling that is not seeded), but both independent probes land in the same regime:
  a single digit to several dozen of the 45 vectors show float noise, and the worst observed cosine
  similarity across both runs never drops below twelve nines. That stability of the FLOOR, not the
  count, is what the live gate below actually pins.

**Cause.** TEI's ONNX Runtime backend (the embedding model's serving backend, `BAAI/bge-m3`) sums
per-token contributions across CPU threads in an order that is not pinned run to run. Floating point
addition is not associative, so summing the same numbers in a different order produces a different
last few mantissa bits of a float32 result. Nothing about the corpus, the pinned model revision, or
the chunker config changed between builds; the drift is purely in how the backend's internal
reduction is scheduled across threads on this machine.

**Consequence: the committed artifact is the frozen reference, not a reproducible derivation.**
`indexes/corpus-0.1.1-bge-m3-03f983e0/chunks.parquet` is committed and loaded as is (`rag-init`'s
`--load-existing` path never rebuilds it); a fresh `task rag:ingest` run is expected to agree with
it within float noise, not reproduce it byte for byte. The plan's fallback for this scenario (Task
5) is a cosine similarity gate instead of a byte equality check, and that gate is now a real,
running live test:
`test_rebuilt_index_vectors_agree_with_the_committed_parquet_within_cosine_tolerance` in
`testing/tests/test_ingest_live.py` rebuilds the index into a throwaway directory and asserts every
vector's cosine similarity against the committed parquet exceeds `0.9999`, a floor two full orders
of magnitude looser than the worst case actually measured.
