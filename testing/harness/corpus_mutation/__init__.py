"""Corpus mutation (SP8 task 7, D32): the OTHER lane, genuinely different from Task 6's metamorphic
suite even though English hands them neighboring words.

Metamorphic (Task 6, `testing/harness/metamorphic/`): the SAME underlying truth, asked with
DIFFERENT wording; the answer must STAY THE SAME regardless of how the question was phrased.

Corpus mutation (this package): a DIFFERENT truth (one registry fact changed), asked with the SAME
wording; the answer must CHANGE to track the new truth. This is the test that catches an agent
answering from parametric or training knowledge, or a stale cache or index, instead of the freshly
retrieved context: an agent that keeps repeating the pre mutation value after the corpus has been
re rendered and re indexed around it fails this lane, even though the QUESTION never changed at all.

No pre rewrite equivalent, and a name collision worth naming explicitly: `evals/mutation/
mutants.py` (the pre rewrite `testing/harness/evals/mutation/` module, `task mutation`) is a
completely UNRELATED idea that happens to share the English word "mutation." It is classic software
mutation testing OF `quality.ir_metrics` itself, a frozen registry of realistic IR metric bugs, each
paired with a witness input proving the Phase 1 test suite has teeth. It has nothing to do with
registry facts, corpus re rendering, or agent answers; this package does not adopt it, does not
delete it, and does not claim it. It remains unclaimed SP7 quality territory, exactly as the SP8
digest (`.superpowers/sdd/sp8-planning-digest.md`, section 2) already names.

The lane, three libraries, never reimplemented:

1. `selection.select_mutation` picks ONE registry fact deterministically (the SAME
   conflict-daniel-contract winning fact Task 6's metamorphic lane and Task 3's manufactured
   failures already seed from), and `selection.mutate_registry` returns a THROWAWAY copy of the
   registry with only that one fact changed. The real, committed `corpus/registry/*.yaml` files are
   never written to.
2. `corpus_tools.render.render_corpus` (SP2) re renders the mutated registry, a real library call,
   never reimplemented here; `selection.affected_doc_ids` then selects only the documents whose OWN
   placements actually cite the mutated fact, so only those need to be written into the ephemeral
   corpus_version and re indexed. A fact that only ever appears in one document's provenance changes
   a re render footprint of one document, not the whole corpus.
3. `scope.EphemeralCorpusVersion` scopes that render under its OWN corpus_version, one that can
   never collide with the committed `corpus-0.1.1` (the frozen artifact rule the committed render
   itself is held to, `testing/tests/test_corpus_build.py`), and removes every byte it wrote once
   the probe is done. Nothing from this lane is ever committed.
4. `rag_tools.ingest` (SP3) re indexes the affected documents for real: TEI embeddings behind a live
   server, a real Postgres load. A library call, never reimplemented.
5. The real agent answers the SAME question the mutated fact concerns, and
   `tracking.answer_tracks_mutated_truth` dereferences that answer against the NEW value through
   SP7's own reference based `quality.agent_metrics.is_fact_grounded`. No home rolled grounding
   logic lives here.

Burst/live only, never hermetic (D32's own text): steps 2 and 4 need a real re render plus a real re
index (TEI embeddings behind a live server), so `corpus_mutation/__main__.py` (`task
corpus-mutation`) is a `docker compose up` dependent operator entrypoint, the same fastlane node
was deleted precedent `testing/tests/test_metamorphic_live.py` already documents for Task 6's own
live lane. Hermetic tests cover ONLY the pure pieces: `selection`'s mutation selection logic (which
fact, deterministic) and its affected document selection, `scope`'s ephemeral corpus_version naming
and cleanup, and `tracking`'s answer-tracks-truth assertion logic over a stub pair of answers, never
a real generation call.
"""
from __future__ import annotations
