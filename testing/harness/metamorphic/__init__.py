"""Metamorphic testing, rebuilt against the registry (SP8 task 6, D32): does the SAME underlying
question, worded differently, still retrieve the same grounding chunk and still support the same
answer? A metamorphic relation never needs its own absolute value label, only the assertion that
related inputs relate, checked deterministically, no judge.

Supersedes the pre rewrite `evals/metamorphic/` (operator entrypoint) and
`evals/datasets/metamorphic_golden.py` (the frozen family), both built on the cold open TOY corpus
(`atlas.domain.corpus`) and its render guard. The shape survives unchanged (a frozen family, a
`family_id`, "generate, freeze, replay", an operator lane that prints a deterministic report and
never gates on an optional LLM assisted proposal step); the content does not. This rebuild is
seeded from `corpus/registry/core.yaml`'s `conflict-daniel-contract` (SP7's own `grounded_not_true`
seed, `docs/measurements/sp3-rag-spine.md`'s named conflict slice), the SAME contradiction
`judge.provisional`'s manufactured cases already consume, one registry fixture doing two jobs
across two sub projects.

Three families, one shared retrieval fixture (`families.STUB_CORPUS`, real text and real chunk ids
copied from the committed `corpus/rendered/corpus-0.1.1` render, never the toy corpus):

- `PARAPHRASE_FAMILY`   natural rewordings of "is my plan contract free" (several already curated
  as real seed cases in `dataset_tools/seed_cases.jsonl`, referenced by comment in `families.py`).
- `TYPO_NOISE_FAMILY`   character level typos of the same question (a customer fat fingering a
  word, not a deliberate adversarial edit).
- `QUERY_PERTURBATION_FAMILY`  pure surface noise (casing, whitespace, punctuation) with zero
  semantic content, the strongest tier: exact retrieval agreement, not merely a floor.

Three deterministic, judge free invariants per family (D32), all pure functions over already
computed retrieval/answer data, never a live call or a wall clock:

1. ID based retrieval agreement (`report.id_based_retrieval_agreement`): the registry's winning
   chunk is retrieved for every member, regardless of wording.
2. Rank overlap floor (`quality.ir_metrics.rank_overlap_at_k`): members' retrieved id sets do not
   drift below a family specific floor (1.0 for pure formatting noise, looser for a real
   paraphrase or typo, which may legitimately lose a distractor also retrieved alongside it).
3. Registry derived answer equivalence (`quality.agent_metrics.answer_correctness_rate`): the one
   frozen, correctly grounded answer text expresses the SAME registry fact
   (`contract_term-daniel-2025:contract_months`) regardless of which member asked for it.

The retrieval stack (`search_chunks`) is the system under perturbation: hermetic tests exercise it
for real, through `InMemoryRetriever` over the small, real content stub corpus (deterministic
keyword overlap, no embedding, no wall clock); the live lane (`metamorphic.__main__`'s own report
plus `task test:live`, covering `testing/tests/test_metamorphic_live.py`, an operator step, deferred
like SP7's) exercises the SAME invariants against the real pgvector/TEI stack over the pinned
`corpus-0.1.1-bge-m3-03f983e0` index.

Out of scope, by name: corpus mutation (D32's OTHER lane, a registry fact changed, the corpus
re rendered and re indexed, the real agent's answer asserted to track the new truth) is SP8 task 7,
a completely different concept that happens to share the word "mutation" with the pre rewrite
`evals/mutation/` module (semantic IR metric mutation testing, unrelated to either lane). Nothing
here mutates the registry, re renders a document, or re indexes anything.
"""
from __future__ import annotations
