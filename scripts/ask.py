"""Answer one real question with the whole pipeline and real models.

    just ask "How do I dispute a card transaction?"
    just ask --list           # show questions the corpus can answer
    just ask --dry-run "..."  # everything except the generation call

Run through `just`, which exports PYTHONPATH and loads .env.

Builds the tau2 collection, cuts it, embeds every chunk with a real embedding
model, searches both arms, blends, reranks, and asks a real language model to answer
using only the passages it was shown.

The embedding cache under .cache/vectors is keyed on model identity, so asking a
second question about the same corpus re-embeds nothing. The first run, or any run
that changes the model or chunking, pays in full.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from atlas.config import Settings
from atlas.contracts import Question, QuestionName
from atlas.corpus.chunk import cut
from atlas.corpus.registry import get_source
from atlas.models.embed import CachedEmbedder
from atlas.models.pricing import cost_usd
from atlas.models.providers import OpenAIEmbedder, selected_provider
from atlas.pipeline import PreparedCorpus

ROOT = Path(__file__).resolve().parents[1]


def _settings() -> Settings:
    return Settings()


def main() -> int:
    parser = argparse.ArgumentParser()
    # Rejoined rather than one quoted argument, so a bare `just ask how do i ...`
    # works without quotes; a no-op once the justfile forwards a quoted "$@".
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument("--list", action="store_true", help="show sample questions and exit")
    parser.add_argument("--dry-run", action="store_true", help="retrieve only, no generation call")
    parser.add_argument("--shown", type=int, default=None, help="passages to show the model")
    args = parser.parse_args()

    settings = _settings()
    source = get_source()

    if args.list:
        print("Sample tau2 tasks (each names the documents an agent must consult):\n")
        for q in source.questions()[:6]:
            wording = " ".join(q.text.split())[:96]
            print(f"  {q.name}\n    {wording}...\n    -> requires {', '.join(q.required)}")
        return 0

    if not args.question:
        parser.error("give a question, or --list to see some")

    asked = " ".join(args.question)
    collection = source(0)
    chunks = cut(collection.documents, settings.chunk)
    question = Question(
        name=QuestionName("q_ask"), text=asked, kind="lookup", required=(),
    )

    print(f"asked:      {asked!r}")
    print(f"corpus:     {len(collection.documents)} documents -> {len(chunks)} chunks")
    print(f"provider:   {selected_provider()} (generation)   openai (embeddings)")
    print(f"models:     {settings.answer.model} / {settings.embedding.model_name}\n")

    if args.shown is not None:
        settings = replace(settings, answer=replace(settings.answer, max_shown=args.shown))

    embedder = OpenAIEmbedder(settings.embedding)

    started = time.perf_counter()
    prepared = PreparedCorpus.build(
        settings, chunks,
        embedder=CachedEmbedder(embedder, ROOT / ".cache/vectors"),
    )
    embedding_cost = cost_usd(settings.embedding.model_name, embedder.input_tokens, 0)
    print(f"embedded:   {len(chunks)} chunks in {time.perf_counter() - started:.1f}s "
          f"({embedder.input_tokens:,} new tokens, ${embedding_cost:.4f})")

    record = prepared.ask(question, generate=not args.dry_run)

    print(f"retrieved:  vector {len(record.vector.hits)}, keyword {len(record.keyword.hits)}, "
          f"fused {len(record.fused.hits)}, reranked {len(record.reranked.hits)}")
    # The passthrough reranker reports 0.0 for everything; naming the ordering
    # stage instead avoids implying a score that was never computed.
    scored = prepared.reranker.name != "passthrough"
    ordered_by = prepared.reranker.name if scored else "rank fusion (reranking is off)"
    print(f"\ntop {min(3, len(record.reranked.hits))} passages that would be shown, ordered by {ordered_by}:")
    for hit in record.reranked.hits[:3]:
        snippet = " ".join(prepared.by_name[hit.chunk].text.split())[:110]
        score = f"  ({hit.score:.4f})" if scored else ""
        print(f"  {hit.rank}. {hit.chunk}{score}\n     {snippet}...")

    answer = record.answer
    if answer is None:
        print("\n--dry-run: retrieval only, no generation call was made.")
        print(f"spent:      ${embedding_cost:.4f} on embeddings")
        return 0

    print(f"\n{'=' * 72}\n{answer.text}\n{'=' * 72}")
    print(f"outcome:    {answer.outcome}")
    print(f"cited:      {', '.join(answer.cited) or '(nothing)'}")
    if answer.violations:
        print(f"violations: {', '.join(answer.violations)}")
    print(f"tokens:     {answer.usage.input_tokens:,} in / {answer.usage.output_tokens:,} out")
    print(f"cost:       ${answer.usage.cost_usd:.4f} generation "
          f"+ ${embedding_cost:.4f} embedding")
    print(f"elapsed:    {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
