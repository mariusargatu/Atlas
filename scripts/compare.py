"""Is the gap between two configurations a result, or is it noise?

    just compare data/settings/naive-keyword.json                 # against the defaults
    just compare data/settings/naive-keyword.json --to data/settings/reranked.json
    just compare --metric recall@k --k 10 data/settings/chunks-512.json

Paired: the two configurations are scored on the same 97 questions, so variance
they share cancels instead of being counted twice.
`evals.stats.smallest_resolvable_difference` is the unpaired worst-case bar; a paired
interval is usually far tighter, which is how a real difference can resolve well under it.

Free against a warm `.cache/vectors`, and about $0.004 cold. Retrieval only: nothing
here generates or judges.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from atlas.config import Settings, load_settings
from atlas.contracts import Question, QuestionName
from atlas.corpus.chunk import cut
from atlas.corpus.gold import GoldIndex
from atlas.corpus.registry import load
from atlas.models.embed import CachedEmbedder
from atlas.models.pricing import cost_usd
from atlas.models.providers import OpenAIEmbedder
from atlas.pipeline import PreparedCorpus
from evals.report import rankings, scored_retrieval
from evals.stats import paired_comparison, smallest_resolvable_difference

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "vectors"
METRICS = ("recall@k", "precision@k", "MRR@k", "nDCG@k", "success@k")


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("settings", help="a settings file: the configuration under test")
    parser.add_argument("--to", default=None,
                        help="what to compare it against; the defaults if omitted")
    parser.add_argument("--metric", default="nDCG@k", choices=METRICS,
                        help="which metric to pair on")
    parser.add_argument("--arm", default="reranked",
                        choices=("vector", "keyword", "fused", "reranked"),
                        help="which arm's ranking to score")
    parser.add_argument("--k", type=int, default=10, help="metric depth")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--limit", type=int, default=None, help="first N questions")
    return parser.parse_args()


def _score_every_question(
    settings: Settings, questions: list[Question], arm: str, metric: str, k: int
) -> tuple[dict[QuestionName, float], float]:
    """One number per question, plus what the embeddings cost this process."""
    collection, _ = load()
    chunks = cut(collection.documents, settings.chunk)
    embedder = OpenAIEmbedder(settings.embedding)
    prepared = PreparedCorpus.build(settings, chunks, embedder=CachedEmbedder(embedder, CACHE))
    gold = GoldIndex.build(prepared.chunks)

    scored: dict[QuestionName, float] = {}
    for question in questions:
        record = prepared.ask(question, generate=False)
        correct = gold.correct(question)
        ranked = rankings(record).get(arm, ())
        bounded = scored_retrieval(ranked, correct, k)[metric]
        # NaN is skipped rather than paired: paired_comparison requires both sides
        # to score the same questions.
        if bounded.value == bounded.value:
            scored[question.name] = bounded.value
    return scored, cost_usd(settings.embedding.model_name, embedder.input_tokens, 0)


def main() -> int:
    args = _parse()
    under_test = load_settings(args.settings)
    baseline = load_settings(args.to) if args.to else Settings()
    _, questions = load()
    if args.limit:
        questions = questions[: args.limit]

    print(f"comparing on {args.metric} of the {args.arm} arm at k={args.k}, "
          f"over {len(questions)} questions\n")
    print(f"  baseline    {args.to or 'defaults'}  ({baseline.run_id})")
    print(f"  under test  {args.settings}  ({under_test.run_id})\n")

    before, cost_a = _score_every_question(baseline, list(questions), args.arm, args.metric, args.k)
    after, cost_b = _score_every_question(under_test, list(questions), args.arm, args.metric, args.k)

    shared = before.keys() & after.keys()
    before = {name: before[name] for name in shared}
    after = {name: after[name] for name in shared}
    comparison = paired_comparison(before, after, args.seed)

    mean_before = statistics.fmean(before.values())
    mean_after = statistics.fmean(after.values())
    resolvable = smallest_resolvable_difference(len(shared))
    # An interval straddling zero means the sign isn't established, which is
    # weaker than "the two are the same".
    resolves = comparison.low > 0.0 or comparison.high < 0.0

    print(f"  baseline    {mean_before:.4f}")
    print(f"  under test  {mean_after:.4f}")
    print(f"  difference  {mean_after - mean_before:+.4f}   "
          f"95% paired interval [{comparison.low:+.4f}, {comparison.high:+.4f}]\n")
    if resolves:
        print(f"  RESOLVED: the interval excludes zero over {comparison.questions} questions.")
    else:
        print(f"  NOT RESOLVED: the interval includes zero over {comparison.questions} "
              "questions, so the sign of this difference is not established. That is not "
              "the same as the two being equal.")
    print(f"\n  For contrast, the unpaired worst-case bar at {len(shared)} questions is "
          f"{resolvable:.3f}. Pairing removes the variance the two configurations share, "
          "which is why a difference well under that bar can still resolve here.")
    spent = cost_a + cost_b
    print(f"\n  Spent ${spent:.4f} on embeddings"
          + (" (everything else was already in .cache/vectors)." if spent == 0 else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
