# Atlas

*A RAG customer support chatbot for a fictional bank, taken apart into stages that can each be measured on their own, and the evaluation and testing apparatus that decides whether any of it works.*

## What this is

A retrieval augmented chatbot answers a customer's banking question from a document set. That is the easy half, and most of a working one fits in an afternoon. The hard half is knowing whether it is any good, and the usual answer is a dashboard of metrics that nobody has checked.

So this repository is built the other way round. The chatbot is a plain sequence of stages rather than a framework, because a stage you cannot run on its own is a stage you cannot check. Each stage is measured on its own. Every number it publishes comes with the best score that was actually available and with what a system that ignores the question would score. And the popular eval metrics are checked too, which is the finding below: two of the four widely used scorers pass every answer this repository's own judge failed.

### The parts of the chatbot

| Stage | What it does | Where |
|---|---|---|
| Corpus and gold set | 698 vendored banking documents and 97 customer tasks, each naming the documents a correct answer must consult | `src/atlas/corpus/tau2.py` |
| Chunking | three strategies: fixed token windows, sentence packing, and recursive paragraph then sentence | `src/atlas/corpus/chunk.py` |
| Embedding | real vectors at a requested width, cached by full model identity so a swap cannot serve stale ones | `src/atlas/models/embed.py` |
| Dense search | cosine similarity against a matrix held in memory | `src/atlas/retrieval/dense.py` |
| Keyword search | BM25 written out rather than imported, beside the set overlap baseline it replaced | `src/atlas/retrieval/sparse.py` |
| Fusion | reciprocal rank fusion, combining two arms by position rather than by score | `src/atlas/retrieval/fuse.py` |
| Reranking | a real cross encoder, off by default because the measurements say it hurts here | `src/atlas/retrieval/rerank.py` |
| Answer writing | structured output, with every citation checked against what the model was actually shown | `src/atlas/models/generate.py` |
| Judging | a versioned scoring guide and prompt, verdict read by field rather than parsed from prose | `evals/judge.py` |

### How each part is evaluated

| What is done | Why it is done | Where |
|---|---|---|
| Marking against an answer key, with the top mark shown beside it | The answer to a question can need twenty one documents when only ten are returned, so the best score available here is 0.604. Without that number on the page, a score of 0.226 reads as a broken system. | `evals/ir_metrics.py` |
| Comparing against a system that ignores the question | A score means nothing until you know what doing no work scores. Every published number carries that floor, and the code refuses to publish one without it. | `evals/baselines.py` |
| Asking whether a difference is real | Two versions are run over the same questions, and the gap is reported with a range around it, so a change that is really luck cannot be sold as an improvement. | `evals/stats.py` |
| Adversaries: breaking the system on purpose | Seven systems with a known fault. Each needs a detector that catches it, and where no measurement does, that gap is written down rather than left quiet. | `evals/adversaries.py` |
| Checking rules against thousands of generated inputs | Rules that must hold whatever the input, tested on inputs a machine invents rather than the few a person thinks of. | `tests/pipeline/test_properties.py` |
| Metamorphic testing | Change the input in a way that should change the answer in a known direction, and check that it does. It is how you test something when nobody knows the single right answer. | `tests/pipeline/test_relations.py` |
| Prompt injection: feeding it a document written to attack it | A document that stops informing the model and starts giving it orders, which every ranking score reads as one slightly irrelevant result. | `evals/injection.py` |
| Judging answers with a model, then measuring the judge's noise floor | How often the judge gives the same answer two different verdicts. Every threshold built on it is a multiple of that number. | `evals/calibration.py` |
| Comparing the judge against labelled verdicts | Whether the judge and a second rater agree more than two raters would agree by chance alone. That is what chance corrected agreement means. The apparatus is built and the labelling pass has not happened: every row in `data/labels.jsonl` was written by a model, and no human has graded an answer here yet. | `evals/labels.py` |
| The contrast: checking the popular metrics themselves | Whether well known eval scorers agree with the judge on the same answers. On this corpus they mostly do not. | `evals/deepeval_suite.py` |
| Naming every run after its settings | Change any setting and the name changes, so two runs that were not the same experiment can never be filed as one. | `src/atlas/config.py` |
| Recording every step of every run | Which question produced which answer, what each stage cost, and how long it took. | `src/atlas/trace.py` |

The rest of this page is the evidence that the apparatus does something.

## Your RAG system is faithful and lying

Here is an answer. Every word of it is backed by the passage the system found. The faithfulness score is a perfect 1.000. The answer is still wrong, and it costs the customer money. The score had no way to catch that, because the document that proves the answer wrong was never handed to the scorer.

That is measured here, not claimed. Across the recorded run, deepeval's Faithfulness passes **all 97 answers**. Its lowest score on any of them is 0.600, against a cut point of 0.50. That includes **all 20 answers this repository's own judge failed**. Answer Relevancy does the same thing, 97 passes out of 97.

A scorer that never says no cannot tell a good answer from a bad one, and no agreement figure can rescue it. Chance corrected agreement, the kappa below, cannot even be worked out when a column never varies, so the table marks those rows rather than printing a number that would read as a measurement. The raw agreement of 0.794 that looks like a result is just the 77 answers the judge passed anyway.

It is not one bad scorer. All four come back the same way:

| | average score | passed | kappa, agreement beyond chance | 95% interval |
|---|---|---|---|---|
| Answer Relevancy | 0.975 | 97 of 97 | 0.000 | [+0.000, +0.000] ⚠ |
| Faithfulness | 0.978 | 97 of 97 | 0.000 | [+0.000, +0.000] ⚠ |
| Contextual Relevancy | 0.637 | 77 of 97 | −0.134 | [−0.266, +0.038] ⚠ |
| refusal_correctness (GEval) | 0.884 | 91 of 97 | +0.065 | [−0.099, +0.265] ⚠ |

Every row carries a warning mark, and not one interval clears zero. Two of them cannot be read at all, because a scorer that passes everything has no pattern to compare against. `just report --render --grain contrast` prints this table straight from a file in the repository, no key needed.

You already know how to catch this. You have been doing it for years under different names. This repository is the translation table.

## The translation table

| You already do this | In AI eval it is called | Why the mapping is real |
|---|---|---|
| Test fixture / test data | Golden dataset | A fixed, versioned input set you run against repeatedly |
| Assertion | Reference based metric (recall@k, exact match) | Both check what happened against what should have happened, by rule rather than by opinion |
| Test oracle | Gold set | The thing that tells you what correct looks like, fixed before the system runs and not written by the system's author |
| Flaky test triage | Noise floor measurement | Both ask whether something really broke or just varies, and answer by running it again |
| API contract testing | Artifact and trace schema contracts | An agreed shape, and rules for what happens when it changes |
| Property based testing | Metric invariant testing | Check rules on inputs a machine invents, not examples picked by hand |
| Metamorphic testing | Metamorphic testing | Same name, same idea. Compare two runs when nobody knows the single right answer |
| Chaos engineering | Fault injection | Cause the failure on purpose, then check the system handles it |
| Mutation testing | Every metric ships an adversary or a written exemption | Break the system on purpose. A metric no broken system catches has to say in writing why not |
| CI release gate | Eval gate | Both stop a merge on a number |
| Penetration testing | Prompt injection and corpus poisoning | A new way in: the documents you load and the text you retrieve. The newest row here, and the one still only half covered (see below) |

Almost none of this is a new skill. Putting the old name next to the new one is what lets a tester see that they already know the ground.

The last row is the newest and the only one with an asterisk, so it is worth saying exactly what stands behind it.

**A mitigation.** Both prompts fence the text they do not control. The retrieved passages, the customer's question, and the answer being judged are each wrapped in delimiters and described to the model as evidence to read, never as orders to follow. The fence around the question was missing at first, and this repository walked straight into what it prevents. On this corpus the question is the benchmark's own script for playing the customer, so the writer was handed a character to act and a line to open with, both addressed to a different model, and it did as it was told. Thirty nine of ninety seven answers came back as the customer's own words.

**An adversary, and the detector that catches it.** `poisoned_corpus` is the seventh adversary: a document that stops talking to the customer and starts talking to the model, closing the passage fence early and telling it to pass every answer, placed first for every question. The injection detector catches it and nothing else does. Every ranking score treats it as one slightly irrelevant result, a mild and unremarkable fault, when what actually happened is that a document told the judge to stop judging.

The detector's patterns were tuned against the real documents rather than against imagination. An early version matched `you must respond` and fired on a genuine document about responding to a dispute within sixty days. A detector that fires on the healthy control is worse than none, because it teaches whoever inherits it to stop reading the output. What ships matches none of the 1010 real chunks, and a test keeps it that way.

**Not proof that the fence holds.** Catching poisoned text before it reaches a model, and a model ignoring that text once it arrives, are two different claims. Only the first is made here. That is the asterisk, and it matters most if you take up the invitation under "Making it yours" and point this at documents you did not write.

## Three paths

| Time | Path | What you leave with |
|---|---|---|
| 5 minutes | Run `just ask "how do I dispute a card transaction?"` | You have watched the whole pipeline answer a real question against real models, for about a cent |
| 1 hour | Read [the decision records](docs/), then `src/atlas/corpus/gold.py` | You can measure retrieval with arithmetic instead of opinion |
| 5 hours | Read `evals/`, starting at `adversaries.py` | You can build and defend an eval suite, and you know what yours cannot catch |

## Getting started

You need [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just). Python 3.12+ is fetched by uv itself. CI runs on Linux (`ubuntu-latest`) only; it is developed on macOS and nothing here is specific to Linux, but Windows is untested.

```
git clone https://github.com/mariusargatu/Atlas && cd Atlas
uv sync
cp .env.example .env     # then put an OpenAI key in it
just ask "how do I dispute a card transaction?"
```

`just test` also downloads a 23 MB cross encoder from huggingface.co the first time it runs, and caches it after that. It is the reranker the [decision record](docs/the-reranker.md) argues against keeping on by default, and two tests score against the real thing rather than a stub. No token is needed for it.

The key is not optional. Every command that runs the pipeline embeds against a real model and generates with a real one, including the checks that block a merge, because a check that cannot tell an embedding model from a hash function is not checking retrieval. That was not always true here, and [the decision record](docs/why-every-run-uses-real-models.md) says what changed and what it cost.

| Command | What it does | Roughly |
|---|---|---|
| `just ask "..."` | one question through every stage, with `--dry-run` to stop before generation | ~$0.009 warm, ~$0.013 cold |
| `just report` | every metric per search arm over the 97 tau2 tasks, against two baselines that ignore the question, recorded to `data/results/runs.jsonl` | free warm, ~$0.004 cold |
| `just report --render` | print a recorded run's table. **No key needed** | free |
| `just report --render --grain contrast` | the recorded judge beside deepeval's four scorers, with the agreement between them. **No key needed, and the shortest route to the claim this page opens with** | free |
| `just report --contrast` | the same answers scored by deepeval too, with the agreement between the two | dollars, not cents |
| `just compare A.json` | is the gap between two configurations real? paired bootstrap over the same 97 questions, against the defaults or `--to B.json` | free warm |
| `just test` | the suite, which embeds the corpus a handful of times, and downloads a 23 MB cross encoder once | a few cents |
| `just label --sample 25` | walk 25 unlabelled answers drawn at random, showing each with its passages and the scoring guide, and record your verdicts into `data/labels.jsonl` | free |
| `just label --report` | chance corrected agreement between those verdicts and the judge's, with its interval | free |
| `just noise-floor` | how often the judge changes its own verdict on the same answer, written to `data/noise_floor.json`. Ten verdicts per question, so start with `--limit` | ~$0.055 a question |
| `just lint` / `just types` | ruff, then mypy over `src/atlas`, `evals` and `scripts` | free, reaches nothing |
| `uv run pytest tests/meta/test_repo_contract.py` | the repo's promises about itself: links, CI tiers, test-count floors | needs a key, like the rest of the suite |
| `just up` / `just down` | local Langfuse for traces, genuinely optional. With it running, every `just report` is also a [dataset run](docs/recording-runs.md) | free |

Embedding all 1010 chunks costs $0.0036: 180,541 tokens at $0.02 per million. Generation is the rest of the bill, and it is the larger half. Over the 97 questions in the recorded judged run it averaged $0.0093 each ($0.9069 of answers, $0.6774 of verdicts, $1.5843 in all; `just report --render --grain contrast`).

That divisor is 97, the whole question set, and it used to be the number of questions the writer answered rather than refused. That was wrong twice over: the arithmetic did not match the figure printed beside it, and the divisor itself was not a number `data/results/runs.jsonl` carries, so a reader could not check it even after spotting the problem. Every figure on this page is meant to be recomputable from a committed file.

The contrast is the exception and says so under its own table: those four scorers spend inside deepeval's client, which never reaches this repository's token accounting, so the $1.5843 above covers the answers and the verdicts only.

## The one command

```
just ask "how do I dispute a card transaction?"
```

It loads the vendored tau2 collection, cuts it, embeds every fragment with a real embedding model, searches both arms, blends, and asks a real language model to answer using only the passages it was shown, printing what it retrieved, what it cited and what it spent. It refuses to start without a key rather than embedding the corpus and discovering half way through that it cannot finish.

`--dry-run` stops after retrieval, so this costs nothing and needs no generation call:

```
$ just ask "how do I dispute a card transaction?" --dry-run
asked:      'how do I dispute a card transaction?'
corpus:     698 documents -> 1010 chunks
provider:   openai (generation)   openai (embeddings)
models:     gpt-5.6-luna / text-embedding-3-small

embedded:   1010 chunks in 0.3s (0 new tokens, $0.0000)
retrieved:  vector 50, keyword 50, fused 70, reranked 10

top 3 passages that would be shown, ordered by rank fusion (reranking is off):
  1. doc_credit_cards_credit_cards_(general)_014#0000
     # Filing a Credit Card Transaction Dispute (Internal) ...
  2. doc_bank_accounts_bank_accounts_(general)_031#0000
     # Internal: Filing a Debit Card Transaction Dispute ...
  3. doc_credit_cards_credit_cards_(general)_018#0000
     # How to Dispute a Credit Card Transaction ...

--dry-run: retrieval only, no generation call was made.
spent:      $0.0000 on embeddings
```

Drop `--dry-run` and it goes on to generate and print the answer, its citations, and the dollar cost, for about a cent.

There was a `just demo` here that scored a slice of twenty tasks and wrote a machine readable summary. It is gone: it existed to demonstrate a pipeline that could not otherwise be run for real, and everything it did is now either a live command above or a check in the suite. A demonstration that has to be kept working alongside the thing it demonstrates is a second implementation.

## How the pieces fit together

The documents are taken from Sierra Research's tau2-bench: 698 authored banking documents and 97 customer service tasks, each naming the documents an agent must consult. That list is the gold set, so the correct answer to every question was fixed by somebody else before this system ran, which is the whole reason for vendoring it.

The pipeline is a plain sequence of seven typed stages (`embed_query`, `vector`, `keyword`, `fuse`, `rerank`, `answer`, `judge`) rather than a framework, because a stage you cannot call alone is a stage you cannot evaluate on its own. Judging is the only optional one:

```
embed_query ─ vector ─┐
                      ├─ blend ─ rerank ─ answer ─ judge
              keyword ┘
```

`embed_query` is separate from `vector` because one is a network round trip to a paid endpoint and the other is a dot product against a matrix already in memory. Time them together, which is what you get if you wrap a retriever and read one number off it, and a warm embedding cache makes the dense arm look an order of magnitude cheaper than BM25. The comparison flips the moment the cache misses.

The repository is organised into two halves. One half repeats: the same cutting strategy, the same vectors, the same blend of two searches, given the same input, always produces the same ranked list. That half is asserted on and blocks a merge when it changes. The other half does not repeat: what a model writes and what a judge decides both vary, so that half cannot be asserted on and has to be measured with an interval and a noise floor instead.

Both halves of that are things you run. `just noise-floor` judges recorded answers several times each and writes the result to `data/noise_floor.json`. `just label` walks those same answers so you can record verdicts of your own, and the calibration code measures agreement between those and the judge's, corrected for chance. The five judge relations in the suite are multiples of the floor, carry the `reporting` marker, and are deselected by default.

The floor is measured and committed, and it is the most uncomfortable number here, so it is worth stating before you find it. Over 30 answers judged five times each, **the judge returned the same verdict on all five repeats for only half of them**. Fifteen of the thirty moved at all, and seven went two of five against their own majority.

**This page used to call that 50% the flip floor, and print `0.500`. It is not a flip rate, and the mistake is the kind this repository exists to catch.** What the file records per question is how many of the five repeats broke ranks. `1 - unanimity` answers "did *any* of five readings disagree", which rises with the number of repeats however steady the judge is; a perfectly calibrated 5% flipper looks worse at ten repeats than at five. What every relation here actually bounds is one second reading disagreeing with one first, which is a rate per **pair**: `m(5-m)/10` per question, averaged over the thirty. That is **0.247** on the shipped column and 0.273 on the other. Both come out of the same committed file, so neither needs a model to check:

```
uv run python3 -c "from evals.calibration import recorded_noise_floor as f; c = f().at_configured; print(c.unanimity_rate, c.flip_floor)"
```

The finding underneath survives the correction, and it is that the noise is spread rather than concentrated. A judge undone by a handful of genuinely ambiguous answers would be survivable, since you would name those questions and move on. Fifteen questions in thirty is not that. Roughly a quarter of second readings disagree with the first. So a single verdict from this judge still carries less information than a judged table implies, and everything built on one verdict per answer inherits that, including every judged figure on this page. The remedy is unchanged by the correction: average over `judge.repeats` rather than reading one verdict at a time, or replace the judge, before treating any bias measured against it as settled.

It was measured again after the answer stage was fixed and barely moved. The writer had been replying with the customer's own words on 39 of 97 questions, so it was reasonable to expect an unstable judge to have been partly a judge asked to grade text that was not an answer. It was not. Whatever makes this judge disagree with itself survives the answers getting better, which points at the judge rather than at what it reads. Because the default model rejects the sampling temperature, both columns run at one setting, so they are the same judge measured twice, and the gap between them, 0.247 against 0.273, is how precise this estimate is at n=30.

What the correction changes is whether the five judge relations mean anything. Each asserts `rate <= 2 * floor`. At 0.500 that bar is 1.000, a rate is a proportion, so not one of them could fail, and the fixture refused on exactly that ground rather than reporting four hollow passes — meaning those four bias mutations had never been executed by any assertion, in CI or out of it. At 0.247 the bar is **0.493**, which a genuinely biased judge can exceed. The fixture's `floor < 0.5` gate now passes instead of erroring, so the only remaining reason those five do not run is that nothing selects them: they still carry the `reporting` marker that `pyproject.toml` deselects, and nobody has paid for a run of them yet.

Nothing about the judge changed. We had been bounding it with the wrong statistic, and this page recommended the wrong statistic in bold — which is worth more as a demonstration than the original number was. The measurement was honest, committed and recomputable, and still the wrong measurement. Being able to recompute a number is not the same as the number answering the question you asked of it.

`data/labels.jsonl` is the other half: verdicts on individual answers, recorded beside the judge's own so a label still means something after the settings move on. Every row names its rater, and `just label --report` groups on that and refuses to pool two raters into one figure, because a kappa computed over a person's verdicts and a model's describes neither. `just label` walks the answers and shows you each one with its passages and the scoring guide, and adds yours.

**Nobody has done that yet, and this page used to imply otherwise.** All 25 rows in the file today carry `"rater": "claude-sonnet-5"`. They are a model grading another model's answers against the same guide, recorded against run `1a737014a9eb`, which the answer prompt has since superseded, and they pass 24 of the 25. A second opinion from a model that shares the judge's priors is not a ground truth, `just label --report` says so in as many words before it prints the figure, and the calibration code refuses to mix that rater with a human one. So there is no human agreement number here to quote, only the apparatus that would produce one. Confirm it for yourself:

```
python3 -c "import json;rows=[json.loads(l) for l in open('data/labels.jsonl')];print(len(rows), {r['rater'] for r in rows})"
```

## Where to go next

The [decision records](docs/) are the design documentation this repository keeps. Each one names a decision, what it costs, and what would have to change to reverse it. They and [the framework mapping](docs/frameworks.md) are the only prose here. Everything else that described the design has been folded into the code it describes, where a reader checking it has the code in front of them.

Then read in this order, which is the order the data moves:

| Where | What it answers |
|---|---|
| `src/atlas/contracts.py` | the types every stage passes, and the protocols behind each seam |
| `src/atlas/corpus/` | where documents come from, how they are cut, and what counts as a correct answer |
| `src/atlas/retrieval/` | two search arms, a blend, and a reranking stage |
| `src/atlas/pipeline.py` | how those compose into one record per question |
| `evals/` | the measuring apparatus. It reads the pipeline's types freely; the enforced rule is the reverse arrow, that `src/atlas` never imports `evals` |
| `evals/systems.py`, `evals/injection.py` | the seven deliberately broken systems and the poisoned document, split out of `adversaries.py` when it hit the 400 line cap. `adversaries.py` keeps the detectors and the table wiring each fault to the one measurement that notices it |

## Making it yours

The seam you want is `CollectionSource` in `src/atlas/contracts.py`. It takes a seed and returns a `Collection`, and that is the entire contract:

```python
class CollectionSource(Protocol):
    @property
    def name(self) -> str: ...
    def __call__(self, seed: int) -> Collection: ...
```

`name` is a property you can only read, not a plain `name: str`, which is the form this page used to print. A plain annotation makes the protocol member settable, so a frozen dataclass cannot satisfy it and mypy says `expected settable variable, got read-only attribute`. But it says that only once something statically claims to implement it, which is why the broken version passed the type checker for as long as nobody used it. The contracts file carries the same note.

`Tau2Source` is the reference implementation: the class is 79 lines and leans on a few module helpers above it. Write your own that reads your documents, and everything downstream works unchanged: chunking, both search arms, the fusion, the reranker, the gold resolver and every measurement, because all of them take a `Collection` and never ask where it came from.

Two caveats the protocol does not cover, and it is better to know them now than to discover them. A collection source returns documents only. Questions come from a separate call, so a new corpus has to supply its own `Question` list with the `required` documents that answer each one. And a source is resolved by name rather than by configuration. `src/atlas/corpus/registry.py` holds the table: `register_source` adds yours under its own `.name`, and `get_source` and `load` read it back, which is what `scripts/ask.py`, `scripts/compare.py`, `scripts/label.py`, `scripts/noise_floor.py`, `scripts/report.py` and `evals/validity.py` all now call instead of constructing `Tau2Source` themselves. This page used to say the corpus was named directly at six call sites and that swapping it meant editing all six; that was true until the registry existed. What is still true is the narrower half: the source name is not a field in `Settings`, so it does not feed `run_id`, and two runs over different corpora would be filed under the same name.

The one thing to get right is the gold set, and it is the one thing you cannot get from the code. `Question.required` names the documents a correct answer has to consult, which is the granularity tau2 supplies and therefore the granularity every number here has. A source that knows *where in a document* an answer sits could measure more, such as whether a chunk boundary cut a value in half, and that needs a gold set with character offsets in it, which this one does not have. Read [`docs/where-the-answers-come-from.md`](docs/where-the-answers-come-from.md) before you write one.

Then, in rough order of how much they change your numbers: how documents are split, which embedding model you use, how the two arms combine, and how the reranker is set. All of them live in `src/atlas/config.py`, all of them feed `Settings.run_id`, so two runs with different settings can never be confused for each other.

## Contributing

PRs are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) has the rules CI actually enforces: `src/atlas` never importing `evals`, and a prompt version bump on every prompt edit. It also explains why CI fails by design on a fork's own PR, since no repository secrets reach a workflow triggered from a fork, and a maintainer reruns the checks from a branch here before merging. Found a wrong number or a bug? Open an issue. Found something about the code putting *your* key or *your* data somewhere it shouldn't? Read [SECURITY.md](SECURITY.md) and open a private security advisory instead. Participation here follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licence

MIT, see [LICENSE](LICENSE). Two things here belong to somebody else, and each carries the upstream licence beside it word for word rather than only a mention in prose: the vendored tau2-bench data (Sierra Research, [`data/tau2/LICENSE`](data/tau2/LICENSE)) and `deploy/docker-compose.yml`, adapted from the compose file Langfuse publishes for running it yourself (Langfuse GmbH, [`deploy/LICENSE.langfuse`](deploy/LICENSE.langfuse)). [NOTICE](NOTICE) says what changed in each and why naming a copyright holder in prose is not on its own enough.
