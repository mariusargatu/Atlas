# Choosing models

Four model names are settings in this repository. One model turns text into vectors. One can reorder a list of chunks before an answer is written. One writes the answer. One grades it. No function picks any of them: all four are fields in [the settings tree](../src/atlas/config.py), so changing a model is a settings edit.

| setting | what it picks | default |
|---|---|---|
| `embedding.model_name` | the model that turns text into vectors | `text-embedding-3-small` |
| `rerank.model_name` | the cross encoder that reorders a candidate list | `Xenova/ms-marco-MiniLM-L-6-v2` |
| `answer.model` | the model that writes the answer | `gpt-5.6-luna` |
| `judge.model` | the model that grades the answer | `gpt-5.6-luna` |

The rerank name is a setting for a stage that is switched off. `rerank.backend` defaults to `passthrough`, so out of the box nothing reorders anything and the reranked row of a table copies the fused row above it. [The reranker](the-reranker.md) explains why.

## What this buys

**A swap is a small file.** All of `data/settings/embedding-large-3072.json` is one `embedding` block naming `text-embedding-3-large` and a `size` of 3072. Nothing else.

**A swap becomes a separate experiment.** All four fields feed the short hash that names a run. Change any one of them and the run is recorded under a new name instead of landing on top of the old one. See [How runs are named](how-runs-are-named.md).

**Every table says which models made it.** `just report --render` needs no API key, and prints the settings block under the numbers with all four names in it.

**The vector cache knows which model made each vector.** The cache key hashes every embedding setting, the model name included, alongside the text. If it hashed only the text, a swap would quietly get the old vectors.

## A swap, measured

The point of all this is that swapping and remeasuring is a command, not a project. `just compare` scores two settings files on the same 97 questions, free once the cache is warm:

```
just compare data/settings/embedding-large-3072.json --arm vector

comparing on nDCG@k of the vector arm at k=10, over 97 questions
  baseline    defaults  (6818412cb6ec)
  under test  data/settings/embedding-large-3072.json  (e6e8907ab489)

  baseline    0.3547
  under test  0.3873
  difference  +0.0326   95% paired interval [-0.0074, +0.0719]
  NOT RESOLVED: the interval includes zero over 97 questions, so the sign of this
  difference is not established. That is not the same as the two being equal.
```

The bigger embedding model looks better on the vector arm alone. Run the same command with `--arm fused`, which scores the combined list, and it looks slightly worse: 0.4130 against a baseline of 0.4218, interval [-0.0332, +0.0147]. Both intervals cross zero, so over 97 questions we have not shown which way either difference goes. That is the honest result: telling you to use the larger model would quote a difference this set cannot settle.

## Where the settings do not reach

There are three of them. All three are on purpose, and all three refuse rather than guess.

**An unpriced model is refused.** [The price table](../src/atlas/models/pricing.py) holds published rates for nine models, and the code raises before a client is built if the name is not one of them. Ask for something else and you get:

```
no published rate for model 'gpt-5-nano', so every cost this run reported would
read $0.0000 while it spent real money.
```

Swapping to another model on that list is a settings edit. Swapping to anything else also means adding its rate to the table, which is code. We chose that on purpose: a wrong price printed next to real money spent is the kind of number that looks right and means nothing.

**The embedding client takes OpenAI embedding names only.** There is one embedding backend, so a name not starting with `text-embedding-` is refused, and another provider's embedder would need a new backend rather than a new setting.

**The provider is an environment variable.** `MODEL_PROVIDER` is read from the environment, which the justfile loads from `.env`, so it never reaches the run id. That would be a hole, except that the Claude client refuses any model name not starting with `claude`. A run on Anthropic has to name a claude model, and naming one moves the run id. `JUDGE_MODEL_PROVIDER` is a second, optional variable that overrides this for the judge specifically, defaulting to `MODEL_PROVIDER` when unset, so a run can answer on one provider and judge on the other without two full runs and a hand join.

## Two mistakes worth keeping

Both were the same bug, and both are why the refusals above exist. The embedder used to fall back to `text-embedding-3-small` for any name it did not recognise, so a run could report one model in its id, print it under every table, and bill another. It raises now, and a test holds it there.

The Claude client used to fall back to `claude-opus-5` the same way. Every run on that provider called Opus while every recorded model name said something else. It raises now too.

A third correction is this page. The record it replaces was titled "No model name is ever written into code", which is stronger than the code: nine names sit in the price table. The true and narrower statement is that no model name inside a function selects a model.

## A default is not a finding

`answer.model` and `judge.model` name the same model, and we have not shown that this is right. Nothing here compares two judge models: `--contrast` scores the same answers with deepeval's own scorers, but hands them the model named in `judge.model`, so that is one model twice. We hold no human labels at all, so we cannot say how often the judge agrees with a person. This page used to claim a single one, which was wrong in both directions: the one label it meant was a deleted placeholder, and `data/labels.jsonl` today holds 25 rows, every one of them graded by `claude-sonnet-5` rather than by anybody. Those are a model's second opinion on another model's answers against the same guide, which is a weaker thing than a label and cannot substitute for one. `JUDGE_MODEL_PROVIDER` makes an answer/judge provider split possible, which is the seam a same-provider self-preference concern needs, but running it and reading the result is still on somebody. Treat these four defaults as a starting point, not as advice.

## What it costs

You cannot tell which model runs by reading the rerank or generate code. That is what the extra step costs, and the settings block under every table is what pays it back. `just ask` prints a `models:` line for the same reason. Only the answer model has a test that fails if the default name turns up in the generation code; the other three names have no such guard.

Remeasuring is free for some swaps and not for others. Retrieval is free against a warm cache. Changing the embedding model is not: at the default chunk settings the corpus is 1010 chunks and 180,541 tokens, and embedding it again with `text-embedding-3-large` costs about $0.023 at its rate of $0.13 per million. A new answer or judge model is a paid run. See [Why every run uses real models](why-every-run-uses-real-models.md).
