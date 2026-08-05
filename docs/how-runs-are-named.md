# How runs are named

Every choice that changes what this system does lives in one settings tree that cannot be edited once it is built, and the name of a run is a short hash of that tree. Change a setting and the name changes with it, so two sets of numbers filed under one name came out of one configuration.

## One tree, two levels deep

The settings tree has eight groups: `chunk`, `embedding`, `search`, `sparse`, `fusion`, `rerank`, `answer` and `judge`. Each group is a small class of plain values. That is the whole shape, and there is no third level.

A few of the values, so you can see what kind of thing they are. `chunk.max_tokens` is 256, so a document is cut into pieces of at most 256 tokens. `search.vector_candidates` is 50, so the vector arm returns 50 chunks. `rerank.backend` is `"passthrough"`, so reranking is off by default and the reranked ranking is the fused one.

Prompt and scoring guide versions are settings too. `answer.prompt_version` must match the version at the top of `data/prompts/answer.md.j2`, and `judge` carries a `rubric_version` and a `prompt_version`. Editing a prompt changes what the system does, so it has to move the run's name.

## The name is the hash

The run id turns the tree into JSON with the keys in sorted order, then hashes that text with blake2s. It keeps six bytes and prints them as twelve hex characters. That is nearly all it is.

The exception is one line long and it is worth understanding, because without it this page's own table is an argument against ever adding a setting. Hashing the *whole* tree means a new field renames every run ever recorded: the store holds 18 rows, and about forty-five references to their ids are scattered across eleven pages in this folder, all of which would go stale at once. The table below is that cost, already paid — three columns of ids for one set of configurations, because editing a Jinja template moved `answer.prompt_version` twice. The repository had reached the point of declining to delete two provably inert fields on these grounds, which is a settings tree frozen by its own filing system.

So a field named in `_ADDED_AFTER_THE_STORE` is left out of the hash while it holds its default. A tree that does not use it hashes exactly as it did before the field existed, and `Settings()` still comes out `6818412cb6ec`. `sparse.stopwords` and `sparse.query_term_frequency` are the two entries today.

What that keeps is the direction that matters: two runs differing in any setting still get different ids, because a tree using the field and a tree not using it hash differently. What it gives up is the reverse claim — an id no longer tells you which fields existed when it was minted. That is the right trade for a repository that intends to keep adding settings, and it is why the exemption is a named list rather than a blanket "skip every default": each entry is a deliberate statement that no recorded run could have set that field, which is true exactly once, on the change that introduces it. Adding an existing field to that list would silently merge runs that really did differ, and a test pins `Settings().run_id` to catch precisely that.

The defaults today hash to `6818412cb6ec`. Eight alternative configurations ship in `data/settings/`, each one a small diff from the defaults. Pass one with `just report --settings data/settings/chunks-512.json` and the run id changes on its own.

| file | what it changes | run id | under prompt 3.0.0 | before that |
|---|---|---|---|---|
| *(the defaults)* | nothing | `6818412cb6ec` | `1a737014a9eb` | `f45992f36859` |
| `chunks-512.json` | `chunk.max_tokens` 512 | `c15bdd1c26c6` | `94dc05e7cd62` | `98ae55b5cbff` |
| `chunks-512-overlap-64.json` | fixed chunker, 512 with 64 overlap | `844a45653c3a` | `414ee40e763d` | `a9b3188dccb5` |
| `embedding-1536.json` | `embedding.size` 1536 | `a1d885548792` | `7dd95a284364` | `a7012a1674ea` |
| `embedding-large-3072.json` | `text-embedding-3-large` at 3072 | `e6e8907ab489` | `f384a657401b` | `d633e6bd8ebc` |
| `naive-keyword.json` | `sparse.scorer` set to `"overlap"` | `a352adaf2b51` | `9667f8170801` | `898a00113886` |
| `reranked.json` | `rerank.backend` set to `"onnx"` | `f50ceb2b7efd` | `45a977139549` | `40560d20707d` |

Every id in that table has moved twice, and the two right hand columns are why this page exists. None of these seven files changed either time. The prompts did, and both version fields are in the tree every one of these hashes is taken over, so a prompt edit renames every configuration at once. That is the mechanism working rather than failing: prompts that ask differently produce runs that cannot be compared to the old ones, and the name is what says so.

The first move was a fence around the retrieved passages, in both prompts, plus the answer's declared outcome and citation list reaching the judge, which rule 2 of the scoring guide grades on and the older body never rendered.

The second move is `data/prompts/answer.md.j2` going 3.0.0 to 4.0.0, and it is worth reading as a lesson rather than a version bump. The passages were fenced and the question was not, and on this corpus the question is tau2's own user simulator script: a persona, verification details, tool instructions, and a quoted line the customer is told to open with. So the writer was handed instructions addressed to a different model, immediately after being told that only the text outside the passage block was instruction. It followed them. Thirty nine of the ninety seven answers in the run under 3.0.0 open with text copied verbatim from the customer's own message, twenty of those cite nothing at all, and the judge passed thirty of the thirty nine. Fencing the question fixes it: of eight answers that echoed under 3.0.0, none echoes under 4.0.0.

That is retrospective, and the store holds every side of it. `f45992f36859` and `1a737014a9eb` keep their rows, because those rows are the evidence for what the older prompts did and deleting them would be deleting a measurement. Retrieval figures are identical across all three ids, since retrieval never touches either prompt. The judged and contrast figures are not, and that is the whole reason the ids differ.

Recording the judged half again is the expensive half of the exercise: 97 answers and 97 verdicts, then four deepeval scorers over each answer on top. It came to $1.58 of accounted spend, $0.91 of answers and $0.68 of verdicts, plus whatever the contrast cost inside deepeval's own client, which this repository's token accounting never sees.

Keeping choices as settings is what makes a claim checkable:

```
just compare data/settings/naive-keyword.json --metric nDCG@k --arm keyword
```

scores plain set overlap against BM25 on the same 97 questions and prints the gap with an interval around it: `difference -0.0981`, interval `[-0.1451, -0.0503]`, which reads as BM25 being 0.098 ahead with the interval clear of zero. nDCG rewards a ranking for putting the correct chunks near the top, 1.0 being a perfect order.

Both flags carry weight and this page used to omit them, quoting `+0.098 [+0.050, +0.145]` against the bare command. `--arm` defaults to `reranked`, which on the defaults is the fused ranking untouched, so the bare command measures a scorer change through an arm that change barely reaches and prints `-0.0385`. The sign is the other half: the comparison reports after minus before, and the file you name is the side under test, so the old scorer being worse shows as a negative number. [The keyword search](the-keyword-search.md) works through the same comparison at length.

Loading settings refuses keys no field matches. Write `"max_token"` instead of `"max_tokens"` and you get `ChunkSettings has no field(s) ['max_token']`. Without that a typo is ignored, the default applies, and the run reports a configuration it was never given.

## What the hash does not cover

The hash describes the system under test, not what you asked of it. Three things you ask for on the command line never reach the tree: the metric depth `--k`, whether the run wrote answers and judged them, and whether it ran `--contrast`, which scores those answers again with the deepeval library so we can see where two scorers disagree.

We learned this the hard way. With only the run id as the key, a retrieval run and a run that also wrote and judged answers landed in one bucket, and the second hid the first. A `--contrast` run got the same key as a plain `--judge` run, so on a fresh clone our most expensive command bought 97 answers, 97 verdicts and 388 deepeval scores, printed the table, then recorded none.

The fix is a wider key: a recorded row is keyed on the run id, `k`, whether answers were written, whether they were judged, and whether a contrast ran. Those last four describe the question you asked, not the system you asked it of, which is why they belong there and not in the hash. The run store holds eighteen rows over fifteen distinct run ids today. `f45992f36859` appears three times, once retrieval only, once with answers written and judged, once with the contrast too, and `1a737014a9eb` twice, retrieval and contrast. All eighteen keys differ, so `just report --render --all` prints all eighteen.

The same id groups things outside the file too. When Langfuse is configured and the whole question set runs, the run appears in the dashboard under its run id, so the row that scored a number and the run you are looking at are the same string. A hash is not a description, though, so every row also carries the whole settings tree beside its numbers, and `just report --render` prints that tree under the table. That means a row can reproduce the name it was filed under: hash the settings on any of the eighteen rows and you get the id already on it. We checked all eighteen, and that includes every row recorded before either prompt version moved. They hash again to the ids they carry, which is the mechanism working rather than a discrepancy.

## Where the tree does not reach

Some behaviour is still a literal in the code, and changing it changes results without changing any run id. The ones we know about:

- The reranker cuts any query and chunk pair longer than 512 tokens down to fit before the cross encoder sees it.
- The keyword arm's token pattern decides what counts as a word at all. Its stopword list used to sit here too, and no longer does: `sparse.stopwords` names it, so the ablation moves the run id. It was worth putting in the tree rather than listing here, because removing the list costs **0.1562 nDCG@10** on the keyword arm, against the 0.0982 the `sparse.scorer` field beside it is worth. The largest lever in that arm was the one outside the identity, which is the argument for this whole section being short. Its sibling `sparse.query_term_frequency` arrived the same way.
- The sentence and paragraph patterns decide where two of the three chunkers may cut.
- The Anthropic client always sends `max_tokens=4096` and a thinking effort of `"medium"`. Neither is in the tree, so switching provider changes what the run does without changing what it is called.
- It sends no temperature either, because the models it targets reject the parameter. That used to mean `answer.randomness` silently did nothing under `MODEL_PROVIDER=anthropic`, a setting that moved the run id and reached no model, which is the same defect `rerank.enabled` was deleted for. It now raises instead of being dropped.
- **And the default OpenAI model does the same thing.** `gpt-5.6-luna` rejects `temperature`, so the client catches the 400 and retries without it. `answer.randomness` and the judge's randomness are therefore inert on the shipped configuration too, while still sitting in the run id and printing under every table. This was not known until the noise floor script went looking: a noise floor is two columns differing only in that value, and the first attempt measured two identical columns.

  The consequence is worth stating plainly rather than filing as a caveat. The floor built from two columns, which this repository was designed around, cannot be measured on its own default model, because there is only one sampling setting to measure. What is still measurable, and is what `data/noise_floor.json` now holds, is how much the result varies from one repeat to the next, because a hosted model does not repeat itself exactly even when nothing is varied. `just noise-floor --randomness 0.0` is that measurement, and the runner refuses any other value on a model that drops it rather than recording two columns that were never different.

One setting has the opposite problem. `embedding.normalise` cannot change a ranking, because the vector search divides by both lengths and is therefore cosine similarity, which does not care about scale. Flipping it files a second experiment that does identical work. We kept it as a trade: deleting a field rehashes every tree, and then no recorded row can reproduce its own name.

`judge.repeats` is a softer case. Nothing reads it today, because the noise floor takes its repeat count as a direct argument and no command calls that path yet. It stays in the hash on purpose: a floor measured over five repeats is not the same measurement as one measured over a single pass, so two runs that differ in it really are two experiments.

Prompt versions are held to their files by the tests, not at run time: edit `data/prompts/answer.md.j2` without moving `answer.prompt_version` and the run keeps the old id until `just test` runs.

## What we got wrong

Four settings were deleted because they moved the run id and changed nothing that ran. `rerank.enabled` was read by nothing, since the pipeline always reranks and `rerank.backend` is how you turn it off. `answer.refusal_allowed` described an intention the prompt enforces and the code never did. `collection_seed` fed a corpus that is a fixed set of files from somebody else and is never regenerated. A `profile` field split runs into the ones allowed to reach a real model and the ones that were not, and every run uses real models now. All four had the defect to watch for here: two runs filed as different experiments while doing identical work.

## What this costs

Adding a setting is more than one edit. The field and its default are one line in a dataclass, and the group list needs an entry only for a whole new group, but the stage code that reads the field is a second edit nothing writes for you. Quick experiments are less quick too: to change a value you write a small JSON file and pass `--settings`, or replace the field in a script. And `just ask` cannot be reconfigured at all: it builds the defaults, with `--shown` its one override.

We think both are worth it. A value you type into a terminal once is the change that never reaches a commit, and a run whose configuration nobody wrote down is a number nobody can check.
