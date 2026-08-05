# Recording runs

Every run writes its numbers twice. Once as a line in `data/results/runs.jsonl`, which is committed to git. Once as a Langfuse dataset run, which lives in a database on your own machine. `just report` writes both in one pass, joined by the run id. Duplicating a store is normally a mistake. We did it because each of the two is bad at the other's job.

## What each store is good at

The committed file is one JSON line per run, 5,708 to 7,051 bytes a row and 107,780 bytes over eighteen rows. Every change to it shows up in a git diff, it survives `docker compose down -v` (which deletes the local stack and its data), and `just report --render` prints any row from it with no API key and no server running, so a fork can compare its own numbers against ours.

What a row cannot tell you is *why* a number came out that way. Run `1a737014a9eb`, the default retrieval settings at k=10 under the answer prompt of the day, scored recall@10 of 0.226 and nDCG@10 of 0.422 on the fused ranking, against a recall ceiling of 0.604 and a baseline that ignores the question and still gets 0.107. (The defaults hash to `6818412cb6ec` now that the answer prompt has moved; its retrieval figures are identical, because retrieval reads no prompt.) Recall@10 is the share of a question's correct chunks that land in the top ten. nDCG@10 rewards a ranking for putting them near the top. And that is the end of it. The row cannot tell you which questions failed, or what the searcher saw when they did.

A Langfuse dataset run answers that and nothing else. Each question becomes a trace, the record of one request and every stage it passed through, and each score is attached to the trace that produced it. So "which questions went wrong" becomes a column you sort rather than an investigation. What it cannot do is exist without six containers: Postgres, ClickHouse, Redis, MinIO and Langfuse's own web and worker services, all declared in `deploy/docker-compose.yml`. The file is the archive and the dashboard is the microscope. Keeping one would cost us either the trace view or a number a reader can check without Docker.

## The join, and where it is not exact

A run id is a short code computed from every setting a run used, so two runs with the same settings get the same id ([how runs are named](how-runs-are-named.md) covers it). It is what a row in the committed file records, and it is passed through as the dataset run's name. Two things to watch.

A run id does not identify one row in the committed file. A row is keyed on five things: the run id, `k`, whether the run generated answers, whether it judged them, and whether a deepeval contrast ran. The store holds eighteen rows over fifteen run ids. `f45992f36859` appears three times at three grains (a grain is how far a run went: retrieval only, retrieval plus judged answers, and one that also ran the contrast) and `1a737014a9eb` twice. `just report --render --run-id f45992f36859` lists all three, and `--grain judge` picks one.

Fifteen ids for seven configurations, because all seven were recorded again when the prompt versions moved and every hash moved with them, and the defaults were then recorded a third time when `answer.prompt_version` went to 4.0.0. [How runs are named](how-runs-are-named.md) has the table of before and after, and the reason. The older ids keep their rows rather than being deleted: they are what the older prompts measured. Seven configurations is also the number of *distinct retrieval settings* in the store, and it does not grow when a prompt does, since no retrieval stage reads either prompt.

A run id is only the whole name when the run executed the whole question set. A partial run, from `--limit` or from resuming a ledger, is named `{run_id}-{pending}of{total}`, say `f45992f36859-5of97`, so a run whose means came from five questions cannot sit beside a run over 97 looking like its equal.

## What actually reaches the dashboard

The question set is uploaded as a dataset named `tau2-banking`, one item per question, keyed on the question's own name so running it again updates in place instead of doubling it.

Per question, we score the reranked ranking and send five numbers: recall, precision, MRR, nDCG and success, each at `k`. The 22 tasks where tau2 names primary documents get a sixth, graded nDCG. For the other 75 it is not a number at all, so we send nothing. A stand in would average into everything downstream. That is 507 scores over 97 traces for a full retrieval run. `--generate` adds a citation violation count per answer, `--judge` a verdict.

Per run, each metric goes out as three scores rather than one: the mean, the ceiling, and the mean as a fraction of the ceiling. A Langfuse score is a name, one value and a comment, so there is nowhere beside the number to put its maximum: either we split it, or 0.604 never appears. We once read our own working retrieval as broken because that number was missing. The same three go out for the two rankings that never read the question, so a dashboard row never shows a mean with no floor under it. Both stores get their numbers from the same scoring function, so the printed table and the dashboard cannot disagree about what nDCG means.

## Documents in the answer key, chunks in the output

A dataset item's expected output is the task's required documents, tau2's own answer key, never a list of chunk ids. Chunk ids are recomputed per run against the chunk set under test, because a chunk level answer key is right for exactly one chunker setting and quietly wrong for every other, and in a dashboard it would look authoritative and outlive the settings that produced it. A test fails if anything chunk shaped reaches that field.

That covers the answer key only, and we were not careful to say so. Each item's task output is the whole question row, which carries the resolved correct and primary chunk ids, and the SDK writes task output onto the trace. So chunk ids do reach Langfuse, as one run's output, which is dated and can be computed again, not as the dataset's answer key, which would be neither.

## The price of keeping two stores

**Two places to keep in step.** A new metric reaches both stores for free, because both go through that same scoring function. A change to what a *run* records does not: the row and the experiment metadata are built in two different scripts, by hand, and nothing fails if only one is edited.

**The stores do not carry the same detail per run.** The committed row holds 36 metric rows, six metrics for each of vector, keyword, fused, reranked and the two nulls. The dataset run publishes the reranked arm and the two nulls only, 54 scores. To compare the arms, use the committed file.

**The dashboard's copy is not in git.** A Langfuse volume is not committed, so wiping your stack loses the traces. The alternative, a directory of committed trace JSON, is a large diff nobody reads.

**One question at a time, on purpose.** The SDK runs 50 items at once by default and we pass 1. The ledger, the append only file recording what a run already paid for, is written from inside the task, and the timings for each stage would stop being comparable to every number already recorded. So the dataset run gets no speedup from the SDK, which is what it costs to keep the two ways of running the question set comparable.

## What we got wrong

**It shipped as a flag.** There was a `--dataset` switch and a `just dataset` recipe. Both are gone. Tracing already decides once from the environment, and a dataset run costs no extra model call, only ingestion, so a second switch was one knob too many.

**Then credentials alone were not enough.** `.env.example` ships working keys pointing at localhost, so "cloned the repo, never ran `just up`" is the ordinary case. A dataset run's first act is to create the dataset, an HTTP call with no fallback, so keys set with the stack down meant a raw connection error after the corpus had been embedded and before a question was asked. We now check we can log in first, one small call, and on failure say Langfuse is configured but not answering and record to the committed file only.

**Partial runs were named after what was asked for.** The old name was `{run_id}-limit{N}`. Then `--limit 5` met a ledger that already held three of those five, and opened a run called `-limit5` whose means were computed over two items. The number in the name is what every average inside the run was divided by, so it now counts the items the run actually contains.
