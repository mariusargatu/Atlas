# Ninety seven questions, and we cannot make more

Every question here comes from one place: the banking knowledge domain of tau2-bench, vendored under `data/tau2/banking_knowledge/`. It ships 698 documents and 97 tasks, and each task names the documents an agent must consult. That list is our gold set, so what counts as correct was fixed by somebody else before our system ran. 97 is a small number and we cannot make it bigger. This page is about what that costs, how sure we can be about any difference, and where our intervals come from.

## Why the number is stuck

The reader for this corpus takes a seed and ignores it. It hands back the same 698 documents every time. There is no generator behind it, so we cannot ask for 200 questions instead of 97, and we cannot hold 40 back for a human to read without taking them out of everything else. All 97 tasks name at least one required document, so the filter that skips tasks without one has never dropped a task: 97 is the whole domain, not the part of it that got through a filter. Default chunking cuts the 698 documents into 1010 chunks, and that does not move between runs either. A test asserts the question count is 97, so a partial checkout cannot quietly narrow every interval here.

## What each published column speaks for

Every recorded run scores all 97 questions. One column is different, and the table says so in a row of its own: graded nDCG needs to know which of a task's required documents holds the answer, and we can work that out for only 22 of the 97 tasks. So `just report --render` prints a `questions` row reading `97 97 97 97 97 22` at the foot of the table. Without that row a reader would take the last column for a score over all 97. You can score fewer with `--limit N`, which takes the first N questions, but neither entry point files the result: the report prints and then says "A partial question set is not a run", and the compare command writes nothing at all.

## The blunt bar, and the version of it we got wrong

The crudest honest answer to "is this gap real?" is `1.96 * 1.0 / sqrt(questions)`. At 97 questions that is **0.199**. At the 22 questions behind graded nDCG it is **0.418**, and the table prints that separately rather than letting one bar cover columns measured over different sets. The 1.0 is the widest the spread of a per question difference can be: two scores that each live between 0 and 1 give a difference between minus 1 and 1, and the largest standard deviation such a thing can have is 1.0. Standard deviation is how far the numbers sit from their own average.

We had this wrong for a while. The constant used to be 0.5, which is the worst case for a single average and not for a difference between two of them. At 97 questions it printed 0.100 under every table when the honest figure is 0.199, and it erred in the direction that flatters you: it told a reader a gap of 0.12 was resolved when it was not. Commit `5fe2ead` fixed it.

## What 0.199 rules out

Recorded run `1a737014a9eb` at k=10. Recall@10 is the share of a question's correct chunks that appear in the top ten. nDCG@10 rewards putting them near the top of the list, not merely inside it.

| system | recall@10 | nDCG@10 |
|---|---|---|
| null: random | 0.012 | 0.027 |
| null: best constant | 0.107 | 0.256 |
| vector | 0.184 | 0.355 |
| keyword | 0.192 | 0.348 |
| fused | 0.226 | 0.422 |
| ceiling | 0.604 | 1.000 |

`null: best constant` is the ten chunks that appear in the most questions' gold sets: the best a ranking that never reads the question can do. The reranked row is missing because it is identical to fused, since the reranker we ship is a passthrough and returns what it was given. Recall cannot reach 1.000 because plenty of questions need more than ten correct chunks and only ten are returned, which is why the ceiling row reads 0.604. Now apply the bar. Fused beats the best constant ranking by 0.166 on nDCG, under 0.199, and keyword and vector are 0.007 apart. If 0.199 were the last word, almost nothing here would be reportable. It is not the last word.

## Pairing, and where the intervals come from

A bootstrap resample is simple. Take your 97 per question numbers, draw 97 of them at random with replacement, and average that draw. We do this 2000 times and read the 2.5th and 97.5th percentile of the 2000 averages. That range is the 95% interval. For a comparison we put the *differences* through it: for each question we subtract the baseline score from the score under test, then resample those differences. The two sets are paired on the question name, and two sets of names that do not match are refused, because pairing on list length alone would line up two different question sets and hand back a tight interval about nothing.

Pairing wins back much of what a set of only 97 questions costs us. Some questions are simply harder, and when both configurations face the same hard question that difficulty cancels instead of counting twice. `just compare A.json --to B.json` is the command, free against a warm `.cache/vectors` and recording nothing, so you can run it as often as you like. Replacing the keyword arm's set overlap scorer with BM25 reproduces as **+0.098 nDCG, 95% interval [+0.050, +0.145]**: half the blunt bar, and it still resolves. Both figures are printed side by side, because the gap between them is the lesson. When the interval straddles zero, `just compare` prints NOT RESOLVED and adds that this is not the same as the two being equal. We report that rather than picking the larger number.

## What the seeds are for

Seed 0 returns all 97 questions unchanged, and every caller outside the resample check passes 0. Any other seed draws a resample of the questions. The documents never vary with the seed, so the corpus is embedded once and every resample after that is free.

Those resamples answer one question and no longer set any bar: does the headline number hold still across draws, or is 97 too few to support any comparison at all? A derived threshold used to take twice the standard deviation across four of them and hand it to a merge gate, advertised as having no free parameter. It had two, and the length of the seed tuple was one of them. [Checking the benchmark](checking-the-benchmark.md) has the measurements. That check now runs its own paired bootstrap over the two arms it already scores and asks whether the interval excludes zero, so there is no threshold anywhere for a resample to derive.

Not every threshold works that way, and it is worth saying so plainly. The blunt bar is arithmetic from the question count, but several other thresholds are constants a person chose: one for headroom, and three more behind the adversarial checks. Only two of those four say in a comment where the number came from. The other two are bare, and a bare constant is the kind of thing that quietly stops being right.

## The question style is not the CLI's question style

Every recorded number on this page and in the run store is measured against tau2's own question text, and that text is not a short question. It is the brief handed to whichever model plays the customer in the benchmark's own multi-turn conversation: a persona, verification details, sometimes a quoted line the customer is told to open with, running to a mean of 3,470 characters. The README's own worked example, `just ask "how do I dispute a card transaction?"`, is a short natural question in a completely different register, and nothing here measures that register.

The gap is not cosmetic. Rephrasing one task's identical underlying need, a customer who got a new email address and needs it updated on the account, from the tau2 role-play style into a short natural question collapsed that question's fused recall@10 from 0.750 to 0.000: the same four gold chunks that ranked 2nd, 6th and 8th under the original phrasing fell to vector ranks 741-1005 and keyword ranks 165-787 out of 1010 chunks under the short one. Both search arms missed it independently. That is not a fusion artefact and not a fluke of one question; it is what happens when a corpus's only evaluable question style and a tool's actual documented usage mode are different things.

Nothing here currently measures or discloses this on a recorded run, and it has not been fixed, because the honest fix is a second, differently phrased question set scored the same way, and nobody has built one. Until then, every number on this page describes retrieval against tau2's long role-play prompts, not against the short questions a reader typing at the command line actually asks.

## What an interval cannot tell you

An interval tells you how steady a number is, never whether it is right. Resampling only reports the variation the sample already contains. If all 97 tasks share a bias, and they were written by one team for one benchmark, resampling will hand you a comfortable interval around a number that is confidently wrong. If two configurations we have good outside reason to believe differ keep coming back with an interval that includes zero, the honest response is to say this corpus cannot support that comparison, not to keep reporting the bigger number.

Running the pipeline again does not test the interval either, at least not for retrieval. Run `f45992f36859` is recorded three times, on two different days, and its retrieval metrics are identical in all three: that half of the pipeline is deterministic, so its spread across repeats is exactly zero. Comparing repeats against an interval only tells you something for the halves that do vary, which are what a model writes and what a judge decides.

Labels draw from the same 97, and it is the labelling script that enforces it. The script offers only the current run's own judged answers, so a name it writes is always a tau2 id, and it checks that the verdict reads "pass" or "fail". The store itself is read back on its schema version alone.

This page used to say the label file was empty, on the grounds that the one label it held, on `task_001`, was a placeholder we deleted. It has not been empty since: `data/labels.jsonl` holds 25 rows, all of them recorded against run `1a737014a9eb` with `"rater": "claude-sonnet-5"`, passing 24 of the 25. What is still empty is the human column. No person has graded an answer here, `just label --report` will not pool a model rater with a human one, and until somebody sits down with the 97 there is no human agreement figure to put in this document or any other.
