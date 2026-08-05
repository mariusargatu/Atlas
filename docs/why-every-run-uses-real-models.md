# Why every run uses real models

Every command here that turns text into vectors, or asks a model to write an answer, talks to a real service. So do the checks that block a merge. There is no free mode, and nothing here runs without an API key. That has a real price for a teaching repository. This page says what the price is, and tells the story of the fake model we used first.

## What needs a key and what does not

Embeddings always go to OpenAI. Writing an answer and judging one read the `MODEL_PROVIDER` environment variable and pick OpenAI or Anthropic. Anthropic has no embeddings endpoint, so only the answering half is a choice.

Plenty still runs on nothing. `just lint` and `just types` run ruff and mypy and never touch the network. `just report --render` prints a table straight out of the run store, so you can read our numbers before you spend a cent. `just label` appends one line to the label file. Everything that embeds or generates needs a key, so the test suite looks for `OPENAI_API_KEY` once, before pytest has even gathered the tests, and stops the run if it is missing. Without that, every test that builds an index failed on its own, each one printing a library error that named a variable rather than the problem.

## The fake model we used first

The first version of this repository had a second embedding backend that needed no key and no network, and every check that could block a merge ran on it. It was not a smaller embedding model. It was this: hash the chunk text with blake2s, take four bytes of the digest as a seed, and ask a random number generator for 256 numbers. The result is stable for a given text and carries nothing at all about what the text means. There was a matching stand in for generation, which cited whichever chunk it was shown first and refused when it was shown none. Three things followed.

**The checks passed.** They tested that the pipeline repeats, not that it works: that blending two ranked lists reads their positions and not their raw scores, that shuffling the document order changes nothing, that a chunk's text really is the slice its range names. A hash function satisfies all of those perfectly. A check a hash function passes is not a check on retrieval.

**The paid path could not have run anyway.** The real embedding client pointed at `https://api.example-embeddings.invalid/v1/embeddings`. The `.invalid` ending is reserved on purpose, so no such address can ever reach a real server. Nothing here had ever reached a real embedding endpoint.

**CI was wired to secrets nothing read.** Three key names sat in the workflow files: `ATLAS_ANSWER_API_KEY` and `ATLAS_JUDGE_API_KEY` on two jobs in `ci.yml`, both marked `continue-on-error`, and `ATLAS_VECTOR_API_KEY` on the nightly job. No client in the repository read any of the three. They looked like the tier that spends real money and were running against no credentials at all.

## The check that was supposed to catch this

The page this one replaced asked the right question. It wrote down the condition that would prove it wrong: the local backend might be so weak that rankings came out effectively random. And it named a signal, the benchmark validity checks going red, because a random baseline and the real system would stop looking different. The condition was met. The signal never fired.

Here is why, and the trap is easy to build again. We search twice for every question and call each search an arm: the vector arm matches on meaning, the keyword arm matches on words. Three validity checks all score the real system the same way, running both arms and merging them into one list before anything is measured. The merge is reciprocal rank fusion, a way of combining two ranked lists by where each chunk sits in them rather than by its score. The keyword arm worked, and it kept the merged list far enough clear of a random ranking that the chance check stayed green while the vector arm carried no information at all. The check could not see a dead arm inside a live blend. What found the fault was a person scoring the arms separately by hand.

So the lesson is not that the old decision was wrong about cost. It is that writing down what would prove you wrong is worth only as much as the thing you expect to notice it. Write down the observation that would change your mind, then ask what has to be true for you to ever see it. The fix is in what `just report` prints: every run scores each arm on its own line, next to two baselines that ignore the question, so a dead arm is a bad row rather than a hidden term.

## What the numbers look like now

From run `1a737014a9eb`, over all 97 tau2 tasks, at k=10. Recall@10 is the share of the chunks that should have been found that turn up in the top ten. nDCG@10 also rewards putting the right chunks near the top rather than anywhere in the ten.

| What ranked the chunks | recall@10 | nDCG@10 |
|---|---|---|
| the vector arm alone | 0.184 | 0.355 |
| the keyword arm alone | 0.192 | 0.348 |
| both arms blended | 0.226 | 0.422 |
| the same ten chunks for every question | 0.107 | 0.256 |
| a random ranking | 0.012 | 0.027 |

The recall@10 ceiling on this corpus is 0.604, because many questions need more than ten chunks to be answered and we return ten. There is no reranked row because it would be identical to the blended row: the reranker we ship is a passthrough, and [`the-reranker.md`](the-reranker.md) says why. Both arms beat a fixed list, and blending beats both arms. The hash backend has no row here: it is deleted, and so is `just demo`, the twenty question command that scored it at the time. What we can say is that random vectors give an arbitrary ranking, and the bottom row is what an arbitrary ranking scores.

## What this costs you

**Money, in small amounts.** Embedding the whole corpus, 1010 chunks cut from 698 documents, is $0.0036: 180,541 tokens at $0.02 per million. One answer is about $0.009 and one judge verdict about $0.006, from the recorded judged run's own totals over 97 questions, so `just ask` is about $0.009 against a warm `.cache/vectors` and about $0.013 the first time. The record this page grew out of said $0.02 a question, roughly double what the run store says, and guessed $0.20 to $0.60 for a full push, which we could not reproduce from the suite. Both figures are gone.

**A cache miss is a bill.** `.cache/vectors` files each vector under the model name, version, size and normalise flag together with the chunk text, so changing the embedding model or the way documents are cut pays for every vector again. That used to cost time. Now it costs money.

**CI cannot run on a fork.** Pull requests from forks cannot read repository secrets, so every job except lint and types fails for an outside contributor. The older record called that a design failure for a project meant to be copied, and it was right. We accepted that cost rather than answering it, and there is no tier here today that a fork can run.

**Repeatability is a weaker claim than it was.** The same text always hashed to the same vector, so two machines produced identical rankings forever. A remote service, whose version can move under us, does not. Our checks now claim only that two runs agree inside one process against one client, and we write that down as the smaller claim it is.

## What is still not real

The same kind of gap, said plainly before somebody finds it. The suite embeds with a real model but does not write answers with one. pytest runs with `-m 'not reporting'`, so any test tagged `reporting` is skipped unless you ask for it. The tests that call a real judge carry that tag, and every other test that touches the answer writer gets a fake model instead. So a push pays for embeddings, and the answer path is checked for its shape rather than for what it writes. That is a hole, and the first place to look if a fault in generation ever gets through a green build.
