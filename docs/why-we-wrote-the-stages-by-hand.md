# Why we wrote the stages by hand

Most retrieval projects start by picking a framework. Atlas does not use one. The pipeline is a handful of small Python functions we wrote ourselves and that you can read in an afternoon. This page says why, what that buys, and what it costs. The table of which Atlas function lines up with which LangChain or LlamaIndex class lives on its own page: [the same pipeline, in LangChain and LlamaIndex](frameworks.md).

## What we built instead

A question goes through seven named stages, and they live in one tuple:

```python
STAGES = ("embed_query", "vector", "keyword", "fuse", "rerank", "answer", "judge")
```

`embed_query` turns the question into a list of numbers, using the same model that turned every chunk into one, so that similar text lands nearby. `vector` then compares the question's numbers against every chunk's. Those two were one stage until recently, and separating them is the point of a later section. `keyword` searches by words, using BM25, a scoring rule that rewards rare words and stops rewarding a word once it repeats a lot. `fuse` blends the two ranked lists with reciprocal rank fusion, which scores a chunk by where it placed in each list rather than by the raw scores, because a cosine similarity and a BM25 score are not on the same scale. `rerank` can reorder the top of that list with a cross encoder, a model that reads the question and one chunk together and scores the pair. `answer` writes the reply from the chunks it was shown. `judge` grades that reply. Those last two are the ones a caller can skip: you can ask for retrieval alone, and judging only happens when a caller also hands over a grader, which most runs do not.

The whole thing is small. The pipeline file is 199 lines, and the five files that do the searching and the ordering come to under 400 between them, comments and docstrings included. Both figures drift with every edit and this page has already quoted a stale one — it said 242 lines for a pipeline file that is 199 — so run `wc -l src/atlas/pipeline.py src/atlas/retrieval/*.py` rather than believing either. Search `uv.lock` for langchain, llamaindex, haystack, langgraph or dspy and you find nothing.

## We declined the wiring, not the components

This is the one place Atlas walks away from what production teams run, so it is worth being exact about how far. We use the same class of embedding model, the same keyword algorithm, the same rank fusion, the same class of cross encoder and the same tracing tool. We skipped only the wiring layer.

Two of those need a note, because stated flat they would claim more than is true. The cross encoder ships turned off, defaulting to a passthrough, and [the reranker page](the-reranker.md) has the measurements that led there. Tracing is real, but it does nothing until you give it credentials. It reads two Langfuse keys from the environment, and when they are missing it quietly switches itself off rather than failing.

## What small functions buy us

Two things, and both are about checking our own work.

**You can call one stage on its own.** Search, fusion and reranking each take a plain argument list and return an answer. That is what lets a test assert something about one arm alone: adding documents a question does not need can never move a chunk *up* the dense ranking, because a cosine score depends only on the question and the chunk. We used to assert that of the keyword arm too. When we replaced a set overlap score with BM25, it stopped being true there, because BM25 weights a word by how rare it is across the whole collection, so new documents really do change old scores. We narrowed the test to the arm where the relation still holds instead of weakening it into something both arms would pass. Neither move is available if the arms only exist inside a chain.

**Every stage hands back a typed object.** They are frozen dataclasses, and a record carries every one of them plus the wall clock time each stage took. Those timings become a median and a slowest tenth per stage, that second figure being the mean of the slowest ten percent of calls, where a stage's real cost hides. Nothing is measured twice to find where a run spent its seconds.

**A timing is only honest if the stage is the thing you think it is.** `embed_query` used to sit inside `vector`, so one figure covered a network round trip to a paid endpoint and a dot product against a matrix already in memory. Because the vector cache has served the query embedding on every run this repository has recorded (every committed row carries `cost_usd.embedding == 0.0`, which is how you can tell) that figure was 1.7 to 5.7 ms against `keyword`'s 13 to 19 ms, and it read as evidence that the dense arm is an order of magnitude cheaper than BM25. It is not; the comparison inverts at any real latency per call. The stages are split so the cache shows up as a number rather than as a footnote, and rows recorded before the split carry no `embed_query` timing at all.

There is also a simpler argument. There is no agent here. Nothing decides which tool to call, nothing loops, nothing retries with a different plan. Two search arms run and meet at `fuse`, and after that it is a straight line. Wrapping a straight line in an orchestration layer adds machinery that does no work, and a teaching repository full of that teaches people to add it too.

## What it costs

You give up two things, and they are the real reasons to use a framework: readers you did not write for other people's document stores, and a choice of index types you can swap without writing one. Atlas reads a single corpus of 698 documents copied into this repository from tau2-bench, cut into 1010 chunks at the default settings, with 97 questions attached. There is a seam for a second corpus, and `src/atlas/corpus/registry.py` now resolves one by name, so the scripts no longer each construct `Tau2Source` in their own code — this page used to say they did, and that stopped being true when the registry landed. tau2 is still its only implementation, and the source name is still not a field in `Settings`, so it does not reach `run_id`. If your documents live in Confluence or S3 or a database, that alone settles it.

Be careful about which gaps are real. We do have caching: one embedder writes vectors to disk and another keeps them in memory, which is why `just report` and `just compare` are free once the cache is warm. We do have basic retries, three of them, passed to every client we build. What we genuinely do not have is readers for outside stores, replies that stream back a word at a time, one deployment serving many separate customers, and any way to change the corpus once it is built.

We also expect a maintenance cost. If the usual retrieval pattern moves on, a framework can absorb that change for its users and our code cannot, so Atlas would need editing where a project built on a framework might need only a version bump. That is a prediction, not something we have measured.

## What we got wrong

Three things, and they are the most useful part of this page. **We promised a page that did not exist.** For the repository's first four days the record called the framework mapping not optional and linked to nothing. `docs/frameworks.md` exists now.

**We got our own stage count wrong.** The record said eight steps from the first commit until 2 August. The tuple in the code was five names when it was first written and is seven now. It has never been eight. That tuple is the single place a stage is named, and a test keeps its own copy to compare against, so renaming a stage fails a check rather than drifting quietly.

**Stages written by hand invite copies written by hand.** Callers that wanted retrieval without generation reassembled the stages themselves, and those copies went stale. That is the real way this choice goes wrong, and it is not the way we expected. The fix was one object that pays the corpus wide costs once and exposes a single entry point, which every script now calls.

## When you would be right to reach for a framework

If you need an agent, none of the reasoning here transfers. If your documents live in real systems, the readers you would get for free settle it on their own.

Inside this repository, the honest trigger is size. This paragraph used to point at a merge gate for it: a repo contract test capping any source file at 400 lines, run on every push. That cap is gone, and the reason it went is worth more than the trigger it provided. A guideline that blocks a merge stops being a guideline, and this one had started deciding how the subject was laid out rather than how the code was. Four modules still open by explaining that they exist because another file hit the cap, so a reader following the poisoned corpus argument now reads three files and the reporting path four, for reasons that have nothing to do with either subject. It had also brought `src/atlas/models/providers.py` to exactly 400 of 400: the next comment anybody wrote would have failed the build, in a repository whose product is largely its comments.

So the trigger is still size and it is now a judgement rather than a number. If the wiring cannot be read in one sitting without us building a framework of our own, a framework has started paying for itself. The other test is unchanged and is the better one anyway: the exercise at the end of [the framework mapping](frameworks.md). Keep the types, replace the insides of the retrieval functions with framework calls, and see whether anything in the eval code notices. If that is hard, the boundaries were never as real as this page claims.
