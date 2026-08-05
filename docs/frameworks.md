# The same pipeline, in LangChain and LlamaIndex

Atlas writes its own small functions instead of using a framework to wire the stages together. That is the one place it does something different from what most teams do, so this page says what you would call instead if you used one.

It does not do anything unusual with the parts themselves. The same kind of embedding model, the same keyword algorithm, the same way of combining two rankings, the same kind of cross encoder, the same tool for tracing. Only the wiring is written out by hand.

## Stage by stage

The pipeline is a list of seven stages. Every row below is one of them, plus chunking and tracing which sit either side.

Turning the question into a vector is a stage of its own, kept apart from the vector search, and that is worth a line here because a framework will not do it for you. Turning a question into a vector is a network round trip to a paid endpoint; the search that follows is a dot product against a matrix already in memory. Time them together, which is how they arrive if you wrap a retriever and read one number off it, and a warm embedding cache makes the dense arm look an order of magnitude cheaper than BM25. The comparison inverts the moment the cache misses.

| Stage | Atlas | LangChain | LlamaIndex |
|---|---|---|---|
| **chunk** | `atlas.corpus.chunk.get_chunker`, choosing between three ways of cutting a document | `RecursiveCharacterTextSplitter`, `SentenceTransformersTokenTextSplitter` | `SentenceSplitter`, `TokenTextSplitter` |
| **embed** | `atlas.models.providers.OpenAIEmbedder`, wrapped in `atlas.models.embed.CachedEmbedder` so the same text is never paid for twice | `OpenAIEmbeddings` with `CacheBackedEmbeddings` | `OpenAIEmbedding` with an ingestion cache |
| **vector** | `atlas.retrieval.dense.VectorIndex`, a matrix of numbers and a cosine similarity | `InMemoryVectorStore.as_retriever()`, or any vector store | `VectorStoreIndex.as_retriever()` |
| **keyword** | `atlas.retrieval.sparse.Bm25Index`, BM25 written out in about fifteen lines | `BM25Retriever` | `BM25Retriever` |
| **fuse** | `atlas.retrieval.fuse.rrf_fuse`, which combines the two rankings by position rather than by score | `EnsembleRetriever` | `QueryFusionRetriever(mode="reciprocal_rerank")` |
| **rerank** | `atlas.retrieval.rerank.OnnxReranker`, a cross encoder that scores each question and chunk pair | `CrossEncoderReranker` | `SentenceTransformerRerank` |
| **answer** | `atlas.models.generate.ModelAnswerWriter` with `data/prompts/answer.md.j2` | `create_retrieval_chain` | `RetrieverQueryEngine` |
| **judge** | `evals.judge.judge` with `data/rubric.md` | LangSmith evaluators | `CorrectnessEvaluator`, `FaithfulnessEvaluator` |
| **trace** | `atlas.trace.observe` | the Langfuse callback, or LangSmith | the Langfuse callback |

Tracing is the row where we do the same thing you would. Ours is a thin wrapper around Langfuse, which is what you would use with either framework too.

## What a framework would give you

Quite a lot, and none of it is what this repository is for.

**Ways to load documents.** Hundreds of connectors for real systems: Confluence, S3, Notion, a database. Atlas reads one folder of files and has nothing like that. If your documents live anywhere real, this alone is a good enough reason to use a framework.

**Ready made retrieval strategies.** Retrievers that merge small chunks into bigger ones, retrievers that walk a tree of summaries, retrievers over a graph. Building an unusual strategy out of parts is the thing frameworks are genuinely good at. Atlas does one strategy.

**The things we simply do not do.** Retrying a failed call, streaming an answer back a word at a time, keeping several customers apart, deleting a document and everything derived from it. All real needs. None of them are here.

## What you would give up

Two things, and they are why we wrote it out.

**You can call each stage on its own.** One call runs the whole line, but the vector search, the keyword search, the fusion and the reranker are each callable by themselves with a plain list of arguments. That is what lets a test make a claim about one arm rather than the whole chain: that adding documents nobody asked about cannot push a chunk higher up the vector ranking. When we replaced the keyword arm with BM25 that same claim became false on the keyword side, because BM25 looks at how common a word is across the whole collection. The test went red and told us so. A claim you can only make about the end of a chain is a claim about the chain.

**Every stage hands back a described object.** A search result, a fused result, a reranked result, an answer, a verdict. Each one is frozen, so nothing downstream can quietly change it, and each has named fields you can make assertions about. One record carries all of them plus how long each stage took, which is how the report can say what each stage cost in time and in money without measuring anything twice.

Put simply: if you cannot see the seams, you cannot make claims about them. This repository is about making claims about them.

## Three places the comparison breaks down

**There is no agent here.** Nothing chooses which tool to call, nothing loops, nothing tries again with a different plan. It is a straight line. If your problem needs an agent, none of the argument above transfers, and a framework is probably the right answer.

**Our fusion is not quite `EnsembleRetriever`.** Both combine rankings the same way, but at the depths Atlas ships, the arithmetic turns it into something stricter than a blend. Anything found by both arms beats everything found by only one. Measured over the 97 questions: the two arms agree on a middle of 17 chunks, and the top ten comes entirely from that overlap for 88 of the 97. That is a property of the depths, not of the method, and it is the sort of thing you would never notice from inside a framework. Which is the argument for writing it out, made against itself.

**This paragraph used to end "so the weights make no difference at all", and that was wrong.** The structural argument above holds only while the two weights are *equal*, which is the one case where weighting is vacuous by symmetry: a chunk in both arms scores `w_v/(k+r_v) + w_k/(k+r_k)` and a chunk in one scores a single term, and equal weights are what stop the single term from ever winning. Raise one weight and it can. Holding `vector_weight` at 1.0 and sweeping `keyword_weight` over the real corpus:

| `keyword_weight` | fused nDCG@10 | fused recall@10 |
|---|---|---|
| 0.5 | 0.4144 | 0.2236 |
| 1.0 *(shipped)* | 0.4218 | 0.2255 |
| 2.0 | 0.4262 | 0.2294 |
| 5.0 | 0.4038 | 0.2168 |
| 20.0 | 0.3654 | 0.1957 |

The repository already knew this on the code side and only the prose disagreed: `tests/pipeline/test_fuse.py::test_the_arm_weights_are_read_rather_than_assumed` exists precisely because the weights change the order, so we shipped a page and a test that contradicted each other. Worth noting and not worth over-reading: 2.0 scores above the shipped 1.0 on both metrics. That gap has had no paired interval put around it, and [the blunt bar](the-question-set.md) at 97 questions is far wider than it, so it is an unexplored direction rather than a finding.

**The judge gives a verdict, not a score.** It returns pass or fail and a reason, against a written guide with a version on it. We compare that against scorers from another library that return a number between zero and one. The number worth reading is how often the two agree once you allow for agreeing by chance, and neither framework's evaluator is shaped around that question.

## If you want to move this onto a framework

The short version: keep the contracts, keep the settings, and replace the insides of the four retrieval functions with framework calls. Everything in the evaluation code reads a record and would not notice.

That is also the real test of whether the seams are as clean as this page claims. If it turns out to be hard, then writing the stages by hand bought less than we think, and [the page explaining that choice](why-we-wrote-the-stages-by-hand.md) needs revisiting.
