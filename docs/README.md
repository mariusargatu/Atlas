# How Atlas works, and why it works that way

Atlas is a small retrieval and answering system with a lot of measurement wrapped around it. These pages explain what it does, what we chose, and what those choices cost. They are written for somebody learning to build and test this kind of system.

Every number here comes from a recorded run, and you can print the whole record yourself with `just report --render`, no API key needed. Where a page claims one arrangement beats another, `just compare` will tell you.

## Read in this order

The first three explain what we are measuring. The next three explain how a question turns into an answer. The last seven explain how we know any of it is true.

| | |
|---|---|
| 1 | [Where the answers come from](where-the-answers-come-from.md) |
| 2 | [The question set](the-question-set.md) |
| 3 | [How runs are named](how-runs-are-named.md) |
| 4 | [Why we wrote the stages by hand](why-we-wrote-the-stages-by-hand.md) |
| 5 | [The keyword search](the-keyword-search.md) |
| 6 | [The reranker](the-reranker.md) |
| 7 | [Choosing models](choosing-models.md) |
| 8 | [Why every run uses real models](why-every-run-uses-real-models.md) |
| 9 | [Checking the benchmark](checking-the-benchmark.md) |
| 10 | [Checking answers](checking-answers.md) |
| 11 | [Recording runs](recording-runs.md) |
| 12 | [How we test](how-we-test.md) |
| 13 | [Test coverage](test-coverage.md) |

## Also here

[The same pipeline, in LangChain and LlamaIndex](frameworks.md) maps every stage to what you would call instead in those two frameworks. Read it if you work in one of them and want to know what carries over.

## What these pages are not

They are not a tutorial. There is a written series that this repository was built to support, and it lives elsewhere. When a page says "the series", that is what it means. Nothing in this folder is a chapter.

They are not a history either. These used to be decision records, one per contested choice, each written in the same rigid four part shape. That shape helped while the choices were being made, then got in the way of explaining the result, so they are ordinary pages now. Where we changed our minds, the page says so, because a wrong turn explained is usually the most useful paragraph on the page.
