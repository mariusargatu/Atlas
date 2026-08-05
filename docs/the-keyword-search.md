# The keyword search

Atlas asks every question twice, and we call each half an arm. The vector arm compares meaning, so "cancel my booking" can match a chunk that says "terminate a reservation". The keyword arm matches the words themselves, so an exact product code or policy name is found rather than swapped for something that only looks close. This page is about the keyword arm: what moving it to BM25 bought, and the check it broke.

## From counting words to BM25

The first version counted matching words: turn the query into a set of words, turn each chunk into a set, score by the overlap. It is what most people write first.

The arm is BM25 now. Both scorers still ship, and the old one is reachable by setting `scorer` to `overlap`, because it is what BM25 is measured against. Both start by lowercasing the text, cutting it into runs of letters and digits (keeping `abc-123` and `v2.1` whole) and dropping 124 very common words.

## The three ideas counting words is missing

**A word in every chunk tells you nothing.** If half the corpus says "account", a chunk containing "account" has earned nothing. BM25 weights each word by how rare it is across the collection. This is inverse document frequency, and it is the idea that matters most here.

**A word said nine times matters more than once, but not nine times more.** Counting sets ignores repetition, and counting raw occurrences overrates it. BM25 makes each extra repeat count for less than the one before it, and `k1` (we use 1.5) sets how fast that tails off.

**A long chunk contains more words by accident.** Without a correction, long chunks win by being long. BM25 scales the score down when a chunk runs longer than the collection average, and `b` (we use 0.75) sets how strong that correction is.

Both constants are the shipped defaults, and both are ordinary published choices for BM25. We wrote the scorer out instead of importing a package, because it is only about forty lines and each of them is one of the three ideas above. Import it and you learn a package name. Read it and you learn why a keyword arm scores what it scores.

## What the change bought

Two recorded runs over the same 97 questions at k=10, so we look at the top ten results. `just report --render --run-id 9667f8170801` prints one of them in full and needs no API key. recall@10 asks what share of the chunks that should have been found came back in the top ten. nDCG@10 scores the same hits, but a hit near the top counts for more than one near the bottom.

| keyword arm | recall@10 | nDCG@10 | run |
|---|---|---|---|
| counting words | 0.144 | 0.250 | `9667f8170801` |
| BM25 | 0.192 | 0.348 | `1a737014a9eb` |

A gap in a table is not yet a result. Both scorers answered the same 97 questions, so we can compare them one question at a time, then draw thousands of random samples of those questions to see how much the gap moves:

```
just compare data/settings/naive-keyword.json --metric nDCG@k --arm keyword
```

Free against a warm cache. It prints `difference -0.0981`, interval `[-0.1451, -0.0503]`. The sign is negative because the file you name is the side under test and `naive-keyword.json` is the old scorer. Read it as BM25 being 0.098 ahead, interval clear of zero.

It also prints a blunter number: at 97 questions, the smallest gap a worst case argument can resolve is 0.199. Ours is under that and is still a result. That bar assumes two measurements with nothing in common, and ours share their questions, so comparing one question at a time cancels the noise that comes from some questions simply being harder than others. We had the bar wrong once, at 0.100. It had been worked out for a single average, which sits between 0 and 1, rather than for a difference between two averages, which sits between -1 and 1 and can vary twice as much.

## What the old number was hiding

**The old arm ranked worse than not reading the question.** `just report --render` prints a row called `null: best constant`. It is the ten chunks that are correct for the most questions, handed back for every question, unchanged. It scores 0.256 nDCG@10 and the word counting arm scored 0.250. Not a clean sweep: on recall@10 the old arm beat that fixed list, 0.144 against 0.107. It lost on the order it put its hits in.

**We could not show that using both arms helped.** Merging the two rankings is the whole shape of this system, and with the old scorer we could not show the merged ranking beating the vector arm on its own: +0.029 nDCG@10, interval [-0.008, +0.065]. That interval crosses zero, so we could not even say which side was ahead. With BM25 it gives +0.067, [+0.031, +0.101]. The design was fine. Its keyword half was too weak to show it.

## Where the old scorer still looks better

We used to write that counting words was worse on every measure. That was false, and our own recorded runs said so. Graded nDCG@10 covers only the 22 tasks where the corpus tells us which document carries the answer, as opposed to the ones you read on the way to it. On those 22 the old scorer puts the answer bearing chunk higher, 0.178 against BM25's 0.141, and the same flip shows up after the two arms are merged. We did not switch back: 22 questions is a small sample, the worst case bar is 0.418 there, and the interval on that 0.037 gap runs from -0.027 to +0.110, so it includes zero and we cannot say which scorer is really ahead. A real observation, then, rather than a result, and one a later page may explain.

## The check that had to change

A relation is something that must stay true between two runs, rather than a value checked inside one run. One said that adding documents a question does not need can never move a chunk *up* the ranking, since documents only add competitors. That held for both arms while the keyword arm read the chunk's own text and the query alone. Inverse document frequency reads the whole collection: a word's weight depends on how many chunks contain it, so adding chunks changes every score and a chunk really can be promoted. Measured over the twelve questions that need three documents or fewer:

| keyword scorer | promotions | comparisons |
|---|---|---|
| counting words | 0 | 274 |
| BM25 | 6 | 239 |

So the relation now covers the vector arm only, and the test says why in its own docstring. The other option was to keep both arms and weaken what we assert, down to something like "the two rankings stay roughly similar". Every version of that needs a threshold, and the threshold gets chosen because it passes.

What we lose is real. Nothing now asserts that the keyword arm's ranking is stable as the corpus grows, because it is not, and there is no weaker honest claim to put there. A bug that made the vector arm read the collection is still caught. The same bug in the keyword arm is not.

## What the two scorers cost

Nothing in the test suite builds the old index or sets `scorer` to `overlap`. It is code with no test on it, kept so the comparison above stays runnable rather than becoming a claim you take on trust. `just compare` is the only thing that runs it, so if that stops being run, the baseline is not a baseline and it should go. BM25 costs time too. In the two runs above, recorded seconds apart, the middle keyword search took 16.0 ms against 2.4 ms over the same 1010 chunks. That is more work per question, and worth paying for on this corpus.
