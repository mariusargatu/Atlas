# Checking the benchmark

A benchmark is a system, and systems have bugs. If ours cannot tell a good retriever from a useless one, then every comparison we run on it is noise with a decimal point on the end. So before we believe any number, we check the benchmark itself. This page says what those checks ask, how many there are, and where they live.

## The trap

Recall at 10 is the share of the chunks a question needs that turn up in our top ten results. It is the headline number here.

Now picture a retriever that never reads the question. It returns the same ten chunks, every time, forever. On this corpus that thing scores recall@10 of 0.107. The real system, vector search and BM25 blended together, scores 0.226 (recorded run `1a737014a9eb`).

So the retriever that does no work reaches nearly half of what the real one reaches. If some change ever pushed the real number down to where the fixed list sits, our tables would still look like tables, and none of the numbers in them would mean anything. It is that close because the questions overlap: a tau2 task needs about twenty one chunks on average, and the single most wanted chunk here is correct for 44 of the 97 questions. Ten chunks out of 1010 sounds like a thin slice. It is not.

## Where the checks live, and how many there are

The validity module holds three checks, one on the fixed list, one on headroom and one on chance, plus a fourth that is a check in everything but name: it asks whether every question has at least one correct chunk.

The two rankings that never read the question, a best constant list and a random one, live next door in their own module. Everything above is measured against them. They moved out when the validity module hit the four hundred line cap; they are pure functions of the chunks and the question set, with no benchmark and no embedder in them, which is what made them the piece to lift.

The validity tests hold 9 tests as this is written. They call the functions above and add the assertions that turn them into a pass or a fail. A floor for the count is pinned in `tests/EXPECTED_MIN_TESTS`, and the check is a band rather than an exact match — floor to floor plus five — so a test that quietly vanishes gets noticed without the number here having to be edited every time one is added. `uv run pytest --collect-only -q tests/measurement/test_validity.py` is the current figure.

The adversary suite runs the fixed list check again, this time as one of a set of alarms pointed at systems we broke on purpose. It registers that check as one of seven detectors, and the same ten passages every time as one of seven broken systems. A test asserts that this broken system is caught by the fixed list check and by nothing else.

The broken systems themselves live in their own file, and the newest of them, a poisoned corpus where a document stops addressing the customer and starts addressing the model, lives in another one along with the patterns that catch it. Both files exist because the adversary suite hit the four hundred line cap holding all three jobs at once.

All of it blocks a merge. The validity and adversary jobs in CI carry no `continue-on-error` flag, and both are named on the blocking list. So is every other job in the workflow, `coverage` included: this page used to name `coverage` as the one that reports without blocking, which stopped being true when it started comparing against a committed baseline. An evaluation numbers job was the other soft one until its flag was removed as a fiction, since the tests it runs were already blocking inside the suite integrity job.

There is a fourth place the fixed list turns up, and it is not a check. Both question ignoring rankings are computed when a run is collected and summarised, so every recorded run carries a `null: random` row and a `null: best constant` row beside the real ones. A run row refuses to be built if a metric is published without them, so you cannot read one of our tables without also seeing what a system that ignores the question scores.

## What each check asks

**The fixed list.** Does the real system beat the same ten chunks by a margin a paired bootstrap can tell from zero? It scores both arms on the same questions, pairs them, and passes when the interval on the gap excludes zero. A test pins the degenerate end: an arm that ties the fixed list question for question produces an interval of exactly [0, 0] and has to fail.

**Headroom.** Is there room left to show an improvement? Not measured against a perfect score of one. On this gold set recall@10 has an average maximum of 0.604, because the average question needs about twenty one chunks and we return ten. The check measures against that ceiling and fails once we have taken 98% of it.

**Chance.** What does a random ranking score, with an interval around it? Random reaches recall@10 of 0.006 here, interval [0.003, 0.010]. The test asserts the real number clears the top of the interval, not just the middle of it.

That figure used to read 0.003, which was not this check's answer but the baseline ranking's, and that baseline was drawing one ranking and reusing it for all 97 questions, so it published a single Monte Carlo sample where the expected value belongs. It now shuffles per question, as the chance check always did, and both land either side of the analytic rate of 10/1010 = 0.0099.

**That rate is exact, and this page used to add a qualifier that gave it away as guesswork:** "0.0099 for a question needing one, a little less averaged over questions needing about twenty one". There is no "a little less". The gold set size cancels, and the two lines are worth having in front of you, because the wrong intuition here is a natural one.

Draw `k` chunks from `N` without replacement. Each individual chunk is equally likely to be in the draw, so each of a question's `g` correct chunks is drawn with probability `k/N`, and by linearity of expectation the number of correct chunks retrieved has expectation `g · k/N`. Recall divides by `g`, so `E[recall@k] = k/N` — for every `g`, with no dependence on it at all. A question needing twenty one chunks and a question needing one have exactly the same expected recall from a random ranking. Needing more chunks makes each one harder to get *all* of, which is what depresses the ceiling to 0.604, and does nothing to the average share.

The recorded baseline, 0.0122, therefore sits *above* 0.0099 rather than below it, and that is ordinary Monte Carlo noise at 97 questions and one draw each, not a gold-set effect. `evals/baselines.py` carries the same derivation in its docstring, added when this was first noticed; the page you are reading is the one it was noticed against.

**Correct answers exist.** Does every question resolve to at least one correct chunk? A companion test injects the fault (it drops every chunk belonging to a document the gold set requires) and asserts the check goes red and names the question, so it cannot pass by never raising.

## The gate is an interval, not a threshold

The fixed list check scores both arms on the same 97 questions, hands the two per question maps to a paired comparison, and passes when the resulting interval excludes zero. Today that reads `a paired gap of 0.118 with interval [+0.0770, +0.1615] over 97 questions, which clears zero`. There is no constant in it and nothing to loosen: the only argument is the resample seed, and moving it moves the bounds in the fourth decimal place. Seeds 0, 1, 7 and 99 give lower bounds of 0.0770, 0.0763, 0.0766 and 0.0746.

**This page used to describe something else, and the something else was not what it claimed.** The gate was twice the population standard deviation of the headline recall across four bootstrap resamples (seeds 1, 2, 3 and 4), and this section called it free of parameters. It had two. The seed tuple was one: across ten disjoint groups of four seeds the threshold ranged 0.0178 to 0.0498, and *shortening the tuple to two seeds dropped the gate from 0.0426 to 0.0128*, a 3.3× loosening that edits nothing resembling a threshold, which is exactly the silent ratchet the paragraph promised was impossible. Using the population standard deviation rather than the sample one was the other, and it did not compute the "twice the standard deviation" this page described either.

It was also the wrong quantity. The two arms are scored on the same questions, so the variance they share cancels and the bar belongs on the spread of the per question *difference*; the old estimator measured the spread of one arm's mean instead. And its guard test opened by asserting the threshold was at least zero, which is true of every input a float can hold.

Two picked numbers still sit nearby and it would be dishonest not to say so: the 0.98 share of the ceiling the headroom check allows, and the tolerance of 0.15 in the resample stability test. Neither is this gate. Bootstrap resamples are still built, and one test uses them to ask whether 97 questions can support any comparison at all, but nothing derives a bar from them any more.

## Seeds resample questions, never documents

The corpus comes from somebody else: 698 documents from tau2-bench, cut into 1010 chunks at the default settings, with 97 tasks. We never regenerate it. So asking for a benchmark at a given seed does not build a new collection. Seed 0 gives the whole question set, and any other seed draws one of those bootstrap resamples. That measures the right thing: how much the headline number moves when the questions happen to be different. Because the documents never move, their vectors never move either, so we embed them once per process rather than once per seed.

## Two ways these checks stopped checking

This is the useful part, because both failures were silent. The fixed list used to be sorted by how many fact records named each document. That was a real signal on the generated corpus this project used to have. On tau2 the key is constant, so the sort collapsed to its tie break and left an alphabetical list. Measured today, an alphabetical list scores recall@10 of 0.042 and the current list scores 0.107, so the check had been facing the weakest opponent available rather than the strongest. It now sorts by how many questions each chunk is correct for.

The headroom check used to compare a raw recall against 0.98. Since recall@10 cannot average above 0.604 here, that was true of every arrangement including a perfect one. A check that cannot fail is a comment. It now compares the share of the achievable ceiling we have taken. Both faults had one cause: a premise died (the generated corpus was deleted) and the code it justified stayed behind looking healthy.

## What it costs

Building the benchmark embeds the corpus with a real model, so the validity job spends real money every time it runs, and it cannot run on a fork at all, because a fork cannot read repository secrets. The machinery is not free either: the validity module is 360 lines against this repository's cap of 400, and the test file is another 150. And the headroom check's failure message advises adding filler documents until real headroom returns, which is advice we could not take. The corpus belongs to somebody else.

## What these checks do not tell you

They say the benchmark can tell systems apart. They do not say that a particular gap between two settings is real. Score two systems on separate questions and, with only 97 to go round, the smallest difference you could trust is 0.199. That is a worst case bar, and it is wider than most of the differences we care about. So `just compare` scores both settings files on the same questions and resamples the pairs, which cancels the noise the two share instead of counting it twice. Done that way, swapping set overlap for BM25 in the keyword arm comes out at +0.098 nDCG with an interval of [+0.050, +0.145]. It costs nothing once the embeddings are cached.

Pass the checks on this page first. Then quote an interval.
