# How we test

Not every file here gets the same treatment. Some of it gets a failing check before the code exists. The rest gets written first and guarded by a few short checks. This page says where that line sits, what CI does with the result, and the two times we drew it in the wrong place.

## What we write the check for first

Test first means: write the check, run it, watch it fail for the reason you expected, then write the code that makes it pass. It makes you say what correct means before you say how to compute it, and for a measurement that is the whole job. We do it without exception for four things, because these four are what this repository is about.

**Properties**, 25 cases. A property is a statement that has to hold for every input, not just the three you thought of. Twelve of the eighteen test functions hand the job of inventing inputs to hypothesis, which generates documents and ranked lists and tries to break them. One fixes what recall returns for a question with no correct chunks: not a number.

**Relations**, 11 cases. A relation says what must stay the same between two runs that differ in a way that should not matter. Shuffle the chunks before you index them and the ranked list has to come back identical, or the order we happened to add them in is quietly part of the ranking. These run the real stages over the real corpus.

**Broken systems**, 23 cases. Seven deliberately broken systems, each wired to the measurement that should notice it, plus a healthy control no detector may fire on. Some break search, some break the answer writer. Can our measurements tell a bad system from a good one?

**The benchmark's own checks**, 9 cases. Whether the corpus and question set can separate systems at all. [Checking the benchmark](checking-the-benchmark.md) explains what each one asks.

## What we do not, and where that line moves

Everything else is plumbing: the types, the settings tree, the index builders, the caches, the answer writer's wiring, the label file. We write it directly and add a short check for the one or two things that would silently break everything downstream. The sparse index test is 70 lines and 6 cases, the config test 82 lines and 4, the label test 49 lines and 4. Not because plumbing does not matter, but because a tutorial about evaluation that opens with two hundred unit tests teaches the wrong reflex and buries its own subject. Coverage is never a floor for the same reason. It does block a merge, but only on a regression against the committed baseline, which is a different question from how much of the tree ran: see [Test coverage](test-coverage.md).

Plumbing is the default, not a ceiling, and three modules have argued their way onto the strict side. The contracts test is 306 lines and 25 cases with its own merge blocking job, though the types it checks look like the dullest code here. The fusion test carries 5 hypothesis properties: two over blending, two over reranking, one over the generator that feeds them. The embedding test spends four of its nine cases on the vector cache key, because a key that collides serves one model's vectors to another model's run and nothing goes red. What moves a module across the line is simple: somebody wants to write a property for it.

## What CI runs

The workflow has ten jobs and every one of them blocks a merge. That set is not just a habit. It is written down as `BLOCKING_JOBS` in `scripts/repo_checks.py`, and a test fails if that list and the workflow file disagree.

| Job | What it runs | Cases | Pinned |
|---|---|---|---|
| `lint-and-types` | `just lint` and `just types` | | |
| `properties` | `tests/pipeline/test_properties.py` | 25 | |
| `contracts` | `tests/pipeline/test_contracts.py` | 25 | |
| `core-relations` | `tests/pipeline/test_relations.py` | 11 | |
| `validity` | `tests/measurement/test_validity.py` | 9 | `validity` |
| `adversaries` | `tests/measurement/test_adversaries.py` | 23 | |
| `repo-contract` | `tests/meta/test_repo_contract.py` | 11 | `repo_contract` |
| `evaluation_numbers` | `tests/measurement/`, all of it | 123 | |
| `coverage` | `scripts/coverage_delta.py` | | |
| `suite-integrity` | `just test`, the whole suite | 251 | `suite_integrity` |

**Read that middle column as a snapshot, not as a fact about the repository.** Every number in it is a hand copy. This page carried 189 for the whole suite while the suite ran 238, and it moved three times in the hour these rows were being written. Only the three rows marked in the last column are pinned anywhere, in `tests/EXPECTED_MIN_TESTS`, and only those three are checked: `tests/meta/test_repo_contract.py` asserts each collects between its floor and floor plus five, so a file that stops being collected fails a build and a table that goes stale does not. `uv run pytest --collect-only -q <path>` prints any row's real number in under a second, and that is the number to trust over this table.

This page used to say that nine of the ten blocked and that `coverage` reported without blocking, "marked `continue-on-error`". Neither half is true. There is no `continue-on-error` anywhere in `.github/workflows/ci.yml`; the only trace of one is a comment recording its removal, `coverage` is on the blocking list with the other nine, and [Test coverage](test-coverage.md) has said all ten block for as long as this page said nine did. Two pages in one repository gave two answers about the same workflow file, which is exactly the failure the `BLOCKING_JOBS` check exists to prevent in the code and nothing prevents in the prose.

What `coverage` blocks on is narrower than the other nine, and that is the distinction the old wording was reaching for and got wrong: it compares the total against `data/coverage_baseline.json` and fails on a regression, never on an absolute floor.

`evaluation_numbers` was the soft job before it, and this page used to say so. It never really was. Every test it collects is also collected by `suite-integrity`, which blocks, so the soft tier was empty and a red result in it had already failed the build one job over. We removed the flag rather than the overlap, because a reader running `just test` should not silently skip the measurement half of the suite.

## The suite checks itself

A test that quietly stops being collected is invisible, and a job that asks for tests and finds none still exits happily, which `continue-on-error` then paints green. Both have happened here. So we pin a floor per job in `tests/EXPECTED_MIN_TESTS`, three keys today, and check that each job collects between its floor and floor plus five. That file is the record, not this page, which is why the numbers live there and the mechanism lives here. A floor alone is satisfied forever by a suite that stopped growing, so adding tests means bumping the number in the same piece of work. Skipping is not available either: a plugin turns a skipped test into a failed session, and only an expected failure declared with a marker is exempt.

That sentence used to read "any skipped test", and the plugin does less than that on its own. It hooks the run phase, so it sees a test that started and skipped. It does not see a module that skips at import, which never reaches the run phase and takes the whole file with it. Three other mechanisms close the gap. The session fixture raises rather than skipping when the key is missing, a job that collects nothing exits 5, and the floors above fail the moment a file stops being collected. The reasons a collection hook is not the fix are written at the bottom of the plugin.

Another 17 cases are about the repository rather than about the code. Every relative link resolves, no CI job names a key that no client reads, every command in the justfile appears in the README, and the pipeline never imports the measuring code. That last one keeps the dependency pointing one way. The measuring code may read the pipeline's types, but the pipeline can never reach for a score.

## Two times the plumbing broke

A *human* verdict is the one artefact here that cannot be regenerated at any price. Everything else comes back if you pay for the run again. This page used to say that of the label file itself, which is a different and weaker claim, and today it is plainly false: all 25 rows in `data/labels.jsonl` were written by `claude-sonnet-5`, so the whole file could be regenerated tomorrow for the price of 25 judge calls. It holds no human verdict at all, and the sentence that treated the file as irreplaceable was borrowing its weight from a labelling pass nobody has run. The labelling script took whatever you typed. "Pass", or an empty line, went in exactly as typed, and nothing that later looked for "pass" would match it. The label was there, useless and permanent. Now it refuses any verdict outside `("pass", "fail")`, and refuses an empty reason, before it appends anything. The four tests on the label store never saw this: they check the store, and the fault was in its caller.

The same script had a quieter fault, and the store is the evidence: it held one label, reading `"looks fine"`. It asked for a question name and nothing else, so you had to already know which questions existed, and you graded an answer you could not see against a guide in another file. Validation had made it impossible to record a *malformed* verdict and done nothing about how hard it was to record a real one. It now walks the unlabelled answers of a run and shows each one with its passages, its citations and the scoring guide, the same evidence the judge is given, because agreement between a human and the judge means nothing if they were looking at different things. We deleted the placeholder rather than keeping it: one label whose reason is "looks fine" is not a smaller version of a calibration set.

What replaced it is not a calibration set either, and the file says so if you read the `rater` field. The 25 rows there were graded by a model against the same guide the judge uses, so they share its priors and cannot stand in for the person the design calls for. `just label --report` prints that warning above the figure and groups on the rater rather than pooling, so the runtime has never claimed more than it has. Four pages of prose did, in four different ways, and this was one of them.

The judge prompt template rendered only the answer's text. Neither the citations nor the outcome reached the model, so rule 2 of the rubric, that every citation must name a chunk the answer was actually shown, could not be applied at all. The two bias relations that mutate the citation list could not change a verdict, because no verdict had ever depended on it. Two merge blocking checks that could not fail. Rendering is a separate function now, so the prompt can be inspected without paying a model. The first version of the test on it could not have failed either.

Neither fix was a return to test first everywhere. Both were a better check on the plumbing, which is what the rule says should happen. But the honest reading is that the code we exempted is the code that broke.

## What this costs

One module had no test that named it: the pricing table, imported by the provider layer, the deepeval suite and three scripts, and producing every dollar figure the README calls recomputable. It showed. Its own error message told you to add a missing model to a table in the provider module, and that table lives in the pricing module. `tests/pipeline/test_pricing.py` now covers it, which closes the specific hole this paragraph was written about. The reason to keep the paragraph is the shape of the gap rather than the gap itself. Nothing flagged the module, and no coverage floor would have, because the provider layer imports it and the lines ran. A module can be executed by the whole suite and checked by none of it, which is the entire argument of [Test coverage](test-coverage.md) turning up in our own tree.

Five checks do not run in CI at all: four judge bias relations and the one control they are measured against. They carry the `reporting` marker, and the default options exclude that marker, so the suite collects five more tests than it selects: 251 of 256 as this was written, against the 189 of 194 the sentence used to claim. Only `-m reporting` selects the other five, and nothing does.

The marker's own description used to say they had their own CI job. There has never been one, and that wording is now corrected. Selecting them needs a recorded noise floor: all five are multiples of the judge's disagreement with itself, and the fixture fails rather than inventing one when it is missing.

`just noise-floor` produces it, and the result is committed. That recipe is new. The function behind it had existed since the judge was written with no caller outside a test, which is why no floor was ever recorded and why these five tests had never run. It judges answers you already bought rather than writing new ones, ten verdicts a question at the default repeats, so `--limit` is worth using first.

Running it produced two findings, and neither is the one it was built for.

The first is that `gpt-5.6-luna` rejects the sampling temperature. A floor is two columns differing only in that value, so on this model they are the same request twice. The runner probes for it with one verdict and refuses rather than recording two identical columns under two labels. The shipped floor is measured at the model's single setting, which is what `--randomness 0.0` means here.

The second is the floor itself, and what this page said about it was three separate mistakes stacked on one number, so it is worth taking apart rather than editing in place.

`data/noise_floor.json` is the source of truth and it records two columns, each 30 answers judged five times. What it stores per column is a per question count of how many of the five repeats broke ranks, and every figure below is derived from that map, so all of them can be recomputed from the committed file without paying a model:

```
uv run python3 -c "from evals.calibration import recorded_noise_floor as f; c = f().at_configured; print(c.unanimity_rate, c.flip_floor, c.spread)"
```

The number this page used to print, **0.533**, was `1 - unanimity_rate`: the chance that at least one of five repeats disagreed with the other four. That is a real measurement and a useless threshold. It grows with the number of repeats no matter how steady the judge is, so it says as much about `judge.repeats` as about the judge. Every relation bounds a rate at which *one* second reading moved, so the quantity it needs is a rate per **pair** of readings, and that is `flip_floor`: **0.247** on the configured column, 0.273 on the other. The gap between 0.500 and 0.247 is not a correction to a measurement, it is two different measurements that were being read as one.

Two smaller errors travelled with it. "The judge agreed with itself on 47%" was the *other* column's unanimity rate quoted beside the configured column's floor, and the two columns are 0.500 and 0.467. And "a pass rate ranging 0.433 to 0.6" appears nowhere in the file: the recorded per repeat pass rates run 0.633 to 0.8 on the configured column and 0.667 to 0.833 on the other.

The consequence is the part worth reading. Every relation asserts `rate <= 2 * floor`. At 0.500 that bar is 1.000, a rate is a proportion, and no proportion exceeds 1.0, so all four assertions were vacuous and the fixture refused on exactly that ground, which is why `-m reporting` used to error before a single relation ran. Those four bias mutations had therefore never been executed by any assertion anywhere. At 0.247 the bar is 0.493, which a genuinely biased judge can exceed, and the fixture's `floor < 0.5` gate now passes. Nothing about the judge changed. We were bounding it with the wrong statistic, and this page recommended the wrong statistic in bold.

That does not mean they run. They still carry the `reporting` marker, the default options still deselect it, and no CI job undoes that, so the reason these five do not execute has simply moved from "the fixture errors first" to "nothing selects them". Nobody has paid for a run yet.

One caveat to carry into reading a result when somebody does. The floor was recorded over `task_073` to `task_102`, and the `small` fixture these relations run on is `task_001` to `task_040`: the two samples do not overlap at all. That is defensible on purpose, because the floor is meant to be a property of the judge rather than of those thirty answers, and disjoint samples stop the floor and the rate it bounds from sharing their sampling noise. But it means a relation that fails narrowly may be reporting a difference between two subsets rather than a bias. Re-record the floor over `small` before believing a close call.

The whole suite reaches a real embedding model, so `just test` costs a few cents and fails on a fork, which cannot read repository secrets. That is the price of [running real models everywhere](why-every-run-uses-real-models.md).
