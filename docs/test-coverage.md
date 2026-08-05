# Test coverage

Coverage is the share of your code that runs at least once while the tests run. If a module reports 30%, then seven lines in ten never ran during any test. It is a cheap number to produce and an easy one to misread, so this page says exactly what we do with it. We measure it in one place, we print it, and we let it block a merge on exactly one thing: a regression against the last number we committed. It has never blocked on an absolute floor, and still does not.

## Where it is measured

One CI job, named `coverage`, running `scripts/coverage_delta.py`. That script runs the same tests `just test` does with the measurement switched on, the same command this page has always documented:

```
uv run pytest --cov=src/atlas --cov=evals --cov-report=json
```

then compares the total against `data/coverage_baseline.json`, a single committed percentage. A total more than one point below the baseline fails the job; anything else passes, including a total that rose, which the script says is worth committing as the new baseline in the same change so the next comparison is against where things actually are.

This was `continue-on-error: true` and blocked nothing, which is most of why this page exists: read on for the case against a floor, which still holds, and for what changed and why in "The hole this used to leave" below.

The other nine jobs block outright, with no comparison to make: they are named in a list the repository checks read, and a test asserts that list and the workflow agree, so a job cannot quietly change tier without somebody editing that line.

Nothing else measures coverage. `just test` is plain `uv run pytest`, and the default options carry no `--cov`, so a full local run produces no coverage figure at all. `pytest-cov` is in the dev dependency group, so the tooling is present. It is simply off unless you ask for it. When you do ask, it writes a `.coverage` database in the repository root, and `.gitignore` lists that file, so it never reaches the repository.

## Why there is no floor

A common rule is 80% or the merge is blocked. We do not have a floor anywhere: no `--cov-fail-under` on the command, no `.coveragerc`, no coverage section in `pyproject.toml`.

The reason is what a floor rewards. Coverage counts lines that ran. It cannot tell whether anything was checked when they ran. A test that imports a module, calls every function and asserts nothing scores exactly the same as a test that pins the behaviour down. So when a percentage stands between somebody and a merge, the quickest way past it is the first kind of test. You get more tests, a higher number, and no more confidence than you started with. Now the number is also lying, which is worse than not having it.

This repository is about measurement. A metric that can be raised without doing the work it claims to stand for is the exact failure the rest of these pages are about, so we are not going to install one at the front door.

## What blocks instead

Nine jobs check behaviour rather than execution, and a tenth, `coverage`, blocks on a different question entirely, whether the total moved backward, which is covered on its own further down:

| Job | What it holds |
|---|---|
| `lint-and-types` | ruff over everything, mypy over `src/atlas`, `evals` and `scripts` |
| `suite-integrity` | the whole suite, through `just test` |
| `properties` | things that must always be true of the chunker and the metric functions, over inputs the test makes up |
| `contracts` | the types each stage hands the next one, and what the answer writer has to fill in |
| `core-relations` | what must stay true between two runs when we change one thing |
| `validity` | whether the benchmark can tell systems apart. See [Checking the benchmark](checking-the-benchmark.md) |
| `evaluation_numbers` | everything under `tests/measurement/`, the half of the suite that checks how a number is arrived at |
| `adversaries` | seven deliberately broken systems, each caught by a named detector, plus a healthy control no detector may fire on |
| `repo-contract` | the cap on how long a source file may be, links that point at files which exist, and the promises this repository makes about itself |

Every one of those can fail for a reason you can read and act on. That is the difference. A coverage figure tells you a line did not run. It does not tell you what should have happened when it did.

## What the number is still worth

Two things now, where there used to be one. If a module sits far below everything around it after all of the tiers above have run, that is a fact about the tiers, not about the module: none of them reach that code, which is a prompt to go and look, and often the fastest way to find a stage nobody is really testing. That reading was always available and still is; it takes a person choosing to look at the per-file table, and nothing forces that.

The second is new and mechanical rather than a prompt: `scripts/coverage_delta.py` actually reads the number back and blocks a merge when the total drops, so a module going from tested to silently untested by a change that deleted its tests, rather than its code, now fails the build instead of only lowering a figure in a log. That is a narrower claim than "you should look at this," and a cheaper one to make good on.

## The hole this used to leave

Coverage existed as text in a job log that `continue-on-error` let go red without consequence, and nothing in the repository read it back. A number nobody reads and nothing acts on is worse than no rule at all, because it implies a check that is not really happening. `scripts/coverage_delta.py` closes that specific hole: the number is read back, and a genuine drop blocks.

Two holes remain, both about what the check still cannot see rather than about anyone failing to look.

You cannot see it locally without paying. Every check that builds an index embeds against a real model, and the test session refuses to start without `OPENAI_API_KEY`, so a full run costs a few cents. The coverage job pays that too, same as before.

The scripts are not in the measurement. The command covers `src/atlas` and `evals` only, while the import-direction walker reaches the scripts as well. So the entry points that produce every published number are outside the coverage figure entirely, regression check included.

## What we got wrong

This page used to be a decision record, and that record described a mechanism nobody had built.

It said coverage was measured on every run and printed in the visible summary, with a module below sixty percent treated as a signal rather than a failure. None of that was true. The run summary prints wall clock and money spent, and it has never carried a coverage figure. Grep the tree for a sixty percent threshold and there is nothing to find, because no code ever computed one. The record was dated 30 July 2026, and the coverage job and `pytest-cov` both arrived the next day, in commit `1f105fd`, so on the day that sentence was written nothing in the repository measured coverage at all.

The rule itself came through all of that intact, which is the useful part, and it held for longer than this page's own history: coverage really was measured and really was never a gate, an absolute one least of all, from the day it first ran until the regression check above replaced `continue-on-error`. What decayed the first time was the prose around it, and it decayed in the most flattering direction: the record described the version of the mechanism we would have liked to have. That is the normal way documentation goes wrong. Not a lie, an intention left in the present tense until it read like a fact.

## Seeing it yourself

```
uv run pytest --cov=src/atlas --cov=evals --cov-report=term
```

That prints the same per-file table a human reads, over the same tests `just test` collects. `uv run python3 scripts/coverage_delta.py` is what CI actually runs, and prints the total against the committed baseline instead of the per-file table. Both embed against real models, so both cost real money. If you only want to know what blocks a merge, you do not need either: a genuine regression is the only way this ever fails.
