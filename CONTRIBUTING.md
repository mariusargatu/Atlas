# Contributing

## Read this first: CI cannot pass on your fork

Every job in CI reaches a real embedding model, and the ones that generate or judge reach a real language model too. There is no keyless tier left. [Why every run uses real models](docs/why-every-run-uses-real-models.md) says what that bought and what it cost. GitHub does not give repository secrets to workflows triggered by a pull request from a fork, so **every check on your PR will fail, and that is expected**. A maintainer runs them again from a branch in this repository before merging.

Do not try to fix this with `pull_request_target`. That trigger runs with secrets in scope against the fork's code, which hands an API key to anyone who opens a pull request.

Run the suite locally instead. It costs a few cents:

```
uv sync
cp .env.example .env     # then put an OpenAI key in it
just test
```

## The rules that are enforced rather than requested

These fail the build, so you will find out anyway. Knowing in advance is faster.

- **File length is a review conversation, not a gate.** There used to be a 400 line cap enforced in CI. It was removed: a guideline that blocks a merge stops being a guideline, and this one had begun deciding how the subject was laid out, splitting one argument across three files so each would fit. If a file has grown a second responsibility, say so in review and split it for that reason.
- **`src/atlas` never imports `evals`.** The pipeline does not know it is being measured. The arrow the other way is fine and used everywhere.
- **Adding a test means bumping `tests/EXPECTED_MIN_TESTS`** in the same change. A floor on its own is satisfied forever by a suite that stopped growing, so the contract test wants the floor within five of the real count.
- **No skipped tests.** A skip turns into a failed session. A failure you declared in advance with a marker is the only exemption.
- **Editing a prompt means moving its version.** Both prompt templates and the rubric carry a version in their front matter, a matching field in the config, and a pinned hash of the body in the tests. All three move together or the build fails. This is not bureaucracy: the version is part of `run_id`, and two different prompts under one version make two runs you cannot compare look comparable. It has happened here.
- **Every command in the justfile appears in the README**, and every relative link in the docs resolves.

## Changing a number

Anything under `data/results/` is a measurement, not a fixture. If your change moves a published figure, record it again (`just report`, plus `--generate --judge` or `--contrast` if the change reaches those stages) and say in the PR which numbers moved and why. If you cannot afford to record a paid grain again, say so. The store is designed to hold rows from several configurations at once, and an honest gap beats a stale number.

`data/labels.jsonl` is the exception and the one artefact that cannot be regenerated at any price. Only add rows to it through `just label`, and only from your own judgement.

## Style

Match the file you are editing. The comments here tend to explain why something is the way it is, usually by naming the failure that produced it. That is deliberate, and a comment that only restates the code is worth deleting. `just lint` and `just types` are the mechanical half.

## Reporting something wrong with a number

That is the most valuable kind of issue this repository can receive. Include the run id, the command, and what you expected. `just report --render --run-id <id>` prints the settings tree that produced any recorded row, and needs no API key.
