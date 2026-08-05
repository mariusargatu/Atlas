# Security

## Reporting

Open a [private security advisory](https://github.com/mariusargatu/Atlas/security/advisories/new) rather than a public issue. There is no service to attack here, since this is a repository you run locally with your own API key. So the realistic reports are about something in the code putting *your* key or *your* data somewhere it should not go. Those are worth reporting privately first.

## What this repository is, in terms of risk

It is a local pipeline plus a docker compose file. It exposes nothing, listens on nothing you did not start yourself, and stores nothing off your machine except what you send to your model provider.

**Your API key.** Read from `.env`, which is gitignored, and passed to the OpenAI and Anthropic SDKs. It is never written to `data/results/runs.jsonl`, never included in a `run_id`, and never sent to Langfuse. `.env.example` ships with placeholder values that are obviously placeholders.

**The local Langfuse stack.** `deploy/docker-compose.yml` contains about fifteen default credentials, including the login shown by `just up`. Every published port is bound to `127.0.0.1`, which is what makes that acceptable, and `just up` prints "Do not put this on a network" every time it runs. If you change a binding to `0.0.0.0`, change the credentials first.

**Prompt injection and corpus poisoning.** Both prompt templates fence untrusted content: retrieved passages, and the answer the judge is grading, are wrapped in delimiters and declared to be evidence rather than instructions. That is a mitigation, not a guarantee, since no defence written into a prompt ever is, and it matters most if you take up the invitation in the README to point this at documents of your own. The vendored tau2 corpus is a fixed artefact from elsewhere that nothing in this repository writes to.

**What is deliberately not defended.** The chunk ids a model returns as citations are checked against what was actually shown, so a fabricated citation is recorded rather than believed. Beyond that, an answer is model output and is treated as untrusted data by the judge and by nothing else. If you render answers into a web page, escape them yourself.

## Dependencies

`uv.lock` is committed with hashes for every pin, so `uv sync` installs exactly what was tested. The cross encoder `just test` downloads is fetched from huggingface.co by name rather than by revision. If you need a model artefact you can reproduce, pin it and set `HF_HUB_OFFLINE=1`.
