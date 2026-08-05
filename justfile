# The one bootstrap every entry point gets. These two lines used to be three separate
# arrangements: this PYTHONPATH, plus a hand-rolled sys.path insert and a load_dotenv at
# the top of two scripts, plus the pair in tests/conftest.py. One entry point had
# neither of the first two and so could not read a key out of .env at all, which made
# the README's opening sequence fail on missing credentials. Every recipe below now gets
# the path and the file, once, from here.
#
# tests/conftest.py keeps its own load_dotenv on purpose: CI runs `uv run pytest`
# directly rather than through a recipe, so pytest needs its own. That is one bootstrap
# per runner, each in exactly one place, rather than three for the same runner.
export PYTHONPATH := "src:."
set dotenv-load := true

# Every variadic recipe below passes "$@" rather than {{args}}. `just` splices {{args}}
# into a line that `sh -cu` then re-parses, so the question the README opens with breaks
# the moment it contains a character the shell reads: `just ask "What's the fee?"` died on
# `unexpected EOF while looking for matching '`, and `a $500 transfer?` on `$5: unbound
# variable`. Worse were the quiet ones -- `Is 3 > 2 a problem?` redirected the whole run
# into a file called `2` and printed nothing, and `checking & savings transfers?`
# backgrounded a *paid* run on a truncated question while just reported a failed recipe.
#
# The obvious repair is worse than the fault and was tried first. Both "{{args}}" and
# {{quote(args)}} collapse every argument into one shell token, and argparse hands a token
# containing a space to the `nargs="*"` positional rather than reading it as a flag -- so
# `just ask --dry-run "..."` arrived as the single question "--dry-run ..." and ran a full
# paid generation instead of stopping before it. This setting is what makes "$@" carry the
# arguments through as the separate, still-quoted words they started as.
set positional-arguments := true

# the whole suite; embeds the corpus, so it costs a few cents
test:
    uv run pytest

# ruff over everything
lint:
    uv run ruff check .

# scripts is checked too, and not as an afterthought: three entry points went on
# importing modules a refactor had deleted while lint, types and 169 tests stayed
# green, because nothing here had ever looked at scripts.
# mypy over src, evals and scripts
types:
    uv run mypy src/atlas evals scripts

# Variadic, so `just ask --list` and `just ask --dry-run "..."` both reach the script.
# It took one quoted argument before, which meant the two flags in ask.py's own usage
# block were reachable only by invoking python directly.
# one question through every stage, with --dry-run to stop before generation
ask *args:
    uv run python3 scripts/ask.py "$@"

# The one command a chapter's table comes from. Retrieval only by default: free against a
# warm .cache/vectors. --generate and --judge are opt in because they are the parts that
# cost real money, and --render reads the recorded store and needs no key at all, so a
# reader reaches a table before they reach for a card.
#
# --contrast adds deepeval's own faithfulness, answer relevancy and contextual relevancy
# beside this repository's judge, and reports the chance-corrected agreement between them.
# It implies --judge and is the most expensive thing here: four scorers per answered
# question, on top of the answer and the verdict.
# score the question set, record the run, print the table (--render needs no key)
report *args:
    uv run python3 scripts/report.py "$@"

# Is a difference between two configurations real, or is it noise? Every decision record
# that says one arrangement beat another quotes an interval, and this is what reproduces
# it. Paired on the question, so the variance the two share cancels instead of counting
# twice. Free against a warm .cache/vectors.
# paired bootstrap between two settings files
compare *args:
    uv run python3 scripts/compare.py "$@"

# record one human verdict into the label store
label *args:
    uv run python3 scripts/label.py "$@"

# How often the judge disagrees with itself on the same answer. Every judge relation in
# tests/measurement/test_judge.py is a multiple of this number, and those five tests are
# the repository's only `reporting`-marked ones: without a recorded floor they abort in
# their fixture rather than measuring, which is why they run nowhere by default.
#
# Ten verdicts per question at the default repeats, about $0.055 each, so start with
# --limit. It re-judges answers already bought rather than writing new ones.
# measure the judge's disagreement with itself and write data/noise_floor.json
noise-floor *args:
    uv run python3 scripts/noise_floor.py "$@"

# The whole trace setup, in one command. --wait blocks until every service reports
# healthy, which matters because the first boot runs database migrations and ClickHouse
# setup and `up -d` alone returns in the middle of them.
# start the local Langfuse stack and wait for it to be healthy
up:
    @echo "starting the trace stack. first run pulls ~1.2 GB and takes several minutes; later runs about 30 seconds."
    docker compose -f deploy/docker-compose.yml up -d --wait
    @echo ""
    @echo "langfuse is up:  http://localhost:${LANGFUSE_PORT:-3000}"
    @echo "sign in as       atlas@example.com / not-a-secret"
    @echo ""
    @echo "that password is in this repository, and so are the fourteen other defaults in"
    @echo "deploy/docker-compose.yml. Nothing here is reachable from another machine, which"
    @echo "is what makes that fine. Do not put this on a network."

# stop the local Langfuse stack
down:
    docker compose -f deploy/docker-compose.yml down
