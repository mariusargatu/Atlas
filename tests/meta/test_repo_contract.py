"""What the repository promises about itself: links, CI tiers, and test-count floors."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.repo_checks import (
    BLOCKING_JOBS,
    REQUIRED_KEY_NAME,
    blocking_jobs,
    broken_links,
    jobs_in,
    justfile_recipes,
    pytest_jobs,
    secrets_read,
    source_files_on_disk,
    workflow_secrets,
)

ROOT = Path(__file__).resolve().parents[2]
FLOOR_FILE = ROOT / "tests" / "EXPECTED_MIN_TESTS"
CEILING_MARGIN = 5

# A suffix list rather than the two keys this repository reads: a machine carrying some
# other provider's key is still a machine the keyless check must describe as having none.
CREDENTIAL_SUFFIXES = ("_API_KEY", "_SECRET_KEY", "_TOKEN", "_PASSWORD")


def test_the_source_walker_descends_into_every_subpackage():
    # Stops the walk from quietly narrowing: a plain glob once stopped at the top level and
    # exempted every module inside a subpackage, which the rules reading this corpus would
    # not have noticed.
    inspected = {p.relative_to(ROOT).as_posix() for p in source_files_on_disk()}
    assert "src/atlas/contracts.py" in inspected, "the walker missed the top level"
    for subpackage in ("corpus", "retrieval", "models"):
        assert any(p.startswith(f"src/atlas/{subpackage}/") for p in inspected), (
            f"the walker never descended into src/atlas/{subpackage}"
        )
    assert any(p.startswith("evals/") for p in inspected), "the walker missed evals"
    assert any(p.startswith("scripts/") for p in inspected), "the walker missed scripts"


def test_every_relative_link_resolves():
    assert broken_links().broken == []


def test_the_link_check_found_links_to_resolve():
    # Without this, deleting every linked document turns the check above green.
    assert broken_links().checked > 0


def test_every_job_the_repository_marks_as_blocking_exists_and_gates_merge():
    # A check wired into no merge gating job would otherwise go unnoticed by the whole suite.
    assert set(blocking_jobs()) == BLOCKING_JOBS


def test_every_job_can_reach_the_key_the_pipeline_needs():
    # Every job runs real models, so a job that cannot reach the key does not run cheaply,
    # it dies on the first embedding call. Read at the workflow level, because a per-job
    # assertion would force one shared key into thirteen copies to satisfy the check.
    assert REQUIRED_KEY_NAME in workflow_secrets(), (
        f"no job can reach {REQUIRED_KEY_NAME}, and every job needs it"
    )


def test_no_job_names_a_key_that_nothing_reads():
    # A secret nobody reads is worse than a missing one: it looks wired up. Three once were,
    # and the jobs meant to exercise the paid path ran against no credentials and passed.
    # Every job, not only the blocking ones: continue-on-error is where a job that quietly
    # does nothing survives longest.
    known = {REQUIRED_KEY_NAME, "ANTHROPIC_API_KEY"}
    named = workflow_secrets() | {
        s for job in jobs_in("ci.yml") for s in secrets_read(job)
    }
    assert named <= known, f"{sorted(named - known)} is read by no client in this repository"


def test_no_blocking_job_runs_the_full_reseeding_command():
    # reseed-full re-embeds the entire collection per seed: minutes of wall clock and a real
    # bill, whatever the tier it sits in is allowed to spend.
    for key, job in blocking_jobs().items():
        assert "reseed-full" not in job.raw, f"{key} runs the full reseeding command"


def _collected(arguments: list[str]) -> int | None:
    """How many tests a selector actually collects, or None if collection failed."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         *arguments],
        capture_output=True, text=True, check=False, cwd=ROOT,
    )
    # The first number, not the total after the slash: with deselections pytest prints
    # "X/Y tests collected (Z deselected)". Exit code 5 means nothing was collected, which
    # is a real answer rather than a failure to get one.
    summary = re.search(r"(\d+)(?:/\d+)? tests? collected", result.stdout)
    if summary is not None:
        return int(summary.group(1))
    return 0 if "no tests ran" in result.stdout or result.returncode == 5 else None


def test_every_pytest_job_collects_tests_and_every_floor_key_names_one():
    # Three sides of one failure. A selector that collects nothing exits 5 and
    # continue-on-error paints it green; a floor key naming no job is read by nothing; and a
    # floor without a ceiling is satisfied forever by a suite that stopped growing. Problems
    # accumulate so one red test names every offender rather than only the first.
    floors = json.loads(FLOOR_FILE.read_text(encoding="utf-8"))
    jobs = pytest_jobs()
    problems: list[str] = []

    unmatched = sorted(set(floors) - set(jobs))
    if unmatched:
        problems.append(f"{unmatched} in {FLOOR_FILE.name} name no pytest job in ci.yml")

    for key, arguments in sorted(jobs.items()):
        collected = _collected(arguments)
        if collected is None:
            problems.append(f"{key}: collection itself failed under {arguments}")
            continue
        if collected == 0:
            problems.append(f"{key}: `pytest {' '.join(arguments)}` collects nothing, so the "
                            "job asserts nothing and continue-on-error hides it")
            continue
        floor = floors.get(key)
        if floor is None:
            continue
        if not floor <= collected <= floor + CEILING_MARGIN:
            problems.append(
                f"{key}: {collected} collected, floor {floor}, ceiling {floor + CEILING_MARGIN}. "
                f"Set {key} to {collected} in {FLOOR_FILE.name} as part of this piece of work"
            )

    assert problems == [], "\n".join(problems)


@pytest.fixture
def clean_environment() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.endswith(CREDENTIAL_SUFFIXES)}
    assert [k for k in env if k.endswith(CREDENTIAL_SUFFIXES)] == []
    return env


def test_the_pipeline_refuses_to_start_without_a_key_rather_than_failing_part_way(
    clean_environment,
):
    # The property belongs to `PreparedCorpus.build`, which constructs both model clients
    # eagerly: a missing key discovered at the answering stage means the whole corpus was
    # embedded and thrown away, a real bill for nothing.
    #
    # --no-dotenv because the justfile loads .env for every recipe, so a developer's real
    # .env would hand this subprocess a live key and the assertion could never fail.
    completed = subprocess.run(
        shlex.split("just --no-dotenv ask --dry-run how much does a dispute cost"),
        env=clean_environment, cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    assert completed.returncode != 0, "the pipeline ran without a key, which it cannot do"
    output = completed.stdout + completed.stderr
    assert "OPENAI_API_KEY" in output, f"the failure never named the key:\n{output[-2000:]}"
    # A run that got as far as printing what it found had already paid for the corpus.
    assert "retrieved:" not in completed.stdout


def test_every_recipe_a_reader_could_run_is_named_in_the_readme() -> None:
    """A table listing eight of nine commands is not obviously wrong to anybody, which is
    why the README's command table drifted twice before this existed."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # The keys already read "just <name>"; do not prefix them again.
    missing = sorted(command for command in justfile_recipes() if command not in readme)
    assert not missing, f"{missing} can be run but is not named in README.md"
