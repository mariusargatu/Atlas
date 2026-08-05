"""Repository hygiene checks: link resolution, and what the CI tiers promise.

Every check returns what it inspected alongside what it found, so a checker pointed at
the wrong directory reports an empty corpus rather than a pass.
"""

from __future__ import annotations

import importlib
import inspect
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

_SOURCE_DIRS = (REPO_ROOT / "src" / "atlas", REPO_ROOT / "evals", REPO_ROOT / "scripts")
_LINKED_DOCS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "NOTICE",
    *sorted((REPO_ROOT / "docs").rglob("*.md")),
)
_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")

# The jobs that must exist and must not be marked continue-on-error.
BLOCKING_JOBS = frozenset({
    "lint-and-types", "properties", "contracts", "core-relations",
    "validity", "adversaries", "suite-integrity", "repo-contract",
    "evaluation_numbers",
    # Blocking on a coverage regression against the committed baseline, never on an
    # absolute floor: see scripts/coverage_delta.py and docs/test-coverage.md.
    "coverage",
})

REQUIRED_KEY_NAME = "OPENAI_API_KEY"


def source_files_on_disk() -> tuple[Path, ...]:
    return tuple(sorted(p for d in _SOURCE_DIRS for p in d.rglob("*.py")))


@dataclass(frozen=True, slots=True)
class LinksResult:
    broken: list[str]
    checked: int


def broken_links() -> LinksResult:
    broken: list[str] = []
    checked = 0
    for path in _LINKED_DOCS:
        if not path.exists():
            continue
        for target in _LINK_PATTERN.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            checked += 1
            if not (path.parent / target.split("#", 1)[0]).resolve().exists():
                broken.append(f"{path.relative_to(REPO_ROOT)}: {target}")
    return LinksResult(broken=broken, checked=checked)


@dataclass(frozen=True, slots=True)
class Job:
    key: str
    name: str
    raw: str
    blocking: bool
    skips_forks: bool


def jobs_in(workflow: str) -> tuple[Job, ...]:
    parsed = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    )["jobs"]
    return tuple(
        Job(
            key=key,
            name=job.get("name", key),
            raw=yaml.safe_dump(job),
            blocking=not job.get("continue-on-error", False),
            skips_forks="fork" in str(job.get("if", "")),
        )
        for key, job in parsed.items()
    )


def blocking_jobs() -> dict[str, Job]:
    return {job.key: job for job in jobs_in("ci.yml") if job.blocking}


def workflow_secrets(workflow: str = "ci.yml") -> set[str]:
    raw = (REPO_ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    top = yaml.safe_load(raw).get("env", {})
    return set(re.findall(r"secrets\.(\w+)", yaml.safe_dump(top)))


def secrets_read(job: Job) -> set[str]:
    return set(re.findall(r"secrets\.(\w+)", job.raw))


_PYTEST_ARGUMENTS = re.compile(r"\buv run pytest\b([^\n]*)")
_RECIPE = re.compile(r"^(\w[\w-]*)(?: \*?\w+)?:\n((?:    .*\n|\n)*)", re.M)


def justfile_recipes() -> dict[str, str]:
    raw = (REPO_ROOT / "justfile").read_text(encoding="utf-8")
    return {f"just {name}": body for name, body in _RECIPE.findall(raw)}


def pytest_jobs(workflow: str = "ci.yml") -> dict[str, list[str]]:
    recipes = justfile_recipes()
    found: dict[str, list[str]] = {}
    for job in jobs_in(workflow):
        for command in re.findall(r"^\s*-?\s*run:\s*(.+)$", job.raw, re.M):
            resolved = recipes.get(command.strip(), command.strip())
            match = _PYTEST_ARGUMENTS.search(resolved)
            if match is not None:
                found[job.key.replace("-", "_")] = shlex.split(match.group(1))
    return found


def collected_test_functions() -> tuple[tuple[str, Callable[..., Any]], ...]:
    # Restricted to functions the module itself defines, so a name imported for reuse
    # elsewhere isn't counted as belonging to it too.
    tests_root = REPO_ROOT / "tests"
    found = []
    for path in sorted(tests_root.rglob("test_*.py")):
        dotted = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
        module = importlib.import_module(dotted)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_") and function.__module__ == module.__name__:
                found.append((f"{path.stem}.{name}", function))
    return tuple(found)
