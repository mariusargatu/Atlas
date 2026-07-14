"""SP10 task 1: verify `.github/workflows/ci.yml`'s hermetic-pr-lane and web-gate jobs stay wired
to the Taskfile.yml targets they mirror, not just declared to mirror them (this repo's own CLAUDE.md
warning applied to the workflow itself: a thing can be "declared and unit tested but not actually
wired into the running graph").

`ci.yml` never calls `task test` or `task web-test` by name. The pytest step runs `uv run pytest -q`
TWICE so the rerun can prove byte stability (a `task test` call would only prove one run passed);
`web-gate` has no Task binary installed at all (`pnpm/action-setup` only), so it runs `web-test`'s
own pnpm commands directly instead. Both jobs therefore MIRROR Taskfile.yml's commands rather than
delegating to it, and a mirror can silently drift: if `test`/`web-test`/`lint` is ever renamed, or
either target's own `cmds:` list changes, ci.yml would keep running the STALE mirrored command with
nothing forcing a review. `task lint` is the one place ci.yml delegates by name; every literal `task
<name>` invocation anywhere in the file must still resolve to a real target too.
"""

from __future__ import annotations

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
CI_WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"
TASKFILE_PATH = ROOT / "Taskfile.yml"


def _taskfile_tasks() -> dict:
    return yaml.safe_load(TASKFILE_PATH.read_text())["tasks"]


def _task_cmds(task_name: str) -> list[str]:
    """A Taskfile target's own `cmds:` list, read fresh (never a copy pasted expectation)."""
    return list(_taskfile_tasks()[task_name]["cmds"])


def _ci_workflow() -> dict:
    return yaml.safe_load(CI_WORKFLOW_PATH.read_text())


def test_lint_test_and_web_test_targets_all_exist_in_taskfile():
    """The three targets this task's own plan text names (lint, test, web-test) must still be real
    Taskfile.yml targets, not renamed or removed out from under ci.yml."""
    names = set(_taskfile_tasks())
    missing = {"lint", "test", "web-test"} - names
    assert not missing, f"Taskfile.yml is missing target(s) ci.yml relies on: {missing}"


def test_every_literal_task_invocation_in_ci_yml_names_a_real_target():
    """`task lint` is ci.yml's one literal by name invocation; it (and any future one) must resolve
    to a real Taskfile.yml target."""
    names = set(_taskfile_tasks())
    invoked = set(re.findall(r"\btask ([a-zA-Z][\w:-]*)", CI_WORKFLOW_PATH.read_text()))
    assert invoked, "expected ci.yml to invoke at least one task by name (task lint)"
    unknown = invoked - names
    assert not unknown, f"ci.yml invokes task target(s) that do not exist: {unknown}"


def test_hermetic_pr_lane_pytest_steps_match_the_test_targets_own_command():
    """The byte stability proof runs `uv run pytest -q` twice instead of calling `task test`; if
    `test`'s own cmd in Taskfile.yml ever changes, this is what would otherwise let ci.yml silently
    keep running the stale command."""
    steps = _ci_workflow()["jobs"]["hermetic-pr-lane"]["steps"]
    pytest_runs = [step["run"] for step in steps if step.get("run", "").strip() == "uv run pytest -q"]
    assert len(pytest_runs) == 2, "expected exactly two byte stability pytest runs in hermetic-pr-lane"
    assert _task_cmds("test") == ["uv run pytest -q"]


def test_web_gate_steps_match_the_web_test_target_commands():
    """web-gate has no Task binary installed, so it mirrors `web-test`'s own pnpm commands as raw
    steps run from the repo root instead of calling `task web-test`. Normalize the `-C frontend`
    prefix and the `frontend/src/api/generated` versus `src/api/generated` path difference (Taskfile's
    own `dir: frontend` runs relative; ci.yml runs from the repo root) and assert the two command
    sequences still agree, in order."""
    steps = _ci_workflow()["jobs"]["web-gate"]["steps"]
    raw_runs = [step["run"] for step in steps if "run" in step]
    # The codegen step joins two commands with `&&`; split them back out so the sequence lines up
    # one for one with Taskfile's own cmds list.
    commands: list[str] = []
    for run in raw_runs:
        commands.extend(part.strip() for part in run.split("&&"))
    normalized = [
        cmd.replace("pnpm -C frontend ", "pnpm ").replace("frontend/src/api/generated", "src/api/generated")
        for cmd in commands
    ]
    assert normalized == _task_cmds("web-test")
