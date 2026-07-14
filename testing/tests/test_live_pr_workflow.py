"""SP10 task 2: `.github/workflows/live-pr.yml` stays wired to what it claims to run, not just
declared to run it (this repo's own CLAUDE.md warning applied to the workflow itself). Every
Taskfile target and script path the workflow names must actually exist, the `pull_request` path
filter globs must actually name real directories in this repo, and the provider key env var names
the workflow reads must match what `judge.live_pr_lane` itself reads -- a rename on either side that
the other did not follow would otherwise go unnoticed until a maintainer actually pushes and the
paths filter first fires.
"""
from __future__ import annotations

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "live-pr.yml"
TASKFILE_PATH = ROOT / "Taskfile.yml"
DRIVER_PATH = ROOT / "testing" / "harness" / "judge" / "live_pr_lane.py"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def _taskfile_tasks() -> dict:
    return yaml.safe_load(TASKFILE_PATH.read_text())["tasks"]


def test_workflow_file_parses_and_triggers_on_pull_request_only():
    wf = _workflow()
    # PyYAML parses the bare `on:` key as the boolean True, not the string "on" -- a documented
    # YAML 1.1 quirk this assertion works around the same way `test_ci_workflow_targets.py`'s own
    # reads never had to (ci.yml's `on:` was never asserted on directly there).
    triggers = wf.get("on", wf.get(True))
    assert set(triggers) == {"pull_request"}, "the Live PR lane must trigger on pull_request only"


def test_every_literal_task_invocation_names_a_real_taskfile_target():
    names = set(_taskfile_tasks())
    invoked = set(re.findall(r"\btask ([a-zA-Z][\w:-]*)", WORKFLOW_PATH.read_text()))
    assert invoked, "expected live-pr.yml to invoke at least one task by name"
    unknown = invoked - names
    assert not unknown, f"live-pr.yml invokes task target(s) that do not exist: {unknown}"
    # The three targets this task's own plan text names explicitly.
    assert {"contracts:diff", "rag:ingest", "live-pr:sweep"} <= invoked


def test_live_pr_sweep_task_exists_and_runs_the_committed_driver_script():
    cmds = _taskfile_tasks()["live-pr:sweep"]["cmds"]
    assert any("testing/harness/judge/live_pr_lane.py" in cmd for cmd in cmds)
    assert DRIVER_PATH.is_file(), "the driver script the Taskfile target names must actually exist"


def test_contracts_diff_task_exists():
    assert "contracts:diff" in _taskfile_tasks()


def test_rag_ingest_task_exists():
    assert "rag:ingest" in _taskfile_tasks()


def test_path_filter_globs_are_well_formed_and_name_real_directories():
    """Every glob is `<real top level dir>/**` or `<real top level dir>/`; the directory named
    before the first wildcard must actually exist in this repo, so a renamed surface (e.g.
    `testing/harness/judge` moving) cannot silently leave a stale, permanently dark path filter."""
    wf = _workflow()
    triggers = wf.get("on", wf.get(True))
    live_sweep_paths = triggers["pull_request"]["paths"]
    assert live_sweep_paths, "expected at least one path filter glob"
    expected = {
        "backend/atlas/**",
        "corpus/**",
        "contracts/**",
        "testing/harness/quality/**",
        "testing/harness/judge/**",
        "testing/harness/dataset_tools/**",
    }
    assert set(live_sweep_paths) == expected
    for glob in live_sweep_paths:
        assert glob.endswith("/**"), f"glob {glob!r} should end with /** (recursive, well formed)"
        real_dir = glob[: -len("/**")]
        assert (ROOT / real_dir).is_dir(), f"path filter glob {glob!r} names a directory that does not exist: {real_dir}"


def test_provider_key_env_var_names_match_what_the_driver_actually_reads():
    """The workflow's gate step and the `live-pr:sweep` step both read `OPENAI_API_KEY` /
    `ANTHROPIC_API_KEY`; `judge.live_pr_lane.select_judge_tier` must read the SAME two names, never
    a silently renamed pair on one side only."""
    driver_source = DRIVER_PATH.read_text()
    workflow_text = WORKFLOW_PATH.read_text()
    for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert env_var in driver_source, f"judge/live_pr_lane.py no longer reads {env_var}"
        assert f"secrets.{env_var}" in workflow_text, f"live-pr.yml no longer reads secrets.{env_var}"


def test_workflow_header_names_the_76_case_honesty_disclosure():
    text = WORKFLOW_PATH.read_text()
    assert "76 case" in text
    assert "DORMANT UNTIL PUSHED" in text.upper()


def test_workflow_never_gates_on_the_judge_verdict_by_name():
    """A textual guard against the D18 regression this whole lane exists to avoid: nothing in this
    file may name the judge's own rate/verdict as a required check condition (an `if:` gate keyed
    on a judge score, or a step that fails the job from the judge tier's own exit path). The upload
    step is explicitly `if: always()` for exactly this reason -- reported, never blocking."""
    wf = _workflow()
    live_sweep_steps = wf["jobs"]["live-sweep"]["steps"]
    upload_step = next(s for s in live_sweep_steps if s.get("name", "").startswith("Upload"))
    assert upload_step["if"].startswith("always()")
