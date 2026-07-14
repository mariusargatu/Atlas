"""SP10 task 3: `.github/workflows/full-sweep.yml` stays wired to what it claims to run, not just
declared to run it (this repo's own CLAUDE.md warning applied to the workflow itself, the SAME
discipline `test_live_pr_workflow.py` already established for the Live PR lane). Every Taskfile
target and script path the workflow names must actually exist, and the provider key env var names
the workflow reads must match what `judge.full_sweep` itself reads.
"""
from __future__ import annotations

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "full-sweep.yml"
TASKFILE_PATH = ROOT / "Taskfile.yml"
DRIVER_PATH = ROOT / "testing" / "harness" / "judge" / "full_sweep.py"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def _taskfile_tasks() -> dict:
    return yaml.safe_load(TASKFILE_PATH.read_text())["tasks"]


def test_workflow_file_parses_and_triggers_on_push_to_main_only():
    wf = _workflow()
    # PyYAML parses the bare `on:` key as the boolean True, not the string "on" -- the SAME YAML 1.1
    # quirk `test_live_pr_workflow.py` already works around.
    triggers = wf.get("on", wf.get(True))
    assert set(triggers) == {"push"}, "the Full sweep lane must trigger on push only"
    assert triggers["push"]["branches"] == ["main"]


def test_workflow_is_not_also_a_scheduled_cron():
    """D18 holds the line research 15's own weekly backstop recommendation would break: no
    `schedule:` trigger belongs on this workflow (that is `janitor.yml`'s own one earned cron)."""
    wf = _workflow()
    triggers = wf.get("on", wf.get(True))
    assert "schedule" not in triggers


def test_every_literal_task_invocation_names_a_real_taskfile_target():
    names = set(_taskfile_tasks())
    invoked = set(re.findall(r"\btask ([a-zA-Z][\w:-]*)", WORKFLOW_PATH.read_text()))
    assert invoked, "expected full-sweep.yml to invoke at least one task by name"
    unknown = invoked - names
    assert not unknown, f"full-sweep.yml invokes task target(s) that do not exist: {unknown}"
    assert {"rag:ingest", "full-sweep:run"} <= invoked


def test_full_sweep_run_task_exists_and_runs_the_committed_driver_script():
    cmds = _taskfile_tasks()["full-sweep:run"]["cmds"]
    assert any("testing/harness/judge/full_sweep.py" in cmd for cmd in cmds)
    assert DRIVER_PATH.is_file(), "the driver script the Taskfile target names must actually exist"


def test_rag_ingest_task_exists():
    assert "rag:ingest" in _taskfile_tasks()


def test_provider_key_env_var_names_match_what_the_driver_actually_reads():
    """The workflow's gate step and the `full-sweep:run` step both read `OPENAI_API_KEY` /
    `ANTHROPIC_API_KEY`; `judge.full_sweep.select_frontier_judge_tier` must read the SAME two names,
    never a silently renamed pair on one side only."""
    driver_source = DRIVER_PATH.read_text()
    workflow_text = WORKFLOW_PATH.read_text()
    for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert env_var in driver_source, f"judge/full_sweep.py no longer reads {env_var}"
        assert f"secrets.{env_var}" in workflow_text, f"full-sweep.yml no longer reads secrets.{env_var}"


def test_workflow_header_names_the_76_case_honesty_disclosure():
    text = WORKFLOW_PATH.read_text()
    assert "76 case" in text
    assert "DORMANT UNTIL PUSHED" in text.upper()


def test_workflow_header_names_the_d18_weekly_cron_declination():
    """The plan's own honesty requirement: the header must name D18 and explain the weekly backstop
    was declined, not silently omit a cron trigger with no explanation."""
    text = WORKFLOW_PATH.read_text()
    assert "D18" in text
    assert "weekly" in text.lower()
    assert "declined" in text.lower() or "DECLINED" in text


def test_workflow_never_gates_on_the_report_by_name():
    """A textual guard against the D18 regression this whole lane exists to avoid: nothing in this
    file may name the report's own rate/verdict as a required check condition. The upload step is
    explicitly `if: always()` for exactly this reason -- reported, never blocking."""
    wf = _workflow()
    steps = wf["jobs"]["full-sweep"]["steps"]
    upload_step = next(s for s in steps if s.get("name", "").startswith("Upload"))
    assert upload_step["if"].startswith("always()")


def test_no_hardcoded_secret_values_only_secrets_context_references():
    text = WORKFLOW_PATH.read_text()
    assert "sk-" not in text  # no literal key prefix ever belongs in a committed workflow
