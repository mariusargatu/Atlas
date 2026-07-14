"""SP10 task 4: `.github/workflows/simulator.yml` stays wired to what it claims to run, not just
declared to run it (this repo's own CLAUDE.md warning applied to the workflow itself, the SAME
discipline `test_live_pr_workflow.py`/`test_full_sweep_workflow.py` already established). Every
Taskfile target and script path the workflow names must actually exist, and the provider key env
var names the workflow reads must match what `judge.simulator_lane` itself reads.
"""
from __future__ import annotations

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "simulator.yml"
TASKFILE_PATH = ROOT / "Taskfile.yml"
DRIVER_PATH = ROOT / "testing" / "harness" / "judge" / "simulator_lane.py"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def _taskfile_tasks() -> dict:
    return yaml.safe_load(TASKFILE_PATH.read_text())["tasks"]


def test_workflow_file_parses_and_triggers_on_workflow_dispatch_only():
    wf = _workflow()
    # PyYAML parses the bare `on:` key as the boolean True, not the string "on" -- the SAME YAML 1.1
    # quirk `test_live_pr_workflow.py`/`test_full_sweep_workflow.py` already work around.
    triggers = wf.get("on", wf.get(True))
    assert set(triggers) == {"workflow_dispatch"}, "the Simulator lane must trigger on workflow_dispatch only (manual)"


def test_workflow_is_not_also_a_scheduled_cron():
    wf = _workflow()
    triggers = wf.get("on", wf.get(True))
    assert "schedule" not in triggers
    assert "pull_request" not in triggers
    assert "push" not in triggers


def test_every_literal_task_invocation_names_a_real_taskfile_target():
    names = set(_taskfile_tasks())
    invoked = set(re.findall(r"\btask ([a-zA-Z][\w:-]*)", WORKFLOW_PATH.read_text()))
    assert invoked, "expected simulator.yml to invoke at least one task by name"
    unknown = invoked - names
    assert not unknown, f"simulator.yml invokes task target(s) that do not exist: {unknown}"
    assert {"simulation", "rag:ingest", "simulation:live"} <= invoked


def test_simulation_live_task_exists_and_runs_the_committed_driver_script():
    cmds = _taskfile_tasks()["simulation:live"]["cmds"]
    assert any("testing/harness/judge/simulator_lane.py" in cmd for cmd in cmds)
    assert DRIVER_PATH.is_file(), "the driver script the Taskfile target names must actually exist"


def test_simulation_task_exists():
    assert "simulation" in _taskfile_tasks()


def test_rag_ingest_task_exists():
    assert "rag:ingest" in _taskfile_tasks()


def test_provider_key_env_var_names_match_what_the_driver_actually_reads():
    """The workflow's gate step and the `simulation:live` step both read `OPENAI_API_KEY` /
    `ANTHROPIC_API_KEY`; `judge.simulator_lane.select_driver_and_evaluator_tiers` must read the SAME
    two names, never a silently renamed pair on one side only."""
    driver_source = DRIVER_PATH.read_text()
    workflow_text = WORKFLOW_PATH.read_text()
    for env_var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert env_var in driver_source, f"judge/simulator_lane.py no longer reads {env_var}"
        assert f"secrets.{env_var}" in workflow_text, f"simulator.yml no longer reads secrets.{env_var}"


def test_workflow_requires_both_keys_not_either_one():
    """The Simulator lane's own cross model boundary (unlike Live PR/Full sweep's 'either one'
    rule): the gate step must fail closed unless BOTH keys are present, an `||` (OR) test on the two
    `-z` (empty) checks, never `&&` (which would wrongly treat one configured key as enough)."""
    wf = _workflow()
    gate_step = next(s for s in wf["jobs"]["live-simulation"]["steps"] if s.get("id") == "gate")
    run_text = gate_step["run"]
    assert "-z \"${OPENAI_API_KEY:-}\" || -z \"${ANTHROPIC_API_KEY:-}\"" in run_text.replace("\n", " ")


def test_workflow_header_names_the_infrastructure_dependency_disclosure():
    """The digest's own 3g: 'manual, pre burst stage' must not be read as infrastructure free."""
    text = WORKFLOW_PATH.read_text()
    assert "NOT infrastructure free" in text or "NOT INFRASTRUCTURE FREE" in text.upper()
    assert "TEI" in text


def test_workflow_header_names_pass_k_and_cross_model_boundary():
    text = WORKFLOW_PATH.read_text()
    assert "pass^k" in text or "pass to the k power" in text.lower()
    assert "k=4" in text
    assert "cross model" in text.lower()


def test_workflow_header_names_dormant_until_pushed():
    text = WORKFLOW_PATH.read_text()
    assert "DORMANT UNTIL PUSHED" in text.upper()


def test_workflow_never_gates_by_name():
    """A textual guard against a D18 regression, the SAME discipline
    `test_live_pr_workflow.py::test_workflow_never_gates_on_the_judge_verdict_by_name` and
    `test_full_sweep_workflow.py::test_workflow_never_gates_on_the_report_by_name` already apply to
    their own lanes: the upload step is explicitly `if: always()`, reported never blocking."""
    wf = _workflow()
    steps = wf["jobs"]["live-simulation"]["steps"]
    upload_step = next(s for s in steps if s.get("name", "").startswith("Upload"))
    assert upload_step["if"].startswith("always()")


def test_no_hardcoded_secret_values_only_secrets_context_references():
    text = WORKFLOW_PATH.read_text()
    assert "sk-" not in text  # no literal key prefix ever belongs in a committed workflow
