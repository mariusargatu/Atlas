"""SP10 task 5: `.github/workflows/burst-benchmark.yml` stays wired to what it claims to run, not
just declared to run it (this repo's own CLAUDE.md warning applied to the workflow itself, the SAME
discipline `test_live_pr_workflow.py`/`test_full_sweep_workflow.py`/`test_simulator_workflow.py`
already established). Every Taskfile target and script path the workflow names must actually exist,
every secret name it reads must match what the script it feeds actually checks for, and -- the part
unique to this lane -- the `always()` destroy step must PROVABLY still run under a simulated earlier
step failure, not merely declare `if: always()` and be trusted at face value (matching SP9's own
spend gate tamper rigor: prove the mechanism bites).
"""
from __future__ import annotations

import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "burst-benchmark.yml"
TASKFILE_PATH = ROOT / "Taskfile.yml"
CI_GATE_PATH = ROOT / "testing" / "harness" / "sentinel" / "ci_gate.py"
BURST_UP_PATH = ROOT / "infra" / "scripts" / "burst-up.sh"
BURST_DESTROY_PATH = ROOT / "infra" / "scripts" / "burst-destroy.sh"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text())


def _taskfile_tasks() -> dict:
    return yaml.safe_load(TASKFILE_PATH.read_text())["tasks"]


def _steps() -> list[dict]:
    return _workflow()["jobs"]["burst-benchmark"]["steps"]


# ---- triggers: manual dispatch only, a protected environment, never cancelled mid run --------------


def test_workflow_file_parses_and_triggers_on_workflow_dispatch_only():
    wf = _workflow()
    # PyYAML parses the bare `on:` key as the boolean True, not the string "on" -- the SAME YAML 1.1
    # quirk every other SP10 workflow test already works around.
    triggers = wf.get("on", wf.get(True))
    assert set(triggers) == {"workflow_dispatch"}, "the Burst benchmark lane must trigger on workflow_dispatch only (manual)"


def test_workflow_is_not_also_a_scheduled_cron_or_a_push_pr_trigger():
    wf = _workflow()
    triggers = wf.get("on", wf.get(True))
    assert "schedule" not in triggers
    assert "pull_request" not in triggers
    assert "push" not in triggers


def test_job_declares_the_protected_environment():
    job = _workflow()["jobs"]["burst-benchmark"]
    assert job["environment"] == "burst-benchmark"


def test_concurrency_never_cancels_a_run_already_mid_benchmark():
    """The OIDC substitution's own reasoning trace: a second manual dispatch must queue, never
    cancel, a run that may already have provisioned real, billed infrastructure."""
    wf = _workflow()
    assert wf["concurrency"]["cancel-in-progress"] is False


def test_job_has_a_finite_timeout_so_a_hung_step_cannot_hold_the_tier_up_forever():
    job = _workflow()["jobs"]["burst-benchmark"]
    assert isinstance(job.get("timeout-minutes"), int)
    assert job["timeout-minutes"] > 0


# ---- referenced Taskfile targets / scripts actually exist -------------------------------------------


def test_every_literal_task_invocation_names_a_real_taskfile_target():
    names = set(_taskfile_tasks())
    invoked = set(re.findall(r"\btask ([a-zA-Z][\w:-]*)", WORKFLOW_PATH.read_text()))
    assert invoked, "expected burst-benchmark.yml to invoke at least one task by name"
    unknown = invoked - names
    assert not unknown, f"burst-benchmark.yml invokes task target(s) that do not exist: {unknown}"
    assert {"burst:up", "sentinel:gate", "matrix:live", "load:k6", "load:join", "burst:destroy"} <= invoked


def test_sentinel_gate_task_exists_and_runs_the_committed_ci_gate_module():
    cmds = _taskfile_tasks()["sentinel:gate"]["cmds"]
    assert any("sentinel.ci_gate" in cmd for cmd in cmds)
    assert CI_GATE_PATH.is_file(), "the sentinel CI gate module the Taskfile target names must actually exist"


def test_ci_gate_reuses_run_probe_never_a_second_copy_of_the_query_classes():
    """REUSE NEVER DUPLICATE (the plan's own global constraint), asserted directly against the
    committed module (test_sentinel_ci_gate.py asserts the same thing at the unit level; this is the
    workflow level cross check that the file it invokes is the reusing one)."""
    source = CI_GATE_PATH.read_text()
    assert "from sentinel.probe import" in source
    assert "PROBE_QUERIES" in source
    assert "run_probe" in source


def test_burst_up_and_destroy_scripts_exist():
    assert BURST_UP_PATH.is_file()
    assert BURST_DESTROY_PATH.is_file()


def test_xk6_sse_module_path_is_the_real_phymbert_org_never_the_nonexistent_grafana_one():
    """A correction caught during authoring (this file's own header names it, in prose, for the
    record): github.com/grafana/xk6-sse does not exist; the real, maintained extension lives at
    github.com/phymbert/xk6-sse. The header is ALLOWED to mention the wrong path in prose (naming the
    correction plainly); the actual BUILD COMMAND must never reference it."""
    build_step = next(s for s in _steps() if s.get("name", "").startswith("Build the xk6-sse k6 binary"))
    run_text = build_step["run"]
    assert "phymbert/xk6-sse" in run_text
    assert "grafana/xk6-sse" not in run_text


def test_xk6_and_k6_versions_are_pinned_not_floating():
    """D26: never a floating alias. Every version this file's own xk6 build step names must be an
    explicit tag, never `latest`."""
    wf = _workflow()
    env = wf["env"]
    for key in ("XK6_VERSION", "XK6_SSE_VERSION", "XK6_BUILD_K6_VERSION"):
        assert re.match(r"^v?\d+\.\d+\.\d+$", env[key]), f"{key}={env[key]!r} is not a pinned semver tag"
    build_step = next(s for s in _steps() if s.get("name", "").startswith("Build the xk6-sse k6 binary"))
    assert "latest" not in build_step["run"]


def test_xk6_sse_build_is_cached_keyed_on_all_three_pinned_versions():
    cache_step = next(s for s in _steps() if s.get("uses", "").startswith("actions/cache"))
    key = cache_step["with"]["key"]
    assert "${{ env.XK6_VERSION }}" in key
    assert "${{ env.XK6_SSE_VERSION }}" in key
    assert "${{ env.XK6_BUILD_K6_VERSION }}" in key
    build_step = next(s for s in _steps() if s.get("name", "").startswith("Build the xk6-sse k6 binary"))
    assert build_step["if"] == "steps.cache-xk6-sse.outputs.cache-hit != 'true'"


# ---- secret names match what the scripts they feed actually read -----------------------------------


def test_burst_up_secret_names_match_what_burst_up_sh_actually_checks():
    burst_up_step = next(s for s in _steps() if s.get("name", "").startswith("task burst:up"))
    env_names = set(burst_up_step["env"])
    burst_up_source = BURST_UP_PATH.read_text()
    for var in ("HCLOUD_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "ATLAS_R2_ENDPOINT", "ATLAS_BURST_DOMAIN", "ATLAS_BURST_ACME_EMAIL"):
        assert var in env_names, f"burst-benchmark.yml's task burst:up step never sets {var}"
        assert f"{var}:-" in burst_up_source, f"infra/scripts/burst-up.sh no longer checks {var}"


def test_burst_destroy_secret_names_match_what_burst_destroy_sh_actually_checks():
    destroy_step = next(s for s in _steps() if s.get("name", "").startswith("task burst:destroy"))
    env_names = set(destroy_step["env"])
    destroy_source = BURST_DESTROY_PATH.read_text()
    for var in ("HCLOUD_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "ATLAS_R2_ENDPOINT"):
        assert var in env_names, f"burst-benchmark.yml's task burst:destroy step never sets {var}"
        assert f"{var}:-" in destroy_source, f"infra/scripts/burst-destroy.sh no longer checks {var}"


def test_matrix_live_step_reads_the_provider_key_env_var_names_the_driver_actually_reads():
    driver_source = (ROOT / "testing" / "harness" / "matrix" / "live_driver.py").read_text()
    matrix_step = next(s for s in _steps() if s.get("name", "").startswith("task matrix:live"))
    env_names = set(matrix_step["env"])
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        assert var in env_names
        assert var in driver_source


def test_no_hardcoded_secret_values_only_secrets_context_references():
    text = WORKFLOW_PATH.read_text()
    assert "sk-" not in text  # no literal key prefix ever belongs in a committed workflow


# ---- honesty disclosures named plainly in the header (matching the other four SP10 lanes) ----------


def test_header_names_the_oidc_substitution_and_its_own_limitation():
    text = WORKFLOW_PATH.read_text()
    assert "OIDC" in text
    assert "no OIDC" in text or "NO OIDC" in text.upper()
    assert "required reviewers" in text.lower()
    # Limitation 2: the required reviewers rule is a Settings action, not committed code.
    assert "SETTINGS" in text.upper()


def test_header_names_dormant_until_pushed():
    text = WORKFLOW_PATH.read_text()
    assert "DORMANT UNTIL PUSHED" in text.upper()


def test_header_claims_the_sentinel_gate_and_names_reuse_discipline():
    text = WORKFLOW_PATH.read_text()
    assert "sentinel" in text.lower()
    assert "go or no go" in text.lower() or "go or no go" in text.lower()
    assert "never reimplement" in text.lower() or "never a second copy" in text.lower()


def test_header_names_the_three_pre_existing_gaps_this_task_does_not_fix():
    text = WORKFLOW_PATH.read_text()
    assert "indexes restore stop" in text.lower()
    assert "atlas-postgres-credentials" in text
    assert "phoenix" in text.lower() and "span" in text.lower()


def test_header_names_janitor_as_the_sole_orphan_backstop():
    text = WORKFLOW_PATH.read_text()
    assert "janitor.yml" in text
    assert "sole" in text.lower() or "no second orphan check" in text.lower()


def test_workflow_never_gates_by_name():
    """Same discipline every other SP10 lane test applies to its own lane: nothing about this
    workflow's own success/failure semantics may silently imply the job succeeding means the
    benchmark result itself was good or bad -- this is a benchmark run, reported via uploaded
    artifacts, never a merge gate (it does not even run on a PR or push event at all, see the
    trigger tests above)."""
    upload_step = next(s for s in _steps() if s.get("name", "").startswith("Upload burst benchmark"))
    assert upload_step["if"].startswith("always()")


# ---- the always() destroy fire drill: a pure simulator of GitHub Actions' own step semantics -------


def _default_condition_would_run(job_status: str) -> bool:
    """GitHub Actions' own documented default: a step with no `if:` runs only while the job's
    status is still 'success' (mirrors the implicit `success()` condition)."""
    return job_status == "success"


def _step_condition_would_run(step: dict, job_status: str) -> bool:
    raw_if = step.get("if")
    if raw_if is None:
        return _default_condition_would_run(job_status)
    condition = str(raw_if).strip()
    if condition == "always()":
        return True
    if condition == "failure()":
        return job_status == "failure"
    if condition.startswith("steps."):
        # A cache-hit style condition on a step that never itself fails the job; only relevant
        # once the job is already in a 'success' state (this simulator only cares about GATING
        # steps for the fire drill below, not this one's own cache-hit branching).
        return _default_condition_would_run(job_status)
    raise AssertionError(f"the fire drill's simulator does not model this condition yet: {raw_if!r}")


def _simulate_job(steps: list[dict], failing_step_index: int) -> list[bool]:
    """Replays GitHub Actions' own step execution semantics across `steps`, marking
    `failing_step_index` as the one step that fails once reached, and returns, for every step, in
    order, whether it actually ran. This is the SAME mechanism every other SP10 workflow test's
    `if: always()` assertion implicitly relies on, made explicit and executable here (matching SP9's
    own spend gate tamper rigor: prove the mechanism bites, never just trust the YAML string)."""
    job_status = "success"
    ran: list[bool] = []
    for index, step in enumerate(steps):
        would_run = _step_condition_would_run(step, job_status)
        ran.append(would_run)
        if would_run and index == failing_step_index:
            job_status = "failure"
    return ran


def _destroy_step_index(steps: list[dict]) -> int:
    return next(i for i, s in enumerate(steps) if s.get("name", "").startswith("task burst:destroy"))


def test_destroy_step_is_the_last_step_and_is_if_always():
    steps = _steps()
    destroy_index = _destroy_step_index(steps)
    assert destroy_index == len(steps) - 1, "task burst:destroy must be the FINAL step (every step before it can fail; this one still must run)"
    assert steps[destroy_index]["if"] == "always()"


def test_fire_drill_no_failure_at_all_destroy_still_runs():
    """The baseline: a clean run reaches destroy too (always() is not ONLY for the failure path)."""
    steps = _steps()
    ran = _simulate_job(steps, failing_step_index=-1)  # nothing fails
    assert ran[_destroy_step_index(steps)] is True


@pytest.mark.parametrize(
    "failing_step_name_prefix",
    [
        "task burst:up",  # the FIRST real step -- and, per this file's own header, the one that
        # genuinely refuses today (the indexes restore stop): the most important case to prove.
        "Sentinel go or no go gate",  # SP10's own claimed gate refusing on a red probe
        "task matrix:live",  # a paid, live step failing mid run
    ],
)
def test_fire_drill_destroy_still_runs_when_an_earlier_step_fails(failing_step_name_prefix):
    """THE fire drill (this file's own header, reasoning trace (5)): simulate EACH of several
    plausible earlier step failures (in the order they'd actually occur) and assert GitHub Actions'
    own real step condition semantics still reach `task burst:destroy`, never skip it."""
    steps = _steps()
    failing_index = next(i for i, s in enumerate(steps) if s.get("name", "").startswith(failing_step_name_prefix))
    ran = _simulate_job(steps, failing_step_index=failing_index)

    assert ran[failing_index] is True, "the step being simulated as 'the one that fails' must itself have actually run"
    assert ran[_destroy_step_index(steps)] is True, (
        f"task burst:destroy did not run in the simulation where {failing_step_name_prefix!r} failed -- "
        "the always() teardown discipline is broken"
    )
    # Every DEFAULT condition step strictly AFTER the failure point must have been SKIPPED (proves
    # the simulator is not vacuously true by running everything regardless of the failure).
    default_condition_steps_after_failure = [
        i for i, s in enumerate(steps)
        if i > failing_index and s.get("if") is None
    ]
    assert default_condition_steps_after_failure, "expected at least one default condition step after the failure point to make this a real test"
    assert all(ran[i] is False for i in default_condition_steps_after_failure), (
        "a default condition step after the simulated failure ran anyway -- the simulator itself is broken, "
        "not proving anything about the real always() discipline"
    )


def test_fire_drill_destroy_still_runs_no_matter_which_earlier_step_fails():
    """The EXHAUSTIVE fire drill, subsuming the named, illustrative scenarios above: simulate the
    failure at EVERY single step index before `task burst:destroy` in turn (not just the three named
    ones the plan calls out) and assert the always() destroy step still runs every time. Removes any
    need to separately argue "did the parametrized list above cover enough of the file" -- this test
    covers all of it, directly, by construction."""
    steps = _steps()
    destroy_index = _destroy_step_index(steps)
    assert destroy_index > 0, "expected at least one step before destroy to actually exercise the drill against"
    for failing_index in range(destroy_index):
        ran = _simulate_job(steps, failing_step_index=failing_index)
        assert ran[destroy_index] is True, (
            f"task burst:destroy did not run in the simulation where step {failing_index} "
            f"({steps[failing_index].get('name') or steps[failing_index].get('uses')!r}) failed"
        )
