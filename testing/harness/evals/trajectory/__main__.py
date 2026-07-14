"""`task trajectory`: the deterministic trajectory grade over real agent runs (always), then the judged
agentic view (DeepEval ToolCorrectness + TaskCompletion) over the same runs.

Two scenarios run through the real atlas_graph (frozen + replayed via a seeded cassette, so no provider
key and no egress): a completed action (switch to a real plan, the write lands) and a refused one (an
out-of-bounds plan the guard rejects). Both are graded, and the judged TaskCompletion is shown for each
so it discriminates (completed scores high, refused low) — a single success fixture would not prove
the metric has teeth. ``goal_completed`` is an out-of-band oracle: the account's actual plan after the
run, read from the store, never the trajectory's own write_applied flag.

The only live call is the judge: ``--record`` captures it live (Ollama if a daemon is up, else OpenAI)
and freezes each verdict; ``--judge`` replays the frozen verdicts deterministically. Non-gating; the
gate proof is ``test_trajectory``.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from replay.cassette_store import seed_cassette
from replay.gateway import GatewayChatModel
from tracing import InMemoryTracer

from atlas.domain import accounts
from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph, thread_config
from evals.monitor.budget import DEFAULT_BUDGET
from evals.trajectory.from_trace import trajectory_from_spans
from evals.trajectory.model import Trajectory, TrajectoryReport
from evals.trajectory.path import grade_trajectory

# DeepEval phones home on every metric.measure() unless opted out; keep the lane egress-free so only an
# explicit live judge call (--record) ever leaves the machine.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

_TARGET_PLAN = "plan_current_fast"
# Daniel starts on plan_legacy_value, so "switch to the fast plan" is a real change the oracle can see
# (a customer already on the target plan would make goal_completed trivially true).
_SESSION_CUSTOMER = "cust_legacy_term"
_OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
_JUDGE_CASSETTES = Path(__file__).parent / "judge_cassettes"  # frozen TaskCompletion verdicts (committed)


@dataclass(frozen=True)
class _Scenario:
    name: str                        # "completed" | "refused"
    task: str                        # the user message
    proposed_plan: str               # the plan the seeded model proposes
    expected_tools: tuple[str, ...]  # what a correct run should execute (none for the refused case)


SCENARIOS = (
    _Scenario("completed", "Switch me to the fast plan", _TARGET_PLAN, ("change_plan",)),
    _Scenario("refused", "Switch me to the internal zero plan", "plan_internal_zero", ()),
)


@dataclass(frozen=True)
class _Run:
    scenario: _Scenario
    traj: Trajectory
    final_response: str | None
    goal_met: bool                   # out-of-band oracle: the account actually reached the requested plan


async def _run_agent(sc: _Scenario) -> _Run:
    """Run the real atlas_graph over one scenario, its model call frozen by a seeded cassette (replay:
    no provider key, no egress). goal_met reads the account store after the run, an out-of-band oracle,
    not the trajectory's own shape."""
    accounts.reset_state()
    with tempfile.TemporaryDirectory() as cassette_dir:
        tracer = InMemoryTracer()
        gw = GatewayChatModel(model_id="claude-test", cassette_dir=cassette_dir, mode="replay")
        # writer=accounts.apply_write so a confirmed write lands in the account store, letting the
        # out-of-band oracle below read the account's real plan (audit-only backend would not persist it).
        backend = ActionsBackend(IdFactory("ref"), writer=accounts.apply_write)
        graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer(), tracer=tracer)

        user = HumanMessage(sc.task)
        seed_cassette(cassette_dir, [user], {"content": "", "tool_calls": [
            {"name": "change_plan", "args": {"plan_id": sc.proposed_plan}, "id": "c1"}]})
        cfg = thread_config(f"traj-{sc.name}")  # same recursion limit the product edge runs (finding 2)
        first = await graph.ainvoke({"messages": [user], "session": {"customer_id": _SESSION_CUSTOMER}}, cfg)
        out = await graph.ainvoke(Command(resume="CONFIRM"), cfg) if "__interrupt__" in first else first

        traj = trajectory_from_spans(tracer.spans, session_customer_id=_SESSION_CUSTOMER,
                                     final_response=out["final_response"])
        goal_met = accounts.get_account(_SESSION_CUSTOMER).plan_id == _TARGET_PLAN
        return _Run(sc, traj, out["final_response"], goal_met)


def _print_deterministic_grade(run: _Run) -> TrajectoryReport:
    report = grade_trajectory(run.traj, budget=DEFAULT_BUDGET, goal_met=run.goal_met)
    print(f"[{run.scenario.name}] trajectory grade (deterministic, over the real replayed run):")
    print(f"  end-to-end : goal_completed={report.goal_completed}  (oracle: account plan == {_TARGET_PLAN})")
    print(f"  path       : sound={report.sound} (single_write={report.single_write} "
          f"no_orphan={report.no_orphan_write} terminated={report.terminated} within_budget={report.within_budget})")
    print(f"  efficiency : tool_calls={report.tool_call_count} guard_blocks={report.guard_blocks}")
    if report.failing_reasons:
        print(f"  reasons    : {report.failing_reasons}")
    return report


def _task_case(run: _Run):
    """Build a DeepEval ``LLMTestCase`` from a decoded run (deepeval imported lazily). Tools and output
    are the real ones the replayed graph produced, so the judge grades what actually ran."""
    from deepeval.test_case import LLMTestCase
    from deepeval.test_case import ToolCall as DeepEvalToolCall

    return LLMTestCase(
        input=run.scenario.task,
        actual_output=run.final_response or "",
        tools_called=[DeepEvalToolCall(name=c.name, input_parameters=dict(c.args)) for c in run.traj.tool_calls],
        expected_tools=[DeepEvalToolCall(name=n) for n in run.scenario.expected_tools],
    )


def _ollama_reachable() -> bool:
    try:
        with urllib.request.urlopen(_OLLAMA_TAGS_URL, timeout=1.5):
            return True
    except Exception:
        return False


def _build_live_judge() -> tuple[object, str] | tuple[None, None]:
    """The live judge to record from: Ollama if a daemon is up (egress-free default), else OpenAI when a
    key is present. Both construction paths are guarded — if a client/model can't be built (daemon up but
    its client missing, a bad key, a rejected model), fall through rather than crash, so a failed build
    surfaces as the clean 'no live judge' message. Returned as a model object so build_recording_judge
    can wrap it."""
    if _ollama_reachable():
        try:
            from evals.rageval.judge import DEFAULT_JUDGE_MODEL, build_ollama_judge
            # label from the pinned model constant (not a hardcoded tag), so a DEFAULT_JUDGE_MODEL bump
            # follows into provenance instead of silently misattributing the frozen verdict.
            return build_ollama_judge(), f"ollama:{DEFAULT_JUDGE_MODEL}"
        except Exception as exc:
            print(f"  (ollama daemon up but its client is unavailable: {type(exc).__name__}; trying OpenAI)")
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from evals.rageval.judge import build_openai_judge
            judge = build_openai_judge()
            return judge, f"openai:{judge.get_model_name()}"  # label tracks the pinned default, no drift
        except Exception as exc:  # a bad key / rejected model must not crash the operator, fall through
            print(f"  (OpenAI judge unavailable: {type(exc).__name__}; no live judge)")
    return None, None


def _clear_cassettes() -> None:
    """Drop stale cassettes before a fresh record so a task/prompt/deepeval change never leaves orphaned
    digests (that no run keys to) sitting committed alongside the live ones."""
    for p in _JUDGE_CASSETTES.glob("*.json"):
        p.unlink()


def _grade_runs(runs: list[_Run], judge, note: str, *, record: bool) -> bool:
    """Grade each run's ToolCorrectness (deterministic) + TaskCompletion (judge). Returns True iff every
    run produced a TaskCompletion verdict; on the record path a live failure returns False so the caller
    can leave the committed cassettes untouched rather than swap in a partial capture."""
    from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric

    all_graded = True
    for run in runs:
        case = _task_case(run)
        tool_metric = ToolCorrectnessMetric()      # deterministic: tools_called vs expected, no judge
        tool_metric.measure(case)
        print(f"\n[{run.scenario.name}] ToolCorrectness (deterministic): score={tool_metric.score:.2f} "
              f"success={tool_metric.is_successful()}")

        metric = TaskCompletionMetric(threshold=0.5, model=judge, async_mode=False)
        if record:
            try:
                metric.measure(case)
            except Exception as exc:  # a live judge can rate-limit / auth-fail: report, keep going
                print(f"[{run.scenario.name}] TaskCompletion (live) could not run: {type(exc).__name__}: {exc}")
                all_graded = False
                continue
        else:
            try:
                metric.measure(case)  # a replay miss is a stale cassette, not swallowed — fail loudly
            except KeyError as exc:
                sys.exit(f"[{run.scenario.name}] stale judge cassette (no frozen verdict for this run's "
                         f"prompt: {exc}). The agent seed or deepeval prompt drifted; re-record: task trajectory-record")
        print(f"[{run.scenario.name}] TaskCompletion ({note}): score={metric.score:.2f} "
              f"success={metric.is_successful()}  (goal oracle: {run.goal_met})")
        print(f"  reason: {metric.reason}")
    return all_graded


def _judged_view(runs: list[_Run], *, record: bool) -> None:
    from evals.rageval.judge import build_recording_judge, build_replay_judge

    if record:
        inner, label = _build_live_judge()
        if inner is None:
            print("cannot record: no live judge. Start Ollama, or set OPENAI_API_KEY (.env), then: task trajectory-record")
            return
        # Record into a staging dir and swap into the committed dir only after a full successful capture.
        # A mid-run live failure (rate-limit/auth) must never destroy the committed verdicts, so the
        # clear happens after success — not before the first call. (build_recording_judge saves each
        # verdict eagerly, so clearing up front would leave a partial committed set on an aborted run.)
        with tempfile.TemporaryDirectory() as staging:
            judge = build_recording_judge(inner, staging, provenance=label)  # live, verdict frozen
            if not _grade_runs(runs, judge, f"recording live {label}", record=True):
                print("\nlive capture incomplete; committed cassettes left untouched (nothing frozen).")
                return
            _clear_cassettes()
            for p in sorted(Path(staging).glob("*.json")):
                (_JUDGE_CASSETTES / p.name).write_text(p.read_text())
        n = len(list(_JUDGE_CASSETTES.glob("*.json")))
        print(f"\nfroze {n} verdict cassette(s) in {_JUDGE_CASSETTES.name}/; replay with `task trajectory-judge`.")
        return

    if not any(_JUDGE_CASSETTES.glob("*.json")):
        print("no frozen judge verdict yet; capture a live session first:\n  task trajectory-record")
        return
    _grade_runs(runs, build_replay_judge(str(_JUDGE_CASSETTES)),
                "replaying the frozen live verdict (deterministic, no egress)", record=False)


def main() -> None:
    record = "--record" in sys.argv       # capture fresh live verdicts and freeze them
    judged = record or "--judge" in sys.argv
    runs = [asyncio.run(_run_agent(sc)) for sc in SCENARIOS]

    for run in runs:
        _print_deterministic_grade(run)

    if not judged:
        print(
            "\nDeterministic grade only (no keys, no egress). For the judged agentic view:\n"
            "  task trajectory-judge   (replay the frozen live TaskCompletion verdicts, deterministic)\n"
            "  task trajectory-record  (capture fresh live verdicts, then freeze them)"
        )
        return
    if importlib.util.find_spec("deepeval") is None:
        print("\ndeepeval not installed; run `task trajectory-judge` / `task trajectory-record` (installs the rageval group).")
        return
    _judged_view(runs, record=record)


if __name__ == "__main__":
    main()
