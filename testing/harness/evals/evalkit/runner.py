"""The runner that repeats trials: drive a case, grade each run, report a RATE not a verdict.

The live model is stochastic, so a single run is a single sample and a single sample lies. Seven
out of ten is not ten out of ten, and one pass cannot tell those two agents apart.
So the runner repeats each case ``k`` times and reports the pass rate. On REPLAY the model is
pinned, so all ``k`` trials are identical and the rate is 0 or 1, which is exactly what lets the PR
lane prove the runner's WIRING deterministically. The variance only appears on LIVE, the nightly
eval's job. (Turning that rate into a confidence interval is the statistics article's work, 07.)

Only the model is allowed to vary between trials: each trial gets a fresh graph, tracer, write
backend, and checkpointer from the ``build`` callback, and the clock and id factories stay pinned,
so a difference between trials is a model difference and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from evals.evalkit.case import EvalCase
from evals.evalkit.graders import Composite, GradeContext, Grader, Verdict

# build() -> (compiled graph, the tracer wired into it), fresh per trial. The caller owns wiring
# (gateway mode, cassette dir, fakes), so the runner stays mode agnostic: same runner on REPLAY/LIVE.
GraphBuild = Callable[[], "tuple[object, object]"]


@dataclass(frozen=True)
class TrialResult:
    index: int
    passed: bool
    verdicts: tuple[Verdict, ...]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passes: int
    k: int
    trials: tuple[TrialResult, ...]
    name: str = ""
    risk: str = ""

    @property
    def rate(self) -> float:
        return self.passes / self.k if self.k else 0.0

    def first_failure_reason(self) -> str:
        """The first failing grader's reason across the trials, the line an SDET reads on a red."""
        for trial in self.trials:
            for verdict in trial.verdicts:
                if not verdict.passed:
                    return verdict.reason
        return ""


def _trial_passed(verdicts: Sequence[Verdict]) -> bool:
    """A trial passes when at least one grader ran and none failed."""
    return bool(verdicts) and all(v.passed for v in verdicts)


async def _drive(graph, case: EvalCase, thread_id: str) -> str:
    """Run the case's turns in order on one thread, and return the text the agent would ship.

    Every terminal path of the graph (render / confirm / binding-block) sets ``final_response``,
    so that channel is the single, authoritative "what shipped" the eval grades.

    The turns are replayed VERBATIM, the correct hermetic default. A ``UserSimulator`` (a scripted
    user, or an LLM user with information asymmetry driven through the gateway, the tau-bench pattern)
    plugs in exactly here and is deferred to the golden dataset article (04). See ``case.py``.
    """
    final = ""
    config = {"configurable": {"thread_id": thread_id}}
    # All turns run on ONE thread_id: under the checkpointer that is what makes a case of many turns a
    # real conversation. Turn 2 resumes turn 1's state (messages, a pending confirm), instead of
    # starting cold. Per trial isolation comes from the fresh graph/checkpointer `build()` hands us,
    # not from changing the thread between turns.
    #
    # A paused confirmation is a LangGraph `interrupt()`, not an ordinary node boundary: the graph
    # only actually resumes that node when the NEXT call is `Command(resume=...)` (the exact contract
    # `confirm()` in atlas_graph.py, `chat_app.py`'s `/chat/resume`, and `test_confirm_graph.py` all
    # share). A fresh `{"messages": [...]}` input instead restarts the graph from START, so the pending
    # confirm is silently never delivered.
    awaiting_confirm = False
    for utterance in case.turns:
        if awaiting_confirm:
            out = await graph.ainvoke(Command(resume=utterance), config)
        else:
            out = await graph.ainvoke(
                {"messages": [HumanMessage(utterance)], "session": {"customer_id": case.customer_id}},
                config,
            )
        awaiting_confirm = "__interrupt__" in out
        final = out.get("final_response") or ""
    return final


async def run_case(case: EvalCase, build: GraphBuild, graders: Sequence[Grader], k: int = 1) -> CaseResult:
    """Drive ``case`` ``k`` times through freshly built graphs, grade each, aggregate to a rate."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    composite = Composite(graders)
    trials: list[TrialResult] = []
    for i in range(k):
        graph, tracer = build()
        final = await _drive(graph, case, thread_id=f"{case.id}-trial{i}")
        ctx = GradeContext(customer_id=case.customer_id, final_response=final, trace=tuple(tracer.spans))
        verdicts = composite.grade(ctx)
        trials.append(TrialResult(index=i, passed=_trial_passed(verdicts), verdicts=verdicts))
    passes = sum(1 for t in trials if t.passed)
    return CaseResult(
        case_id=case.id, passes=passes, k=k, trials=tuple(trials),
        name=case.name, risk=case.risk,
    )


__all__ = ["CaseResult", "GraphBuild", "TrialResult", "run_case"]
