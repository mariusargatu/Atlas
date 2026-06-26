"""Project a dataset contract case (`contracts/dataset/schema.json`, the registry generated and
hand authored JSONL rows) onto `EvalCase`, the runner's input.

The mirror of `GoldenCase.to_eval_case()`. Both authoring surfaces keep their own record type and
schema; they meet here, at the run spec. Provenance is deliberately NOT projected: it stays on the
dataset record and is normalised separately (`evals.evalkit.provenance`), per the rule
`to_eval_case` states.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from evals.evalkit.case import EvalCase
from evals.evalkit.risk import risk_of

#: The session an identity-independent case runs under. A factoid case ("what is region-north's
#: coverage_type?") is answered identically by every session, but the graph still needs one. Sarah is
#: the least trap laden identity: current plan, term free, uncapped, so no grader can read an
#: identity-dependent fact as a failure. One constant, one place to change.
NEUTRAL_SESSION = "cust_current"


def _fact_prose(expected_facts: Sequence[Mapping[str, object]]) -> str:
    """Render `expected_facts` as the prose `EvalCase.expected` carries for a human reader.

    For a synthetic case the registry IS the oracle, so this is generated from the ground truth,
    never authored. Sorted by fact_id so the rendering is deterministic.
    """
    parts = [
        f"{fact['fact_id']} = {fact['value']}" if "value" in fact else str(fact["fact_id"])
        for fact in sorted(expected_facts, key=lambda f: str(f["fact_id"]))
    ]
    return "; ".join(parts)


def _all_expected_tool_calls(case: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    """Every expected tool call the case declares, top level AND nested in each turn's checkpoint.

    A single-turn case holds its expectation at the top level; a multi-turn case holds it per turn,
    under `turns[].checkpoint.expected_tool_calls`, since each checkpoint is what that turn must have
    triggered. Reading only the top level (the prior behaviour) let a multi-turn case whose
    expectations live entirely in checkpoints project to an empty grader tuple: it ran and reported
    green while checking nothing. This is the small local equivalent of the recursive walk
    `test_seed_dataset.py`'s `_every_tool_call` performs; not imported from there since that is test
    code and this is harness code.
    """
    calls = list(case.get("expected_tool_calls") or ())
    for turn in case.get("turns") or ():
        calls.extend((turn.get("checkpoint") or {}).get("expected_tool_calls") or ())
    return tuple(calls)


def graders_for(case: Mapping[str, object]) -> tuple[str, ...]:
    """The grader names this case's SHAPE calls for, in a stable order.

    Derived, never declared per case: the generated cases would otherwise need a hand kept grader
    list, the exact "derive, do not hand keep" rule this repo holds everywhere else. Every name
    returned here must exist in `metric_graders.GOLDEN_GRADERS` or the suite build fails loudly.

    `end_state.account_assertions` and `expected_facts` are independent SHAPE signals, not mutually
    exclusive alternatives: a case can declare a write assertion AND a read fact in the same
    checkpoint (the schema permits both, even though none of today's cases combine them), and each
    needs its own grader. An `elif` here would silently drop `answer-true-vs-account` the day a case
    finally does combine them, exactly the "runs green while checking less than it looks like" defect
    this function exists to avoid, so both conditions are checked independently.
    """
    names: list[str] = []
    if (case.get("end_state") or {}).get("account_assertions"):
        names.append("write-applied-after-confirm")
    if case.get("expected_facts"):
        names.append("answer-true-vs-account")
    if case.get("expected_doc_ids"):
        names.append("retrieval-ids-recalled")
    if _all_expected_tool_calls(case):
        names.append("tool-calls-match")
    return tuple(names)


def dataset_case_to_eval_case(case: Mapping[str, object]) -> EvalCase:
    """One dataset case as the runner's input. See the module docstring for what is not projected."""
    return EvalCase(
        id=str(case["case_id"]),
        turns=tuple(str(turn["user"]) for turn in case["turns"]),
        customer_id=str(case.get("customer_id") or NEUTRAL_SESSION),
        expected=_fact_prose(case.get("expected_facts") or ()),
        name=str(case["case_id"]),
        risk=risk_of(case),
        graders=graders_for(case),
        expected_doc_ids=tuple(str(i) for i in (case.get("expected_doc_ids") or ())),
        expected_tool_calls=_all_expected_tool_calls(case),
    )


__all__ = ["NEUTRAL_SESSION", "dataset_case_to_eval_case", "graders_for"]
