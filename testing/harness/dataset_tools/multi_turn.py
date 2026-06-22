"""The multi turn case runner (SP7 Task 5, D24): a dataset contract case's `turns` array driven
through the REAL atlas graph (`atlas.orchestration.atlas_graph.build_atlas_graph`), one turn at a
time on ONE thread, over a fresh `determinism.checkpointer.new_checkpointer()` per case (an
`InMemorySaver`, hermetic, never shared across cases or callers). Two grading tiers, per D24:

  - Per turn `checkpoint` blocks (`expected_intent`, `expected_tool_calls`) are DIAGNOSTIC: they
    describe what should have happened mid trajectory, useful for debugging a failing case, but
    never gate `MultiTurnResult.passed`. `expected_tool_calls` uses the dataset contract's own
    server dot tool naming convention (`gc-0002`'s `catalog.get_plan`); the runtime graph's own
    `AIMessage.tool_calls` carry the bare tool name the model actually invoked (least agency binds
    per intent, not per MCP server, `atlas.domain.binding`), so a recognized namespace prefix is
    stripped before comparing. This normalization is best effort and diagnostic only, never gating.
  - `end_state.account_assertions` is GATING: the real environment outcome after every turn ran
    (`atlas.domain.accounts`, the one mutable, write through account store this repo's CLAUDE.md
    names), dereferenced by dot path and compared with the SAME `str()` typed coercion
    `quality.agent_metrics`/`corpus_tools.verify` already use elsewhere, so an int, a `Decimal`, or
    a plain string account field compares the same way a JSON literal from the case does. A case
    with no `end_state` at all (or one with no `account_assertions`) passes vacuously: nothing
    declared, nothing to fail.

A write turn pauses at the confirmation interrupt (`atlas_graph.pre_action_guard`'s own
`interrupt()`); this runner resumes it once, immediately, with a fixed `confirm_resume` value,
mirroring `evals.simulation.driver.drive_conversation`'s own established pattern for a scripted
turn. The dataset contract's `turns` array has no separate "confirm" turn of its own (`gc-0002`'s
second turn, "Yes, switch me to it.", IS the write request); auto confirming every interrupt is
this runner's own documented policy for a scripted, no human in the loop case, not a new schema
concept.

Cross provider simulation, a dynamic, LLM played user turn rather than a literal scripted one,
already exists as its own lane (`evals.simulation`, ADR 019, the persona simulation lane the SP7
digest's D24 paragraph names): that lane informs (many trials, a pass rate with a confidence
interval, failures frozen into fixtures) and never gates. This module is the named seam a frozen
simulation fixture would run through if it were ever promoted into the dataset contract's `turns`
shape; it does not build a live simulator of its own, and does not attempt to (SP7's own scope
line: SP8 owns the calibrated judge, not a second simulation driver).

No reference free faithfulness, judge, or rubric anything lives here: SP8's boundary, the 04/05
grader boundary this repo's CLAUDE.md names.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory

from atlas.domain import accounts
from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import HANDOFF_PREFIX, build_atlas_graph, thread_config

from quality.agent_metrics import ToolCallMetrics, tool_call_metrics

# The MCP server namespaces the dataset contract's `expected_tool_calls` prefixes a tool name with
# (`catalog.get_plan`, `actions.change_plan`, `account.get_contract`, mirrored off
# `atlas.mcp_servers.*_server.py`'s own module names). Stripped for the diagnostic comparison
# only; see `_bare_tool_name`.
_KNOWN_TOOL_NAMESPACES = ("account", "catalog", "actions", "knowledge")
_DEFAULT_CONFIRM = "CONFIRM"


def _bare_tool_name(tool: str) -> str:
    """Diagnostic only normalization: the dataset contract's `expected_tool_calls` names an MCP
    server namespace, while the runtime graph's `AIMessage.tool_calls` carries the bare tool name
    the model actually invoked (binding is per intent, not per server,
    `atlas.domain.binding.INTENT_TOOLS` only ever names bare tool strings). Strips a recognized
    namespace prefix so the two sides compare on the same name; an unrecognized prefix, or no dot
    at all, passes through unchanged rather than guessed at, since this check never gates."""
    server, dot, bare = tool.partition(".")
    return bare if dot and server in _KNOWN_TOOL_NAMESPACES else tool


@dataclass(frozen=True)
class TurnResult:
    """One turn's diagnostic readout. `intent_match`/`tool_call_metrics` are `None` when the case
    carries no `checkpoint` block (or an empty one) for this turn: nothing declared, nothing to
    diagnose, never a false pass or fail."""

    turn_index: int
    user: str
    observed_intent: Optional[str]
    expected_intent: Optional[str]
    intent_match: Optional[bool]
    observed_tool_calls: tuple[dict, ...]
    expected_tool_calls: tuple[dict, ...]
    tool_call_metrics: Optional[ToolCallMetrics]
    final_response: str
    refused: bool


@dataclass(frozen=True)
class AccountAssertionResult:
    path: str
    expected: object
    actual: object
    found: bool
    passed: bool


@dataclass(frozen=True)
class MultiTurnResult:
    """`passed` is GATING and reflects `account_assertions` alone (vacuously `True` when the case
    declares no `end_state`/`account_assertions`). `turns` is diagnostic context, always
    populated, never a factor in `passed`."""

    case_id: str
    thread_id: str
    turns: tuple[TurnResult, ...]
    account_assertions: tuple[AccountAssertionResult, ...]
    passed: bool


def _resolve_path(obj: object, path: str) -> tuple[bool, object]:
    """Dot path traversal over a frozen dataclass's attributes or a mapping's keys, the two shapes
    `atlas.domain.accounts.Account` and its own nested value types (`Usage`, `Bill`, ...) take. An
    unresolvable segment returns `(False, None)` rather than raising, so one bad path fails its own
    assertion instead of aborting every other assertion in the case."""
    current = obj
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return False, None
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return False, None
    return True, current


def _check_account_assertions(
    end_state: Optional[Mapping[str, object]], customer_id: str
) -> tuple[AccountAssertionResult, ...]:
    assertions = (end_state or {}).get("account_assertions") or ()
    if not assertions:
        return ()
    account = accounts.get_account(customer_id)
    results = []
    for assertion in assertions:
        path, expected = assertion["path"], assertion["equals"]
        found, actual = _resolve_path(account, path)
        # the SAME typed coercion `quality.agent_metrics`/`corpus_tools.verify` apply elsewhere: a
        # Decimal, int, or bool account field compares the same way a JSON literal does.
        passed = found and str(actual) == str(expected)
        results.append(
            AccountAssertionResult(path=path, expected=expected, actual=actual, found=found, passed=passed)
        )
    return tuple(results)


def _observed_tool_calls(messages: Sequence[BaseMessage], start: int) -> tuple[dict, ...]:
    """Every tool call an `AIMessage` proposed strictly AFTER index `start` (the message count
    before this turn began), name and args, in message order. Reading straight off the message
    history rather than `AtlasState.tools_called` (which the runtime keeps as names only, no args)
    gives the diagnostic comparison the SAME `{tool, args}` shape `tool_call_metrics` expects."""
    calls = []
    for message in list(messages)[start:]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or ():
                calls.append({"tool": call["name"], "args": dict(call.get("args") or {})})
    return tuple(calls)


async def run_multi_turn_case(
    case: Mapping[str, object],
    model: BaseChatModel,
    *,
    customer_id: str,
    retriever=None,
    tracer=None,
    cache=None,
    ids: Optional[IdFactory] = None,
    backend: Optional[ActionsBackend] = None,
    thread_id: Optional[str] = None,
    confirm_resume: str = _DEFAULT_CONFIRM,
) -> MultiTurnResult:
    """Run one dataset contract `turns` array case end to end against the real graph.

    `ids` defaults to a fresh, deterministic `IdFactory` (hermetic, no wall clock, no randomness),
    the graph's own idempotency key source. `backend` defaults to an `ActionsBackend` wired with
    `accounts.apply_write` as its writer, the write through that makes `end_state.
    account_assertions` meaningful against real environment state (without a writer a confirmed
    action reaches only the audit log, never the account store `_resolve_path` reads). `thread_id`
    defaults to the case's own `case_id`; either way this call always builds its OWN fresh
    checkpointer (`new_checkpointer()`), never shared across cases or callers, so two calls never
    interfere even when given the same `thread_id`.
    """
    ids = ids or IdFactory("mt-idem")
    if backend is None:
        backend = ActionsBackend(IdFactory("mt-ref"), writer=accounts.apply_write)
    checkpointer = new_checkpointer()
    graph = build_atlas_graph(
        model, ids, backend, checkpointer, retriever=retriever, tracer=tracer, cache=cache
    )
    tid = thread_id or str(case["case_id"])
    config = thread_config(tid)
    session = {"customer_id": customer_id}

    turn_results: list[TurnResult] = []
    prior_len = 0
    for index, turn in enumerate(case["turns"]):
        human = HumanMessage(str(turn["user"]))
        out = await graph.ainvoke({"messages": [human], "session": session}, config)
        while "__interrupt__" in out:  # a write turn pauses at the confirmation gate; see docstring
            out = await graph.ainvoke(Command(resume=confirm_resume), config)

        messages = list(out.get("messages") or ())
        observed_calls = _observed_tool_calls(messages, prior_len)
        prior_len = len(messages)

        checkpoint = turn.get("checkpoint") or {}
        expected_intent = checkpoint.get("expected_intent")
        observed_intent = out.get("intent")
        intent_match = None if expected_intent is None else expected_intent == observed_intent

        raw_expected_calls = tuple(checkpoint.get("expected_tool_calls") or ())
        metrics = None
        if raw_expected_calls:
            normalized = tuple(
                {"tool": _bare_tool_name(str(c["tool"])), "args": c.get("args") or {}}
                for c in raw_expected_calls
            )
            metrics = tool_call_metrics(normalized, observed_calls)

        final_response = str(out.get("final_response") or "")
        turn_results.append(
            TurnResult(
                turn_index=index,
                user=str(turn["user"]),
                observed_intent=observed_intent,
                expected_intent=expected_intent,
                intent_match=intent_match,
                observed_tool_calls=observed_calls,
                expected_tool_calls=raw_expected_calls,
                tool_call_metrics=metrics,
                final_response=final_response,
                refused=final_response.startswith(HANDOFF_PREFIX),
            )
        )

    assertions = _check_account_assertions(case.get("end_state"), customer_id)
    passed = all(assertion.passed for assertion in assertions)
    return MultiTurnResult(
        case_id=str(case["case_id"]),
        thread_id=tid,
        turns=tuple(turn_results),
        account_assertions=assertions,
        passed=passed,
    )


__all__ = [
    "AccountAssertionResult",
    "MultiTurnResult",
    "TurnResult",
    "run_multi_turn_case",
]
