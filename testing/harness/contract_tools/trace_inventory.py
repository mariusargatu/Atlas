"""D11 style golden inventory of the graph's own trace vocabulary (SP6 task 2).

Runs a fixed set of multi turn scenarios through `build_atlas_graph` under `InMemoryTracer` and
dumps every OBSERVED `(name, kind, attrs)` tuple `tracer.open(...)` produced to the committed
`contracts/trace/span_inventory.json`. `backend/atlas/adapters/trace_translation.py` reads that
file as its fail closed allowlist: a new, unreviewed span vocabulary shape must regenerate this
file (and get reviewed in the SAME diff) before the OTel adapter will export it -- the same
"regenerate and assert equality" mechanism `contract_tools.mcp_snapshot` already established for
MCP tool schemas (see that module's own header comment).

Regeneration:

    uv run python -m contract_tools.trace_inventory --write

then review the resulting `git diff contracts/trace/span_inventory.json` in the SAME change:
`testing/tests/test_span_inventory.py` fails loud on any drift the committed file does not already
reflect. `--check` (the default) prints a diff without writing.

`kind="tool"` spans are recorded under a WILDCARD name (`TOOL_WILDCARD`), never the real dynamic
tool name (`get_account_summary`, `search_knowledge`, `change_plan`, ...): see
`trace_translation.py`'s own module docstring, point 1, for why.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import tempfile

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.types import Command

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from replay.cassette_store import seed_cassette
from replay.gateway import GatewayChatModel
from tracing import InMemoryTracer

from atlas.adapters.inmemory_retriever import InMemoryRetriever
from atlas.adapters.resilience import ProviderError
from atlas.domain.actions import ActionsBackend
from atlas.domain.retrieval import RetrievalConfig
from atlas.orchestration.atlas_graph import build_atlas_graph
from determinism.canonical import serialize_tool_result

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
INVENTORY_PATH = REPO_ROOT / "contracts" / "trace" / "span_inventory.json"

# Mirrors `trace_translation.TOOL_WILDCARD` (duplicated, not imported: harness may import backend,
# never the reverse, and this constant's VALUE is the shared contract between the two files, cross
# checked by `testing/tests/test_trace_translation.py`, not by a Python import).
TOOL_WILDCARD = "*"

_SESSION = {"customer_id": "cust_current"}


def _new_graph(cassette_dir, *, model=None, fallback_model=None, tracer=None, retriever=None):
    gw = model or GatewayChatModel(model_id="claude-test", cassette_dir=cassette_dir, mode="replay")
    backend = ActionsBackend(IdFactory("ref"))
    return build_atlas_graph(
        gw, IdFactory("idem"), backend, new_checkpointer(),
        retriever=retriever, tracer=tracer, fallback_model=fallback_model,
    )


class _RaisingModel(BaseChatModel):
    """A minimal test double whose `_agenerate` raises a given exception directly -- no cassette,
    no resilience machinery, just `atlas_graph.agent()`'s own `except ProviderError` routing."""

    exc: Exception

    @property
    def _llm_type(self) -> str:
        return "raising-model"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("the graph is async only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        raise self.exc


class _PlainAnswerModel(BaseChatModel):
    answer: str = "plain answer"

    @property
    def _llm_type(self) -> str:
        return "plain-answer"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("the graph is async only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.answer))])


class _AlwaysRefusingRetriever:
    """Every retrieval attempt exhausts the ladder: `_knowledge_call` returns `("", "refusal")`,
    triggering `tools_read`'s `refusal_trigger` guard span and, downstream, the `refusal` node."""

    def search_chunks(self, query: str, k: int, config: RetrievalConfig):
        from atlas.adapters.resilience import RetrievalError

        raise RetrievalError("every seam down", provider_key="postgres")


async def _scenario_answer_and_cache(tracer: InMemoryTracer) -> None:
    """Knowledge only Q&A, twice: turn/agent llm/tool(search_knowledge)/pre_render_guard(true)/
    render/embed/retrieve/rerank/assemble on the first call, plus the cache node on the second (a
    fresh thread, same generic cacheable question)."""
    with tempfile.TemporaryDirectory() as tmp:
        query = "what is a data cap"
        user = HumanMessage("What is a data cap?")
        toolcall = [{"name": "search_knowledge", "args": {"query": query}, "id": "k1"}]
        seed_cassette(tmp, [user], {"content": "", "tool_calls": toolcall})
        chunks = InMemoryRetriever().search_chunks(query, config=RetrievalConfig())
        passages = serialize_tool_result(
            [{"doc_id": c.doc_id, "chunk_id": c.chunk_id, "score": c.score, "text": c.text} for c in chunks]
        )
        ai = AIMessage(content="", tool_calls=toolcall)
        tool_msg = ToolMessage(content=passages, tool_call_id="k1", name="search_knowledge")
        seed_cassette(tmp, [user, ai, tool_msg], {"content": "A data cap is a monthly limit.", "tool_calls": []})

        graph = _new_graph(tmp, tracer=tracer)
        await graph.ainvoke({"messages": [user], "session": _SESSION}, {"configurable": {"thread_id": "inv-cache-1"}})
        await graph.ainvoke({"messages": [user], "session": _SESSION}, {"configurable": {"thread_id": "inv-cache-2"}})


async def _scenario_bind_guard_block(tracer: InMemoryTracer) -> None:
    """A `policy_question` turn (binds knowledge+catalog only) whose model calls a READ tool
    (`get_account_summary`, unreachable on this intent) -> `tools_read`'s `bind_guard`."""
    with tempfile.TemporaryDirectory() as tmp:
        user = HumanMessage("Tell me about my account")
        seed_cassette(
            tmp, [user],
            {"content": "", "tool_calls": [{"name": "get_account_summary", "args": {}, "id": "b1"}]},
        )
        graph = _new_graph(tmp, tracer=tracer)
        session = {**_SESSION, "intent": "policy_question"}
        await graph.ainvoke({"messages": [user], "session": session}, {"configurable": {"thread_id": "inv-bind"}})


async def _scenario_budget_guard_read_block(tracer: InMemoryTracer) -> None:
    """Four `search_knowledge` calls in one batch exceeds `DEFAULT_BUDGET.max_retrieval_rounds`
    (3) -> `tools_read`'s `check_budget` fires its `budget_guard` span."""
    with tempfile.TemporaryDirectory() as tmp:
        user = HumanMessage("Search four times please")
        toolcalls = [
            {"name": "search_knowledge", "args": {"query": f"q{i}"}, "id": f"s{i}"} for i in range(4)
        ]
        seed_cassette(tmp, [user], {"content": "", "tool_calls": toolcalls})
        graph = _new_graph(tmp, tracer=tracer)
        await graph.ainvoke({"messages": [user], "session": _SESSION}, {"configurable": {"thread_id": "inv-budget"}})


async def _scenario_refusal_trigger_and_refusal_node(tracer: InMemoryTracer) -> None:
    """Retrieval exhausts the ladder (`_AlwaysRefusingRetriever`) -> `refusal_trigger` guard span,
    then the terminal `refusal` node span."""
    with tempfile.TemporaryDirectory() as tmp:
        query = "is my plan contract free"
        user = HumanMessage("Is my plan contract-free?")
        toolcall = [{"name": "search_knowledge", "args": {"query": query}, "id": "r1"}]
        seed_cassette(tmp, [user], {"content": "", "tool_calls": toolcall})
        graph = _new_graph(tmp, tracer=tracer, retriever=_AlwaysRefusingRetriever())
        await graph.ainvoke({"messages": [user], "session": _SESSION}, {"configurable": {"thread_id": "inv-refusal"}})


async def _scenario_pre_action_guard_unreachable(tracer: InMemoryTracer) -> None:
    """A `troubleshooting` turn (binds knowledge+reads, no writes) whose model calls a WRITE tool
    anyway -> `route_after_agent` still routes to "act" (any WRITE_TOOLS call does), and
    `pre_action_guard`'s own unreachable check fires."""
    with tempfile.TemporaryDirectory() as tmp:
        user = HumanMessage("Just checking in")
        seed_cassette(
            tmp, [user],
            {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "w1"}]},
        )
        graph = _new_graph(tmp, tracer=tracer)
        session = {**_SESSION, "intent": "troubleshooting"}
        await graph.ainvoke({"messages": [user], "session": session}, {"configurable": {"thread_id": "inv-unreach"}})


async def _scenario_pre_action_guard_single_write(tracer: InMemoryTracer) -> None:
    """An `action` turn whose model proposes TWO write tool calls in one batch ->
    `guardrules.check_single_write` fails -> `pre_action_guard`'s `{ok, reason}` shape (no `tool`
    key, distinct from the unreachable/scope/bounds call sites' `{ok, reason, tool}` shape)."""
    with tempfile.TemporaryDirectory() as tmp:
        user = HumanMessage("Switch my plan and add an addon")
        toolcalls = [
            {"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "m1"},
            {"name": "add_addon", "args": {"addon_id": "addon_1"}, "id": "m2"},
        ]
        seed_cassette(tmp, [user], {"content": "", "tool_calls": toolcalls})
        graph = _new_graph(tmp, tracer=tracer)
        session = {**_SESSION, "intent": "action"}
        await graph.ainvoke({"messages": [user], "session": session}, {"configurable": {"thread_id": "inv-singlewrite"}})


async def _scenario_successful_write(tracer: InMemoryTracer) -> None:
    """A clean write, confirmed: `pre_action_guard`'s bounds check span (`{ok, reason, tool}`, true
    this time), the write's own `tool` span (`args`+`proposal`), and `execute_action`'s success
    shape (`applied`+`reference`)."""
    with tempfile.TemporaryDirectory() as tmp:
        user = HumanMessage("Switch me to the fast plan")
        seed_cassette(
            tmp, [user],
            {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c1"}]},
        )
        graph = _new_graph(tmp, tracer=tracer)
        cfg = {"configurable": {"thread_id": "inv-write-ok"}}
        await graph.ainvoke({"messages": [user], "session": _SESSION}, cfg)
        await graph.ainvoke(Command(resume="CONFIRM"), cfg)


async def _scenario_confirmation_error(tracer: InMemoryTracer) -> None:
    """A malformed (non "CONFIRM") resume -> `ConfirmationError` -> `execute_action`'s failure
    shape (`applied`+`reason`, distinct from the success shape's `applied`+`reference`)."""
    with tempfile.TemporaryDirectory() as tmp:
        user = HumanMessage("Switch me to the fast plan")
        seed_cassette(
            tmp, [user],
            {"content": "", "tool_calls": [{"name": "change_plan", "args": {"plan_id": "plan_current_fast"}, "id": "c2"}]},
        )
        graph = _new_graph(tmp, tracer=tracer)
        cfg = {"configurable": {"thread_id": "inv-write-err"}}
        await graph.ainvoke({"messages": [user], "session": _SESSION}, cfg)
        await graph.ainvoke(Command(resume="not-confirm"), cfg)


async def _scenario_agent_failure_no_fallback(tracer: InMemoryTracer) -> None:
    """A non retryable primary failure, no fallback configured -> `agent()`'s own
    `agent_failure` span carrying `{ok, reason, retryable}`."""
    with tempfile.TemporaryDirectory() as tmp:
        model = _RaisingModel(exc=ProviderError("bad request", retryable=False, provider_key="primary-model"))
        graph = _new_graph(tmp, model=model, tracer=tracer)
        user = HumanMessage("What are your hours?")
        await graph.ainvoke({"messages": [user], "session": _SESSION}, {"configurable": {"thread_id": "inv-nofallback"}})


async def _scenario_agent_failure_fallback_also_fails(tracer: InMemoryTracer) -> None:
    """A retryable primary failure WITH a fallback configured, and the fallback ALSO fails ->
    `agent_failure`'s OTHER shape, `{ok, reason}` (no `retryable` key)."""
    with tempfile.TemporaryDirectory() as tmp:
        primary = _RaisingModel(exc=ProviderError("primary down", retryable=True, provider_key="primary-model"))
        fallback = _RaisingModel(exc=RuntimeError("fallback also down"))
        graph = _new_graph(tmp, model=primary, fallback_model=fallback, tracer=tracer)
        user = HumanMessage("What are your hours?")
        await graph.ainvoke({"messages": [user], "session": _SESSION}, {"configurable": {"thread_id": "inv-bothfail"}})


async def _scenario_agent_fallback_succeeds(tracer: InMemoryTracer) -> None:
    """A retryable primary failure, fallback succeeds -> the `agent`/`llm` span's OTHER shape,
    `{model, degradation_mode}` (the plain happy path only ever carries `{model}`)."""
    with tempfile.TemporaryDirectory() as tmp:
        primary = _RaisingModel(exc=ProviderError("primary down", retryable=True, provider_key="primary-model"))
        fallback = _PlainAnswerModel(answer="fallback answered")
        graph = _new_graph(tmp, model=primary, fallback_model=fallback, tracer=tracer)
        user = HumanMessage("What are your hours?")
        await graph.ainvoke({"messages": [user], "session": _SESSION}, {"configurable": {"thread_id": "inv-fallback-ok"}})


SCENARIOS = (
    _scenario_answer_and_cache,
    _scenario_bind_guard_block,
    _scenario_budget_guard_read_block,
    _scenario_refusal_trigger_and_refusal_node,
    _scenario_pre_action_guard_unreachable,
    _scenario_pre_action_guard_single_write,
    _scenario_successful_write,
    _scenario_confirmation_error,
    _scenario_agent_failure_no_fallback,
    _scenario_agent_failure_fallback_also_fails,
    _scenario_agent_fallback_succeeds,
)


def _tuple_key(span) -> tuple[str, str, tuple[str, ...]]:
    name = TOOL_WILDCARD if span.kind == "tool" else span.name
    return (name, span.kind, tuple(sorted(span.attributes.keys())))


def _judge_shape(tracer: InMemoryTracer) -> None:
    """The judge's own span shape (SP8 task 1): D29 runs the judge as a batch teardown stage, never
    wired into the live graph (`atlas_graph.py` never imports `judge/` at all), so this shape is
    never produced by any SCENARIO above -- inventoried directly here instead, the one call site
    that is not a graph scenario."""
    from judge.contract import JudgeContract
    from judge.emission import emit_verdict

    emit_verdict(tracer, None, JudgeContract("judge-model", "groundedness-v1", "template-hash"), "grounded")


def _cost_shape(tracer: InMemoryTracer) -> None:
    """The matrix's own generator cost span shape (SP9 task 5): the SAME batch, report time
    disposition the judge shape above documents (`atlas_graph.py` never imports `matrix/` either) --
    inventoried directly here, the one other call site that is not a graph scenario."""
    from matrix.cost_emission import emit_cost

    emit_cost(tracer, None, model_id="anthropic:claude-sonnet-5", input_tokens=120, output_tokens=45, usd=0.00081)


async def _collect() -> list[tuple[str, str, tuple[str, ...]]]:
    tracer = InMemoryTracer()
    for scenario in SCENARIOS:
        await scenario(tracer)
    _judge_shape(tracer)
    _cost_shape(tracer)
    return sorted({_tuple_key(s) for s in tracer.spans})


def collect_inventory() -> list[tuple[str, str, tuple[str, ...]]]:
    """Drive every scenario once and return the deduped, sorted `(name, kind, attr_keys)` tuples."""
    return asyncio.run(_collect())


def render_inventory() -> str:
    """The exact bytes a committed `contracts/trace/span_inventory.json` holds (and what `--write`
    writes): a JSON array of `{"name", "kind", "attrs"}` objects, sorted, two space indent, one
    trailing newline -- mirrors `contract_tools.mcp_snapshot.render_snapshot`'s own discipline."""
    data = [{"name": name, "kind": kind, "attrs": list(attrs)} for name, kind, attrs in collect_inventory()]
    return json.dumps(data, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the current inventory to contracts/trace/span_inventory.json")
    args = parser.parse_args(argv)

    fresh = render_inventory()
    if args.write:
        INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        INVENTORY_PATH.write_text(fresh)
        print(f"wrote {INVENTORY_PATH}")
        return 0

    committed = INVENTORY_PATH.read_text() if INVENTORY_PATH.is_file() else None
    if committed == fresh:
        print("span_inventory: unchanged")
        return 0
    print(f"span_inventory: DRIFTED from {INVENTORY_PATH} (run with --write to regenerate, then review the diff)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
