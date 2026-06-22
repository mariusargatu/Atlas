"""SP7 Task 6: proof that the corrected dataset contract examples (gc-0001, gc-0002) AND the hand
curated seed dataset's own agentic/multi turn cases actually run through the REAL atlas graph
(`atlas.orchestration.atlas_graph.build_atlas_graph`), via Task 5's runner
(`dataset_tools.multi_turn.run_multi_turn_case`). This is the hard requirement a prior SP7 finding
named by name: a schema valid case that references a tool that does not exist, an intent
`classify_intent` cannot emit, or a tool unreachable without a real action cue is behaviorally
impossible and worthless as a golden case, no matter how cleanly it validates against the schema.

No cassette machinery here: a scripted, cassette free fake model (the same "deterministic,
cassette free" pattern `test_atlas_graph.py`'s own `_StubModel` and `test_ladder.py`'s own fakes
already use for this exact graph) drives each turn from a fixed script keyed by the turn's own
question text, since every case run here is fully scripted end to end (no LLM in the loop, no
network, no keys). Every scripted final answer is checked against the real render guards it must
pass (`atlas.domain.guard`): no forbidden "no contract" style cue substring, no other customer's
name, no unsafe markup.

Covers, per the SP7 Task 6 report: the two corrected contract examples; all 7 hand authored single
turn action (write) cases; all 4 hand authored multi turn trajectory cases; and 3 representative
read only cases (the flagship grounded_not_true baseline, one hallucination bait case, one plain
factoid) proving the troubleshooting -> search_knowledge -> answer path also runs for real, not
only the write surface.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from contract_tools import loader

from atlas.domain import accounts

from dataset_tools.multi_turn import run_multi_turn_case
from testing.tests.fixtures.catalog_expectations import EXPECTED_CURRENT_PLAN

SEED_PATH = Path("testing/harness/dataset_tools/seed_cases.jsonl")


def _load_seed() -> dict[str, dict]:
    cases = [json.loads(line) for line in SEED_PATH.read_text().splitlines() if line.strip()]
    return {c["case_id"]: c for c in cases}


SEED = _load_seed()
EXAMPLES = loader.load_examples("dataset")

# ---- the scripted, cassette free fake model -------------------------------------------------------

# question text -> (tool_name, tool_args, final_answer_or_None). `final_answer` is None for a pure
# write turn: the graph never calls the model a second time for a write (pre_action_guard ->
# confirm -> END, the confirm node's own fixed "Done. ..." text ships, never a model composed one).
_SCRIPT: dict[str, tuple[str, dict, str | None]] = {
    "Is my plan contract free?": (
        "search_knowledge", {"query": "contract terms"},
        "Your account shows a 12 month minimum term on file, so an early exit may incur the "
        "termination fee.",
    ),
    "I want to switch my plan, tell me about the Fiber 100 Legacy plan first.": (
        "get_plan", {"plan_id": "plan_legacy_value"}, "The Fiber 100 Legacy plan has a 12 month term.",
    ),
    "Yes, switch me to it.": ("change_plan", {"plan_id": "plan_legacy_value"}, None),
    "Yes, switch me over to it.": ("change_plan", {"plan_id": "plan_current_fast"}, None),
    "Switch me to the Fiber 100 Legacy plan.": ("change_plan", {"plan_id": "plan_legacy_value"}, None),
    "I want to cancel my contract, switch me to the current plan.": (
        "change_plan", {"plan_id": "plan_current_fast"}, None,
    ),
    "Add the Sky Sports add on to my account.": ("add_addon", {"addon_id": "sky_sports"}, None),
    "Remove the static IP add on from my account.": ("remove_addon", {"addon_id": "static_ip"}, None),
    "Please reset my modem, it is not working.": ("reset_modem", {}, None),
    "Book an engineer visit for 2026-08-01T09:00.": ("book_engineer", {"slot": "2026-08-01T09:00"}, None),
    "Open a ticket about my wifi dropping in the evening.": (
        "open_ticket", {"subject": "Wifi dropping in the evening"}, None,
    ),
    "I want to switch my plan, tell me about the current Fiber 100 plan first.": (
        "get_plan", {"plan_id": "plan_current_fast"}, "The current Fiber 100 plan has no minimum term.",
    ),
    "Can you check my current data usage?": (
        "get_usage", {}, "You have used 240.5 GB this billing period.",
    ),
    "Also, please reset my modem, it is not working.": ("reset_modem", {}, None),
    "And add a Sky Sports add on too.": ("add_addon", {"addon_id": "sky_sports"}, None),
    "Am I over my data cap this month?": (
        "get_usage", {}, "You have used 512.0 GB against a 500 GB cap this period.",
    ),
    "Please upgrade me to the current Fiber 100 plan, no cap.": (
        "change_plan", {"plan_id": "plan_current_fast"}, None,
    ),
    "Do you offer a Quantum 5G plan?": (
        "search_knowledge", {"query": "Quantum 5G"}, "We do not currently offer a Quantum 5G plan.",
    ),
    "OK, what is your fastest current plan then?": (
        "search_knowledge", {"query": "fastest plan"},
        "Our fastest current plan is Fiber 500, offering 500 Mbps download and upload.",
    ),
    "What is the monthly_price of plan-fiber-500?": (
        "search_knowledge", {"query": "plan-fiber-500 monthly_price"},
        "Fiber 500 is priced at $39.99 per month.",
    ),
}


def _text_of(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


class ScriptedTurnModel(BaseChatModel):
    """Every generation call within a turn is decided purely from `_SCRIPT[question]`: the FIRST
    call after the turn's own `HumanMessage` proposes the scripted tool call, the SECOND (once a
    `ToolMessage` answers it) returns the scripted final answer. A pure write turn's script entry
    has `final_answer=None` and is never called a second time (the graph does not loop back to the
    model for a write, see the module docstring), so that branch is simply never reached for those
    questions; the assertion right below would fail loudly instead of silently returning `None` as
    content if the graph ever did call back a second time for one of those questions.
    """

    script: dict[str, tuple[str, dict, str | None]] = {}

    @property
    def _llm_type(self) -> str:
        return "scripted-turn"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError("the graph is async only")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        question = _text_of(messages[last_human])
        tool_name, tool_args, final_answer = self.script[question]
        since_human = messages[last_human + 1 :]
        already_called = any(isinstance(m, ToolMessage) for m in since_human)
        if not already_called:
            calls = [{"name": tool_name, "args": tool_args, "id": f"c{last_human}"}]
            msg = AIMessage(content="", tool_calls=calls)
        else:
            assert final_answer is not None, f"unscripted second call for {question!r}"
            msg = AIMessage(content=final_answer)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _model() -> ScriptedTurnModel:
    return ScriptedTurnModel(script=_SCRIPT)


# ---- every scripted final answer clears the real render guards it will actually be checked against,
# ---- BEFORE any graph run: a bug here would otherwise show up as a confusing handoff deep inside a
# ---- graph run instead of a direct, readable assertion. -------------------------------------------


def test_every_scripted_final_answer_clears_the_render_guards():
    from atlas.domain import guard

    for question, (_, _, answer) in _SCRIPT.items():
        if answer is None:
            continue
        assert guard.check_render_safe(answer).ok, question
        for customer_id in accounts.SEED:
            assert guard.check_no_other_customer(answer, customer_id).ok, (question, customer_id)
        for customer_id in accounts.SEED:
            assert guard.check_render_truth(answer, customer_id).ok, (question, customer_id)


# ---- the two corrected dataset contract examples ---------------------------------------------------


@pytest.mark.asyncio
async def test_gc_0001_runs_through_the_real_graph_and_is_never_a_ghost_tool_call():
    """gc-0001 (contracts/dataset/examples/single_turn_case.json) used to reference
    `account.get_contract`, a tool that does not exist anywhere in `backend/atlas/mcp_servers/`.
    Corrected to `knowledge.search_knowledge` (the real, reachable tool on a troubleshooting turn):
    this proves the corrected case actually completes a turn through the real graph rather than
    merely validating against the schema."""
    case = EXAMPLES["single_turn_case"]
    assert case["case_id"] == "gc-0001"
    result = await run_multi_turn_case(
        case, _model(), customer_id="cust_legacy_term", thread_id="gc-0001"
    )
    assert len(result.turns) == 1
    turn = result.turns[0]
    assert turn.refused is False
    assert turn.final_response  # a real answer shipped, not a dangling empty turn
    assert result.passed is True  # no end_state declared: vacuous pass


@pytest.mark.asyncio
async def test_gc_0002_runs_through_the_real_graph_end_to_end_and_gates_on_the_real_account():
    """gc-0002 (contracts/dataset/examples/multi_turn_case.json) used to bind a RAG registry plan
    id ("fiber-500") to catalog.get_plan/actions.change_plan (KeyError against the real
    atlas.domain.catalog.CATALOG), an intent classify_intent cannot emit ("plan_change"), and a
    turn 1 with no action cue (catalog tools are unreachable on the troubleshooting intent that
    phrasing would actually classify to). Corrected to the real backend catalog, a real
    classify_intent output ("action", earned by "switch my" in turn 1), and the real dotted tool
    names: this proves it now completes both turns and the write actually lands on the real
    account store, not merely that the JSON validates."""
    case = EXAMPLES["multi_turn_case"]
    assert case["case_id"] == "gc-0002"
    result = await run_multi_turn_case(
        case, _model(), customer_id=case["customer_id"], thread_id="gc-0002"
    )

    assert len(result.turns) == 2
    turn1, turn2 = result.turns
    assert turn1.observed_intent == "action"
    assert turn1.intent_match is True
    assert turn1.tool_call_metrics.recall == 1.0
    assert turn1.refused is False
    assert "Done" in turn2.final_response
    assert result.passed is True
    assert accounts.get_account("cust_current").plan_id == "plan_legacy_value"


# ---- all 7 hand authored single turn action cases: real tool, real args, real write -----------------

_ACTION_CASE_IDS = tuple(sorted(
    cid for cid, case in SEED.items()
    if case.get("intent") == "action" and len(case.get("turns", ())) == 1
))


@pytest.mark.parametrize("case_id", _ACTION_CASE_IDS)
@pytest.mark.asyncio
async def test_seed_action_case_runs_through_the_real_graph(case_id: str):
    case = SEED[case_id]
    assert case["intent"] == "action"
    assert len(case["turns"]) == 1
    customer_id = case["customer_id"]
    result = await run_multi_turn_case(case, _model(), customer_id=customer_id, thread_id=case_id)

    assert result.turns[0].observed_intent == "action"
    assert result.turns[0].intent_match is None  # no checkpoint block on a single turn top level case
    assert "Done" in result.turns[0].final_response
    assert result.turns[0].refused is False
    if case["end_state"] is not None:
        assert result.passed is True


# ---- fix round 1 (SP7 Task 6 review, Important #2): 4 of the 5 previously "vacuous end_state" action
# ---- tools (add_addon, remove_addon, book_engineer, open_ticket) mutate a real, tuple valued account
# ---- field, and `_check_account_assertions`'s existing `str(actual) == str(expected)` comparison
# ---- gates on the WHOLE tuple with zero runner code changes. Only `reset_modem` stays vacuous: it is
# ---- a real dead end (`apply_reset_modem` returns `None`, commits nothing), not a tuple indexing
# ---- limitation. -----------------------------------------------------------------------------------

_GATING_ACTION_CASES: tuple[str, ...] = (
    "seed-action-add-sky-sports",
    "seed-action-remove-static-ip",
    "seed-action-book-engineer",
    "seed-action-open-ticket",
)


@pytest.mark.parametrize("case_id", _GATING_ACTION_CASES)
@pytest.mark.asyncio
async def test_seed_action_case_end_state_is_real_and_gating(case_id: str):
    """Each of these 4 cases now carries a real `end_state.account_assertions`, empirically derived
    by running the case through the real graph and reading the resulting `atlas.domain.accounts`
    state back (not hand guessed): `addons`/`bookings` for add_addon/remove_addon/book_engineer, and
    the whole `tickets` tuple (nested `Ticket` dataclass repr) for open_ticket. This proves the
    assertion resolves against a real field and passes on the real post run account, not merely that
    `end_state` is non null."""
    case = SEED[case_id]
    assert case["end_state"] is not None
    assert case["end_state"]["account_assertions"]
    customer_id = case["customer_id"]
    result = await run_multi_turn_case(case, _model(), customer_id=customer_id, thread_id=case_id)

    assert result.passed is True
    assert result.account_assertions
    assert all(a.found and a.passed for a in result.account_assertions)


@pytest.mark.parametrize("case_id", _GATING_ACTION_CASES)
@pytest.mark.asyncio
async def test_seed_action_case_end_state_fails_when_tampered(case_id: str):
    """The gating half of the proof: a DELIBERATELY WRONG `end_state` must fail, not pass vacuously
    no matter what the account looks like. Without this, a case could carry a syntactically present
    but semantically meaningless `end_state` and still always report `passed is True`."""
    case = SEED[case_id]
    tampered = {
        **case,
        "end_state": {
            "account_assertions": [
                {"path": a["path"], "equals": f"WRONG-{a['equals']}"}
                for a in case["end_state"]["account_assertions"]
            ]
        },
    }
    customer_id = case["customer_id"]
    result = await run_multi_turn_case(
        tampered, _model(), customer_id=customer_id, thread_id=f"{case_id}-tampered"
    )

    assert result.passed is False
    assert any(not a.passed for a in result.account_assertions)


@pytest.mark.asyncio
async def test_seed_action_reset_modem_end_state_stays_vacuous_a_genuine_no_op():
    """`actions.reset_modem` -> `atlas.domain.accounts.apply_reset_modem` returns `None` and commits
    nothing: a real operational action with NO persistent account field to assert against at all.
    This is a DIFFERENT reason than the other four action tools above (which all mutate a real tuple
    valued account field and now carry a real gating `end_state`, see
    `test_seed_action_case_end_state_is_real_and_gating`): the prior report's blanket "tuple has no
    index syntax" explanation never actually applied to `reset_modem`, which has no field to index
    into in the first place (SP7 Task 6 review, Minor #2). Proven directly here: the account is BYTE
    IDENTICAL before and after the run."""
    case = SEED["seed-action-reset-modem"]
    assert case["end_state"] is None
    before = accounts.get_account("cust_current")
    result = await run_multi_turn_case(
        case, _model(), customer_id="cust_current", thread_id="seed-action-reset-modem-noop-proof"
    )
    after = accounts.get_account("cust_current")

    assert after == before  # genuinely unchanged, nothing for an end_state to assert against
    assert result.passed is True  # vacuous pass by necessity, not by oversight


# ---- all 4 hand authored multi turn trajectory cases -------------------------------------------------

_MULTI_TURN_CASE_IDS = tuple(sorted(
    cid for cid, case in SEED.items()
    if len(case.get("turns", ())) > 1
))


@pytest.mark.parametrize("case_id", _MULTI_TURN_CASE_IDS)
@pytest.mark.asyncio
async def test_seed_multi_turn_case_runs_through_the_real_graph(case_id: str):
    case = SEED[case_id]
    customer_id = case["customer_id"]
    result = await run_multi_turn_case(case, _model(), customer_id=customer_id, thread_id=case_id)

    assert len(result.turns) == len(case["turns"])
    for turn_result, turn in zip(result.turns, case["turns"]):
        checkpoint = turn.get("checkpoint")
        if checkpoint is None:
            continue
        assert turn_result.observed_intent == checkpoint["expected_intent"]
        assert turn_result.intent_match is True
        assert turn_result.tool_call_metrics.recall == 1.0
        checkpoint_tool = checkpoint["expected_tool_calls"][0]["tool"]
        if checkpoint_tool != "knowledge.search_knowledge":
            # search_knowledge's own "args": {} convention (every seed case with this tool) is a
            # deliberate placeholder: the query text is real free text, not a fixed ground truth
            # value, so an exact argument match is never asked of it here, only tool selection
            # (recall, above). Every OTHER tool's checkpoint args ARE a fixed, checkable value
            # (a plan id, an addon id, a slot), so those keep the strict exact match.
            assert turn_result.tool_call_metrics.argument_exact_match == 1.0
        assert turn_result.refused is False
    if case["end_state"] is not None:
        assert result.passed is True


@pytest.mark.asyncio
async def test_seed_mt_daniel_cancel_contract_end_state_matches_the_real_account_after_the_run():
    case = SEED["seed-mt-daniel-cancel-contract"]
    result = await run_multi_turn_case(
        case, _model(), customer_id="cust_legacy_term", thread_id="seed-mt-daniel-cancel-contract-2"
    )
    assert result.passed is True
    account = accounts.get_account("cust_legacy_term")
    assert account.plan_id == "plan_current_fast"
    assert str(account.bill.amount) == str(EXPECTED_CURRENT_PLAN.monthly_price)


# ---- 3 representative read only cases: flagship baseline, hallucination bait, plain factoid ----------

_READ_ONLY_CASES: dict[str, str] = {
    "seed-flagship-daniel-contract-free": "cust_legacy_term",
    "seed-bait-quantum-5g-1": "cust_current",
    "gen-fact-plan-fiber-500-monthly_price": "cust_current",
}


@pytest.mark.parametrize("case_id", sorted(_READ_ONLY_CASES))
@pytest.mark.asyncio
async def test_seed_read_only_case_runs_through_the_real_graph(case_id: str):
    case = SEED[case_id]
    assert case["intent"] == "troubleshooting"
    customer_id = _READ_ONLY_CASES[case_id]
    result = await run_multi_turn_case(case, _model(), customer_id=customer_id, thread_id=case_id)

    turn = result.turns[0]
    assert turn.observed_intent == "troubleshooting"
    assert turn.refused is False
    assert turn.final_response
    assert result.passed is True  # no end_state on any of these three: vacuous pass
