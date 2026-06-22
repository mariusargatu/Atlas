"""The multi turn case runner (SP7 Task 5, D24), hermetic: a scripted case driven through the REAL
atlas graph in replay mode (the gateway's cassette store, no live model, no network), proving the
runner end to end for a read then confirmed write conversation, that a deliberately wrong
`end_state` is caught (gating), and that a deliberately wrong per turn checkpoint is flagged but
never gates (diagnostic).
"""
from __future__ import annotations


import jsonschema
import pytest
from contract_tools import loader
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from determinism.canonical import serialize_tool_result
from replay.gateway import GatewayChatModel

from atlas.domain import accounts, catalog

from dataset_tools import multi_turn
from dataset_tools.multi_turn import AccountAssertionResult, ToolCallMetrics, run_multi_turn_case

from testing.tests.fixtures.catalog_expectations import EXPECTED_LEGACY_PLAN

CUSTOMER = "cust_current"  # Sarah, seeded on plan_current_fast (no term)
TARGET_PLAN = "plan_legacy_value"  # Fiber 100 Legacy: has_term, price from EXPECTED_LEGACY_PLAN

TURN1_USER = "I want to switch my plan, tell me about the Fiber 100 Legacy plan first."
TURN1_ANSWER = "The Fiber 100 Legacy plan has a 12 month term."
TURN2_USER = "Yes, switch me to it."


def _case(**overrides) -> dict:
    case = {
        "case_id": "mt-plan-switch",
        "split": "dev",
        "origin": "authored",
        "candidate_source": None,
        "source_trace_id": None,
        "intent": "action",
        "hop_count": 1,
        "adversarial_class": None,
        "failure_class": None,
        "answerable": True,
        "expected_doc_ids": [],
        # doc_type is a plain "string" in the schema (never nullable); this case is not grounded
        # in any one doc type, so the field is simply omitted, the same convention
        # `dataset_tools.generator._base_case` uses for a case with no doc_type at all.
        "expected_tool_calls": [
            {"tool": "catalog.get_plan", "args": {"plan_id": TARGET_PLAN}},
            {"tool": "actions.change_plan", "args": {"plan_id": TARGET_PLAN}},
        ],
        "expected_facts": [],
        "refusal_class": None,
        "persona": None,
        "turns": [
            {
                "user": TURN1_USER,
                "checkpoint": {
                    "expected_intent": "action",
                    "expected_tool_calls": [{"tool": "catalog.get_plan", "args": {"plan_id": TARGET_PLAN}}],
                    "note": "plan details must come from the catalog, not the model",
                },
            },
            {
                "user": TURN2_USER,
                "checkpoint": {
                    "expected_intent": "action",
                    "expected_tool_calls": [
                        {"tool": "actions.change_plan", "args": {"plan_id": TARGET_PLAN}}
                    ],
                },
            },
        ],
        "end_state": {
            "account_assertions": [
                {"path": "plan_id", "equals": TARGET_PLAN},
                # Decimal -> str typed coercion, exercised for real, not merely asserted in prose
                {"path": "bill.amount", "equals": str(EXPECTED_LEGACY_PLAN.monthly_price)},
            ]
        },
    }
    case.update(overrides)
    return case


def _plan_tool_text() -> str:
    """The exact bytes `catalog_server.get_plan` returns for `TARGET_PLAN`: real catalog data, not
    a guess, so a seeded cassette key matches what the runtime graph will actually produce and ask
    for in its own second call."""
    plan = catalog.get_plan(TARGET_PLAN)
    payload = {
        "id": plan.id, "name": plan.name,
        "has_term": plan.has_term, "early_termination_fee": plan.early_termination_fee,
    }
    return serialize_tool_result(payload)


def _seed_turn1_read(tmp_path, seed_cassette, user1: HumanMessage) -> tuple[AIMessage, ToolMessage]:
    """Seed turn 1's two model calls (the tool call proposal, then the follow up answer once the
    read loop re enters `agent`) and return the reconstructed `(ai, tool_msg)` pair a caller needs
    to key any later turn's cassette on the real accumulated history."""
    get_plan_call = {"name": "get_plan", "args": {"plan_id": TARGET_PLAN}, "id": "c1"}
    seed_cassette(tmp_path, [user1], {"content": "", "tool_calls": [get_plan_call]})
    ai1 = AIMessage(content="", tool_calls=[get_plan_call])
    tool_msg1 = ToolMessage(content=_plan_tool_text(), tool_call_id="c1", name="get_plan")
    seed_cassette(tmp_path, [user1, ai1, tool_msg1], {"content": TURN1_ANSWER, "tool_calls": []})
    return ai1, tool_msg1


def _seed_the_conversation(tmp_path, seed_cassette) -> None:
    """Seed every cassette entry the real graph will need to drive `_case()` end to end: turn 1's
    read (see `_seed_turn1_read`), then turn 2's tool call (a write pauses at confirmation, no
    second model call needed to finish it, mirroring
    `evals.simulation.driver.drive_conversation`'s own established pattern)."""
    user1 = HumanMessage(TURN1_USER)
    ai1, tool_msg1 = _seed_turn1_read(tmp_path, seed_cassette, user1)

    ai1_final = AIMessage(content=TURN1_ANSWER, tool_calls=[])
    user2 = HumanMessage(TURN2_USER)
    change_plan_call = {"name": "change_plan", "args": {"plan_id": TARGET_PLAN}, "id": "c2"}
    seed_cassette(
        tmp_path, [user1, ai1, tool_msg1, ai1_final, user2], {"content": "", "tool_calls": [change_plan_call]}
    )


def _model(tmp_path) -> GatewayChatModel:
    return GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")


# ---- the case itself validates against the dataset contract ------------------------------------


def test_the_scripted_case_validates_against_the_dataset_schema():
    jsonschema.validate(_case(), loader.load_schema("dataset"))


# ---- happy path: read then confirmed write, end to end, gated on the real account state --------


@pytest.mark.asyncio
async def test_a_scripted_multi_turn_case_passes_end_to_end(tmp_path, seed_cassette):
    _seed_the_conversation(tmp_path, seed_cassette)
    result = await run_multi_turn_case(_case(), _model(tmp_path), customer_id=CUSTOMER, thread_id="mt-happy")

    assert result.passed is True
    assert result.case_id == "mt-plan-switch"
    assert len(result.turns) == 2

    turn1, turn2 = result.turns
    assert turn1.observed_intent == "action"
    assert turn1.intent_match is True
    assert turn1.tool_call_metrics == ToolCallMetrics(precision=1.0, recall=1.0, argument_exact_match=1.0)
    assert turn1.refused is False
    assert turn1.final_response == TURN1_ANSWER

    assert turn2.intent_match is True
    assert turn2.tool_call_metrics == ToolCallMetrics(precision=1.0, recall=1.0, argument_exact_match=1.0)
    assert "Done" in turn2.final_response
    assert turn2.refused is False

    assert result.account_assertions == (
        AccountAssertionResult(
            path="plan_id", expected=TARGET_PLAN, actual=TARGET_PLAN, found=True, passed=True
        ),
        AccountAssertionResult(
            path="bill.amount", expected=str(EXPECTED_LEGACY_PLAN.monthly_price),
            actual=EXPECTED_LEGACY_PLAN.monthly_price, found=True, passed=True
        ),
    )
    # the real environment outcome, read straight from the account store (not the audit log alone)
    assert accounts.get_account(CUSTOMER).plan_id == TARGET_PLAN


# ---- a deliberately wrong end_state is caught: GATING ------------------------------------------


@pytest.mark.asyncio
async def test_a_deliberately_wrong_end_state_is_caught(tmp_path, seed_cassette):
    _seed_the_conversation(tmp_path, seed_cassette)
    wrong_end_state = {"account_assertions": [{"path": "plan_id", "equals": "plan_current_fast"}]}
    case = _case(end_state=wrong_end_state)
    result = await run_multi_turn_case(case, _model(tmp_path), customer_id=CUSTOMER, thread_id="mt-wrong")

    assert result.passed is False
    assert result.account_assertions == (
        AccountAssertionResult(
            path="plan_id", expected="plan_current_fast", actual=TARGET_PLAN, found=True, passed=False
        ),
    )


@pytest.mark.asyncio
async def test_an_unresolvable_account_path_fails_its_own_assertion(tmp_path, seed_cassette):
    _seed_the_conversation(tmp_path, seed_cassette)
    end_state = {"account_assertions": [{"path": "no_such_field", "equals": "anything"}]}
    case = _case(end_state=end_state)
    result = await run_multi_turn_case(case, _model(tmp_path), customer_id=CUSTOMER, thread_id="mt-badpath")

    assert result.passed is False
    assert result.account_assertions == (
        AccountAssertionResult(
            path="no_such_field", expected="anything", actual=None, found=False, passed=False
        ),
    )


# ---- a deliberately wrong per turn checkpoint is flagged but never gates: DIAGNOSTIC ------------


@pytest.mark.asyncio
async def test_a_wrong_expected_intent_is_flagged_but_never_gates(tmp_path, seed_cassette):
    _seed_the_conversation(tmp_path, seed_cassette)
    case = _case()
    wrong_checkpoint = {**case["turns"][0]["checkpoint"], "expected_intent": "troubleshooting"}
    case["turns"] = [{**case["turns"][0], "checkpoint": wrong_checkpoint}, case["turns"][1]]
    result = await run_multi_turn_case(
        case, _model(tmp_path), customer_id=CUSTOMER, thread_id="mt-wrong-intent"
    )

    turn1 = result.turns[0]
    assert turn1.observed_intent == "action"
    assert turn1.expected_intent == "troubleshooting"
    assert turn1.intent_match is False
    # the checkpoint mismatch never touches the gating verdict, driven only by account_assertions
    assert result.passed is True


@pytest.mark.asyncio
async def test_a_wrong_expected_tool_call_is_flagged_but_never_gates(tmp_path, seed_cassette):
    _seed_the_conversation(tmp_path, seed_cassette)
    case = _case()
    wrong_checkpoint = {
        "expected_intent": "action",
        "expected_tool_calls": [{"tool": "catalog.get_plan", "args": {"plan_id": "plan_current_fast"}}],
    }
    case["turns"] = [{**case["turns"][0], "checkpoint": wrong_checkpoint}, case["turns"][1]]
    result = await run_multi_turn_case(
        case, _model(tmp_path), customer_id=CUSTOMER, thread_id="mt-wrong-args"
    )

    turn1 = result.turns[0]
    # the tool NAME matches (precision/recall are name only, per tool_call_metrics' own docstring);
    # only the argument differs, so argument_exact_match alone catches the planted divergence.
    assert turn1.tool_call_metrics == ToolCallMetrics(precision=1.0, recall=1.0, argument_exact_match=0.0)
    assert result.passed is True  # still gated only by end_state, which this case never touched


# ---- diagnostic fields are None when a turn carries no checkpoint block ------------------------


@pytest.mark.asyncio
async def test_a_turn_with_no_checkpoint_block_has_no_diagnostic_verdict(tmp_path, seed_cassette):
    _seed_the_conversation(tmp_path, seed_cassette)
    case = _case()
    case["turns"] = [{"user": TURN1_USER}, case["turns"][1]]  # turn 1 carries no checkpoint at all
    result = await run_multi_turn_case(
        case, _model(tmp_path), customer_id=CUSTOMER, thread_id="mt-no-checkpoint"
    )

    turn1 = result.turns[0]
    assert turn1.expected_intent is None
    assert turn1.intent_match is None
    assert turn1.tool_call_metrics is None
    assert turn1.observed_tool_calls  # the tool call still happened; only the diagnostic is silent


# ---- a case with no end_state at all passes vacuously -------------------------------------------


@pytest.mark.asyncio
async def test_a_case_with_no_end_state_passes_vacuously(tmp_path, seed_cassette):
    _seed_turn1_read(tmp_path, seed_cassette, HumanMessage(TURN1_USER))

    single_turn_case = _case(turns=_case()["turns"][:1], end_state=None)
    result = await run_multi_turn_case(
        single_turn_case, _model(tmp_path), customer_id=CUSTOMER, thread_id="mt-no-end-state"
    )

    assert result.account_assertions == ()
    assert result.passed is True


# ---- fresh thread/checkpointer per call: two runs of the same case never interfere -------------


@pytest.mark.asyncio
async def test_two_independent_runs_of_the_same_case_do_not_interfere(tmp_path, seed_cassette):
    _seed_the_conversation(tmp_path, seed_cassette)
    model = _model(tmp_path)
    accounts.reset_state()
    first = await run_multi_turn_case(_case(), model, customer_id=CUSTOMER, thread_id="mt-same-id")
    accounts.reset_state()
    second = await run_multi_turn_case(_case(), model, customer_id=CUSTOMER, thread_id="mt-same-id")

    assert first.passed is True
    assert second.passed is True
    assert first.turns[1].final_response == second.turns[1].final_response


# ---- default thread_id and ids/backend wiring ----------------------------------------------------


@pytest.mark.asyncio
async def test_thread_id_defaults_to_the_case_id(tmp_path, seed_cassette):
    _seed_the_conversation(tmp_path, seed_cassette)
    result = await run_multi_turn_case(_case(), _model(tmp_path), customer_id=CUSTOMER)
    assert result.thread_id == "mt-plan-switch"


# ---- _bare_tool_name: unit coverage of the diagnostic namespace normalization -------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("catalog.get_plan", "get_plan"),
        ("actions.change_plan", "change_plan"),
        ("account.get_bill", "get_bill"),
        ("knowledge.search_knowledge", "search_knowledge"),
        ("get_plan", "get_plan"),  # already bare, no dot at all
        ("unknown_namespace.get_plan", "unknown_namespace.get_plan"),  # unrecognized prefix, untouched
    ],
)
def test_bare_tool_name(raw, expected):
    assert multi_turn._bare_tool_name(raw) == expected


# ---- _resolve_path: unit coverage of the dot path traversal -------------------------------------


def test_resolve_path_over_a_frozen_dataclass_attribute():
    account = accounts.get_account(CUSTOMER)
    assert multi_turn._resolve_path(account, "plan_id") == (True, account.plan_id)


def test_resolve_path_over_a_nested_dataclass_attribute():
    account = accounts.get_account(CUSTOMER)
    assert multi_turn._resolve_path(account, "bill.amount") == (True, account.bill.amount)


def test_resolve_path_unresolvable_segment_is_reported_not_raised():
    account = accounts.get_account(CUSTOMER)
    assert multi_turn._resolve_path(account, "not_a_real_field") == (False, None)
    assert multi_turn._resolve_path(account, "bill.not_a_real_field") == (False, None)
