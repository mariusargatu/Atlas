"""`matrix.spend_gate`, hermetic (SP9 task 5): cumulative per provider dollar tracking against a
hard ceiling, pure arithmetic, no network, no keys. Ollama is the one deliberate exception: always
runs, cost always zero, never rationed against a remaining balance at all.
"""
from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from replay.gateway import GatewayChatModel, GatewayMode

from matrix.spend_gate import (
    ALWAYS_RUNS,
    CEILINGS_USD,
    DroppedCell,
    SpendGate,
    build_generator_gateway,
    check_spend,
    cost_from_usage,
    dropped_cell_for,
    embedding_cost_usd,
    generation_cost_usd,
    record_spend,
)


# ---- pricing table: pure math -------------------------------------------------------------------


def test_generation_cost_usd_charges_input_and_output_at_their_own_rate():
    in_price, out_price = 2.0, 10.0
    from matrix.spend_gate import GENERATION_PRICE_PER_1M

    assert GENERATION_PRICE_PER_1M["anthropic"] == (in_price, out_price)
    cost = generation_cost_usd("anthropic", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(in_price + out_price)


def test_generation_cost_usd_is_zero_for_ollama():
    assert generation_cost_usd("ollama", input_tokens=10_000_000, output_tokens=10_000_000) == 0.0


def test_generation_cost_usd_scales_linearly_with_tokens():
    half = generation_cost_usd("openai", input_tokens=500_000, output_tokens=0)
    full = generation_cost_usd("openai", input_tokens=1_000_000, output_tokens=0)
    assert full == pytest.approx(half * 2)


def test_generation_cost_usd_rejects_an_unpriced_provider():
    with pytest.raises(ValueError, match="voyage"):
        generation_cost_usd("voyage", input_tokens=1, output_tokens=1)


def test_embedding_cost_usd_charges_the_single_embed_rate():
    from matrix.spend_gate import EMBEDDING_PRICE_PER_1M

    cost = embedding_cost_usd("openai", tokens=1_000_000)
    assert cost == pytest.approx(EMBEDDING_PRICE_PER_1M["openai"])


def test_embedding_cost_usd_rejects_an_unpriced_provider():
    with pytest.raises(ValueError, match="anthropic"):
        embedding_cost_usd("anthropic", tokens=1)


def test_pricing_table_is_not_the_judges_provider_tiers():
    """The plan's own docstring rule: this table must not reuse judge/live_provisional.py's
    _PROVIDER_TIERS (scoped to judge model tiers, explicitly not reusable here)."""
    from judge.live_provisional import _PROVIDER_TIERS
    from matrix.spend_gate import GENERATION_PRICE_PER_1M

    assert GENERATION_PRICE_PER_1M is not _PROVIDER_TIERS
    assert set(GENERATION_PRICE_PER_1M) == {"openai", "anthropic", "ollama"}


def test_generation_price_agrees_with_the_judges_own_tier_for_the_same_default_model():
    """Two separate tables (deliberately, per the docstring above), but they must never silently
    disagree on the PRICE OF THE SAME NAMED MODEL. `replay.providers.DEFAULT_MODEL_IDS` names the
    exact openai/anthropic model this gate's own table prices; `judge.live_provisional`'s own tier
    table separately prices that same model id as one entry in its cheapest first sweep. If a
    vendor price changes, both tables change together (or the difference is spelled out in a
    comment); this test is the cross check the two tables never had before, so a future one sided
    edit fails LOUD here rather than silently under (or over) pricing a live cell."""
    from judge.live_provisional import _PROVIDER_TIERS
    from matrix.spend_gate import GENERATION_PRICE_PER_1M
    from replay.providers import DEFAULT_MODEL_IDS

    for provider in ("openai", "anthropic"):
        default_model_id = DEFAULT_MODEL_IDS[provider]
        tier_prices = {model_id: (in_price, out_price) for model_id, in_price, out_price in _PROVIDER_TIERS[provider]}
        assert default_model_id in tier_prices, (
            f"{provider}'s default model {default_model_id!r} has no entry in "
            f"judge.live_provisional's own tier table; the two tables cannot be cross checked"
        )
        assert GENERATION_PRICE_PER_1M[provider] == tier_prices[default_model_id], (
            f"matrix.spend_gate prices {provider}:{default_model_id} at "
            f"{GENERATION_PRICE_PER_1M[provider]}, but judge.live_provisional's own tier table "
            f"for the SAME model prices it at {tier_prices[default_model_id]} -- reconcile the "
            f"two tables (or document the specific reason they intentionally differ)"
        )


# ---- ceilings: the hard numbers named by the plan's own global constraint -----------------------


def test_ceilings_match_the_plans_own_hard_numbers():
    assert CEILINGS_USD == {"openai": 20.0, "anthropic": 10.0, "ollama": 0.0}


def test_ollama_is_the_one_always_runs_provider():
    assert ALWAYS_RUNS == frozenset({"ollama"})


# ---- SpendGate / check_spend / record_spend: the skip and log arithmetic -------------------------


def test_a_fresh_gate_has_zero_spent_and_full_headroom():
    gate = SpendGate()
    assert gate.spent_usd("openai") == 0.0
    assert gate.remaining_usd("openai") == 20.0
    assert gate.remaining_usd("anthropic") == 10.0


def test_check_spend_allows_a_cell_within_budget():
    gate = SpendGate()
    decision = check_spend(gate, "openai", estimated_usd=5.0)
    assert decision.allowed is True
    assert decision.provider == "openai"
    assert decision.remaining_before_usd == 20.0


def test_check_spend_skips_a_cell_over_the_remaining_budget_never_silently():
    gate = SpendGate(spent={"openai": 19.0})
    decision = check_spend(gate, "openai", estimated_usd=5.0)
    assert decision.allowed is False
    assert decision.reason  # never silent: a human readable reason is always present
    assert "openai" in decision.reason


def test_check_spend_at_exactly_the_remaining_budget_is_allowed():
    """The boundary: a cell costing EXACTLY what remains does not overspend."""
    gate = SpendGate(spent={"anthropic": 5.0})
    decision = check_spend(gate, "anthropic", estimated_usd=5.0)
    assert decision.allowed is True


def test_record_spend_returns_a_new_gate_never_mutates_the_old_one():
    gate = SpendGate()
    updated = record_spend(gate, "openai", 3.5)
    assert gate.spent_usd("openai") == 0.0  # the original is untouched
    assert updated.spent_usd("openai") == 3.5
    assert gate is not updated


def test_record_spend_accumulates_across_calls():
    gate = SpendGate()
    gate = record_spend(gate, "anthropic", 2.0)
    gate = record_spend(gate, "anthropic", 3.0)
    assert gate.spent_usd("anthropic") == pytest.approx(5.0)


def test_ollama_always_runs_even_with_zero_remaining_budget():
    gate = SpendGate()  # ollama's own ceiling is $0
    decision = check_spend(gate, "ollama", estimated_usd=0.0)
    assert decision.allowed is True
    assert "ollama" in decision.reason.lower()


def test_ollama_always_runs_even_if_a_caller_mistakenly_estimates_a_nonzero_cost():
    """Ollama is never rationed against a remaining balance at all, per the plan's own
    ceiling of zero dollars that always runs regardless, even when a caller bug estimates a
    nonzero cost -- it still cannot block Ollama."""
    gate = SpendGate()
    decision = check_spend(gate, "ollama", estimated_usd=1.0)
    assert decision.allowed is True


def test_a_provider_ceiling_is_independent_of_the_others():
    gate = SpendGate(spent={"openai": 20.0})  # openai exhausted
    assert check_spend(gate, "openai", estimated_usd=0.01).allowed is False
    assert check_spend(gate, "anthropic", estimated_usd=9.99).allowed is True  # untouched


# ---- live-money safety: a zero/unknown estimate is never silently admitted against a paid --------
# ---- provider's ceiling (`GeneratorComponent.estimated_usd` defaults to 0.0) -----------------------


def test_a_paid_provider_cell_with_a_zero_estimate_is_refused_not_silently_admitted():
    """The real bug this closes: `estimated_usd` defaults to `0.0` on `GeneratorComponent`, so a
    live driver that forgets to pass an honest estimate must never be silently treated as free --
    only `ALWAYS_RUNS` (Ollama) legitimately runs at zero cost. Full headroom on the gate makes the
    point sharply: this is refused for having NO estimate, not for exceeding a budget."""
    gate = SpendGate()  # full $20 openai headroom
    decision = check_spend(gate, "openai", estimated_usd=0.0)
    assert decision.allowed is False
    assert "openai" in decision.reason


def test_anthropic_zero_estimate_is_also_refused_not_only_openai():
    gate = SpendGate()  # full $10 anthropic headroom
    decision = check_spend(gate, "anthropic", estimated_usd=0.0)
    assert decision.allowed is False
    assert "anthropic" in decision.reason


def test_a_negative_estimate_is_refused_the_same_way_as_zero():
    """Never a real live driver value, but a defensive floor: nothing below "a real positive
    estimate" is ever treated as admissible against a paid provider."""
    gate = SpendGate()
    decision = check_spend(gate, "openai", estimated_usd=-1.0)
    assert decision.allowed is False


def test_a_paid_provider_cell_with_a_real_positive_estimate_is_still_admitted_within_budget():
    """The fix narrows exactly the zero/unknown-estimate gap; an honest, real, positive estimate
    within budget is unaffected."""
    gate = SpendGate()
    decision = check_spend(gate, "anthropic", estimated_usd=0.5)
    assert decision.allowed is True


def test_ollama_zero_estimate_still_always_runs_the_one_deliberate_exception():
    """Ollama is priced at $0 by construction (`GENERATION_PRICE_PER_1M["ollama"] == (0.0, 0.0)`)
    and is the one provider this refusal must never catch: `ALWAYS_RUNS` legitimately admits a
    zero estimate because the cost genuinely is zero, not unknown."""
    gate = SpendGate()
    decision = check_spend(gate, "ollama", estimated_usd=0.0)
    assert decision.allowed is True


# ---- dropped_cells: never silent ------------------------------------------------------------------


def test_dropped_cell_for_carries_the_component_id_provider_and_reason():
    gate = SpendGate(spent={"openai": 20.0})
    decision = check_spend(gate, "openai", estimated_usd=0.5)
    dropped = dropped_cell_for("gpt-5.6-sol::config-a", decision)
    assert isinstance(dropped, DroppedCell)
    assert dropped.component_id == "gpt-5.6-sol::config-a"
    assert dropped.provider == "openai"
    assert dropped.reason == decision.reason
    assert dropped.to_dict() == {
        "component_id": "gpt-5.6-sol::config-a",
        "provider": "openai",
        "reason": decision.reason,
    }


# ---- cost_from_usage: the backward compatible cost column read -----------------------------------


def test_cost_from_usage_is_none_when_no_usage_metadata_exists():
    """An OLD cassette replayed with no usage_metadata at all: cost is UNAVAILABLE, never a
    silently wrong zero."""
    assert cost_from_usage("anthropic", None) is None
    assert cost_from_usage("anthropic", {}) is None


def test_cost_from_usage_computes_a_real_number_when_usage_metadata_exists():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "total_tokens": 2_000_000}
    cost = cost_from_usage("anthropic", usage)
    assert cost == pytest.approx(generation_cost_usd("anthropic", 1_000_000, 1_000_000))


def test_cost_from_usage_is_zero_not_none_for_ollama_with_real_usage():
    usage = {"input_tokens": 500, "output_tokens": 500}
    assert cost_from_usage("ollama", usage) == 0.0


# ---- build_generator_gateway: generator calls route through RECORD mode --------------------------


class _StubProvider(BaseChatModel):
    """A minimal stand in for a live provider `inner`; never actually invoked by these tests (only
    the gateway's own MODE is under test here, not a real call). Mirrors test_gateway.py's own
    `_StubProvider` shape, the smallest real `BaseChatModel` `GatewayChatModel` will accept."""

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="stub"))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_build_generator_gateway_always_constructs_record_mode(tmp_path):
    gw = build_generator_gateway(
        provider="anthropic", model_id="claude-sonnet-5", inner=_StubProvider(), cassette_dir=tmp_path
    )
    assert isinstance(gw, GatewayChatModel)
    assert gw.mode is GatewayMode.RECORD


def test_build_generator_gateway_keys_the_model_id_with_the_provider_tag(tmp_path):
    gw = build_generator_gateway(
        provider="openai", model_id="gpt-5.6-sol", inner=_StubProvider(), cassette_dir=tmp_path
    )
    assert gw.model_id == "openai:gpt-5.6-sol"
