from __future__ import annotations

import pytest

from atlas.models.pricing import RATES, cost_usd, require_priced


def test_cost_usd_matches_the_published_rate_by_hand() -> None:
    # Every other dollar figure in the suite is a hand written literal, so a transposed
    # rate or an off-by-one in the per-million-token division would ship silently.
    rate_in, rate_out = RATES["gpt-5.6-luna"]
    assert cost_usd("gpt-5.6-luna", 1_000_000, 0) == pytest.approx(rate_in)
    assert cost_usd("gpt-5.6-luna", 0, 1_000_000) == pytest.approx(rate_out)
    assert cost_usd("gpt-5.6-luna", 500_000, 200_000) == pytest.approx(
        (500_000 * rate_in + 200_000 * rate_out) / 1e6
    )


def test_cost_usd_is_zero_for_zero_tokens_and_lenient_for_an_unpriced_model() -> None:
    assert cost_usd("gpt-5.6-luna", 0, 0) == 0.0
    # Deliberately lenient: the money is already spent by the time cost_usd runs, so
    # refusing here would lose an answer already paid for.
    assert cost_usd("a-model-nobody-priced", 1_000_000, 1_000_000) == 0.0


def test_require_priced_passes_through_every_rated_model_unchanged() -> None:
    for model in RATES:
        assert require_priced(model) == model


def test_require_priced_refuses_before_anything_is_spent() -> None:
    with pytest.raises(ValueError, match="no published rate"):
        require_priced("a-model-nobody-priced")
