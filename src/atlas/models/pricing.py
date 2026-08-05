"""What each model costs, and the refusal that keeps an unpriced one out."""

from __future__ import annotations

# US dollars per million tokens (input, output).
RATES: dict[str, tuple[float, float]] = {
    "gpt-5.6-luna": (1.25, 10.00),
    "gpt-5.4": (1.25, 10.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Deliberately lenient: an unpriced model bills zero rather than raising, since a
    missing price is a reporting gap, not a reason to discard an answer already paid
    for. `require_priced` is what catches this earlier, before money is spent."""
    rate_in, rate_out = RATES.get(model, (0.0, 0.0))
    return (input_tokens * rate_in + output_tokens * rate_out) / 1e6


def require_priced(model: str) -> str:
    """Refuses a model whose price nobody wrote down, before it can spend anything.
    Called at client construction, not inside `cost_usd`, so it fires while the run is
    still worth nothing."""
    if model not in RATES:
        raise ValueError(
            f"no published rate for model {model!r}, so every cost this run reported would "
            f"read $0.0000 while it spent real money. Add it to atlas.models.pricing.RATES. "
            f"Priced models: {', '.join(sorted(RATES))}"
        )
    return model
