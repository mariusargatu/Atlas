"""The matrix's spend gate (SP9 task 5): cumulative per provider dollar tracking against a hard
ceiling, so a batched live matrix run can never silently overspend. Pure, immutable (the SAME
"report the state, never mutate it" discipline `atlas.domain.budget`'s `Budget`/`check_budget`
already establish): `SpendGate` never grows in place, `record_spend` returns a NEW instance rather
than touching the one it was given.

Ceilings (hard, per the plan's own "SPEND CEILINGS" global constraint): OpenAI $20 cumulative,
Anthropic $10 cumulative, Ollama always $0 AND always runs (local, free -- never rationed against a
remaining balance at all, the one deliberate exception `check_spend`/`ALWAYS_RUNS` encode).

Pricing table: OWN, small, deliberately NOT `judge/live_provisional.py`'s `_PROVIDER_TIERS`. That
table is explicitly scoped (its own module docstring) to a cheapest first judge model tier sweep
and is, per its own disclosure, "not reusable as is for the primary agent model or a free/local
Ollama provider" -- there are no tiers here at all, one generation rate and one embedding rate per
paid provider, Ollama priced at zero by construction. Recheck before trusting the cost math:
hardcoded vendor pricing goes stale on the vendor's own schedule, not this repo's.

A cell that would exceed its provider's remaining budget is SKIPPED, never silently capped or
truncated (the plan's own "never silent" doctrine, already applied elsewhere in this repo to the
degradation ladder and the contract narrowing rules): `check_spend` only ever REPORTS the decision;
the caller is the one that appends a `DroppedCell` (`dropped_cell_for`) to the run manifest's own
`dropped_cells` list.

`cost_from_usage` is the cost column's own backward compatible read (the cassette schema change
this same task makes, `replay/cassette.py`): `None` (never a silently wrong zero) when no
`usage_metadata` exists at all, exactly the case an OLD cassette replays into.

`build_generator_gateway` is the one place a live matrix run would construct a generator's gateway:
ALWAYS `RECORD` mode, never `LIVE` -- an unchanged cell rerun then replays for free, matching the
three modes `replay/gateway.py` already built (D19's own seam, no new record/replay mechanism).
This function only ever constructs the gateway; whether to call it at all is `check_spend`'s
decision, made first, by the caller.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from replay.gateway import GatewayChatModel

# Pricing snapshot 2026-07-10 (the same date replay.providers.DEFAULT_MODEL_IDS's own comment
# carries for these exact default model ids), USD per 1M tokens. Generation is (input, output);
# embedding is a single rate (no input/output split for an embed call). Recheck before trusting
# this: exactly the kind of hardcoded assumption that goes stale on the vendor's own schedule, not
# this repo's. Anthropic's Sonnet 5 is at introductory pricing through 2026-08-31 ($3.00/$15.00
# after), the same note judge/live_provisional.py's own (separate, NOT reused here) tier table
# already carries for the same model. The openai and anthropic rates here MUST agree with
# judge/live_provisional.py's own _PROVIDER_TIERS entry for the SAME default model id (this is the
# one place both tables price the identical model; test_matrix_spend_gate.py's own
# test_generation_price_agrees_with_the_judges_own_tier_for_the_same_default_model cross checks
# this on every run, so the two tables can never silently drift apart on a shared model again).
GENERATION_PRICE_PER_1M: dict[str, tuple[float, float]] = {
    "openai": (5.00, 30.00),      # gpt-5.6-sol, replay.providers.DEFAULT_MODEL_IDS's own openai default
    "anthropic": (2.00, 10.00),   # claude-sonnet-5, introductory pricing through 2026-08-31
    "ollama": (0.0, 0.0),         # local decode, always free
}

EMBEDDING_PRICE_PER_1M: dict[str, float] = {
    "openai": 0.02,   # text-embedding-3-small
    "ollama": 0.0,    # bge-m3 via TEI is also local/free; never actually routed through this key
}

CEILINGS_USD: dict[str, float] = {"openai": 20.0, "anthropic": 10.0, "ollama": 0.0}

#: Providers this gate never rations: always allowed, regardless of `CEILINGS_USD`/`spent`. A
#: frozenset named explicitly, never a membership test against CEILINGS_USD's own zero value, so a
#: future paid provider that happens to price at $0 today is never silently swept in by an
#: accidental price coincidence.
ALWAYS_RUNS: frozenset[str] = frozenset({"ollama"})


def generation_cost_usd(provider: str, input_tokens: int, output_tokens: int) -> float:
    """Pure cost math from `GENERATION_PRICE_PER_1M`. Raises on an unpriced provider rather than
    silently costing it at zero -- an unpriced provider is a real gap in the table, never a free
    ride by omission."""
    if provider not in GENERATION_PRICE_PER_1M:
        raise ValueError(
            f"no generation price for provider {provider!r} (matrix.spend_gate.GENERATION_PRICE_PER_1M)"
        )
    in_price, out_price = GENERATION_PRICE_PER_1M[provider]
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


def embedding_cost_usd(provider: str, tokens: int) -> float:
    """Pure cost math from `EMBEDDING_PRICE_PER_1M`, the same fail closed rule on an unpriced
    provider that `generation_cost_usd` uses."""
    if provider not in EMBEDDING_PRICE_PER_1M:
        raise ValueError(
            f"no embedding price for provider {provider!r} (matrix.spend_gate.EMBEDDING_PRICE_PER_1M)"
        )
    return (tokens / 1_000_000) * EMBEDDING_PRICE_PER_1M[provider]


@dataclass(frozen=True)
class SpendDecision:
    """`check_spend`'s own report: never a bare bool. `reason` is always populated (even on the
    allowed path), the same "never silent" doctrine applied to a decision that ALLOWS spend too, so
    a manifest reader never has to guess why a cell ran."""

    allowed: bool
    provider: str
    estimated_usd: float
    remaining_before_usd: float
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.reason:
            object.__setattr__(
                self, "reason",
                f"{self.provider}: ${self.estimated_usd:.4f} fits within ${self.remaining_before_usd:.4f} remaining",
            )


@dataclass(frozen=True)
class SpendGate:
    """Cumulative spend so far, per provider. `ceilings` defaults to the hard `CEILINGS_USD` table
    above, constructor overridable only so a test can exercise the drop path against a tiny ceiling
    cheaply -- production code never overrides it. Immutable: `record_spend` returns a NEW
    `SpendGate`; this one is never mutated in place."""

    spent: Mapping[str, float] = field(default_factory=dict)
    ceilings: Mapping[str, float] = field(default_factory=lambda: dict(CEILINGS_USD))

    def spent_usd(self, provider: str) -> float:
        return self.spent.get(provider, 0.0)

    def remaining_usd(self, provider: str) -> float:
        if provider in ALWAYS_RUNS:
            return float("inf")
        return self.ceilings.get(provider, 0.0) - self.spent_usd(provider)


def check_spend(gate: SpendGate, provider: str, estimated_usd: float) -> SpendDecision:
    """Would `estimated_usd` more spend on `provider` fit under its remaining ceiling? Ollama (and
    any future `ALWAYS_RUNS` member) is always allowed, cost or no cost -- the plan's own ceiling of
    zero dollars that always runs regardless, the one deliberate exception. Never mutates `gate`;
    the caller decides whether/when to fold the spend in via `record_spend`.

    LIVE MONEY SAFETY: for every OTHER provider, `estimated_usd` must be a real, positive number to
    be admitted at all. `GeneratorComponent.estimated_usd` defaults to `0.0` (matrix/generators.py's
    own hermetically safe default), so a live driver that forgets to compute an honest upfront estimate
    (`matrix.spend_gate.generation_cost_usd`) would otherwise look, to this function, exactly like a
    free cell -- silently defeating the hard ceiling the plan's own SPEND CEILINGS constraint
    requires. A zero or negative estimate against a paid provider is refused, never silently
    admitted; only `ALWAYS_RUNS` legitimately runs at zero cost, because for that provider zero is
    the real, known price, not an unknown one."""
    remaining = gate.remaining_usd(provider)
    if provider in ALWAYS_RUNS:
        return SpendDecision(True, provider, estimated_usd, remaining, f"{provider} always runs (local, free)")
    if estimated_usd <= 0.0:
        return SpendDecision(
            False, provider, estimated_usd, remaining,
            f"{provider}: a zero or unknown cost estimate can never be admitted against a paid "
            "provider's ceiling -- the caller must pass a real, positive estimated_usd "
            "(matrix.spend_gate.generation_cost_usd), never rely on the hermetically safe 0.0 default",
        )
    if estimated_usd <= remaining:
        return SpendDecision(True, provider, estimated_usd, remaining)
    return SpendDecision(
        False, provider, estimated_usd, remaining,
        f"would exceed {provider}'s remaining budget (${remaining:.4f} left, cell costs ${estimated_usd:.4f})",
    )


def record_spend(gate: SpendGate, provider: str, usd: float) -> SpendGate:
    """A NEW `SpendGate` with `usd` added to `provider`'s running total. Ollama's own spend is still
    recorded (always zero anyway, by `GENERATION_PRICE_PER_1M`'s own construction), so a report over
    `spent` stays honest even for the provider that always runs -- it is simply never what gates
    it."""
    updated = dict(gate.spent)
    updated[provider] = updated.get(provider, 0.0) + usd
    return SpendGate(spent=updated, ceilings=gate.ceilings)


@dataclass(frozen=True)
class DroppedCell:
    """One manifest `dropped_cells` entry: which cell, which provider, and why -- never a silent
    skip, per the plan's own "never a silent cap" doctrine."""

    component_id: str
    provider: str
    reason: str

    def to_dict(self) -> dict:
        return {"component_id": self.component_id, "provider": self.provider, "reason": self.reason}


def dropped_cell_for(component_id: str, decision: SpendDecision) -> DroppedCell:
    """The manifest entry for a cell `check_spend` refused. Callers pass the SAME `SpendDecision`
    `check_spend` returned, so the logged reason is always the exact one that made the call."""
    return DroppedCell(component_id=component_id, provider=decision.provider, reason=decision.reason)


def cost_from_usage(provider: str, usage_metadata: Optional[Mapping[str, Any]]) -> Optional[float]:
    """The cost column's own backward compatible read. `None` (never a silently wrong zero) when no
    usage data exists at all -- exactly what an OLD cassette (recorded before this task) replays
    into: cost reported as UNAVAILABLE, not zero. A real, non empty `usage_metadata` (a NEW RECORD
    mode capture) computes a real number, `provider` in `ALWAYS_RUNS` included (Ollama correctly
    reports $0.0, a real number, not "unavailable" -- it always ran for real, the fact that its
    price is zero is not the same fact as "we do not know")."""
    if not usage_metadata:
        return None
    input_tokens = usage_metadata.get("input_tokens", 0) or 0
    output_tokens = usage_metadata.get("output_tokens", 0) or 0
    return generation_cost_usd(provider, input_tokens, output_tokens)


def build_generator_gateway(
    *, provider: str, model_id: str, inner: Any, cassette_dir: Path,
) -> GatewayChatModel:
    """Every generator axis call in a live matrix run goes through `RECORD` mode, never `LIVE`: an
    unchanged cell rerun then replays for free (D19's own seam, matching the three modes
    `replay/gateway.py` already built, no new record/replay mechanism). `model_id` is tagged with
    `provider` (the same `provider:model` shape `replay.providers.provider_tag` already uses), so a
    provider swap is always visible in the cassette key. Whether to call this at all is
    `check_spend`'s decision, made first, by the caller; this function only ever constructs the
    gateway."""
    return GatewayChatModel(
        model_id=f"{provider}:{model_id}", mode="record", cassette_dir=cassette_dir, inner=inner,
    )


__all__ = [
    "ALWAYS_RUNS",
    "CEILINGS_USD",
    "EMBEDDING_PRICE_PER_1M",
    "GENERATION_PRICE_PER_1M",
    "DroppedCell",
    "SpendDecision",
    "SpendGate",
    "build_generator_gateway",
    "check_spend",
    "cost_from_usage",
    "dropped_cell_for",
    "embedding_cost_usd",
    "generation_cost_usd",
    "record_spend",
]
