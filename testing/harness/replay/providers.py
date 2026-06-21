"""Provider selection for the gateway's `inner` model (record / eval lanes only).

The gateway (ADR-007) is the provider seam: replay needs no model, record wraps a real
`BaseChatModel`. This factory builds that inner model from env, so swapping Anthropic ↔ Ollama is a
one line change with no edits to the graph. Determinism is preserved by replay, never by the
provider. Ollama is not byte stable, so the rule stays: record once, replay forever. Live models
(any provider) never gate a merge.

We construct provider classes directly (no LiteLLM, no `init_chat_model` umbrella): the gateway is
already our provider abstraction layer, so a second one would be redundant. LiteLLM earns its place
only when you need cross vendor routing / budgets / a hosted proxy (ADR-025).

    MODEL_PROVIDER = ollama | anthropic | openai   (default: ollama)
    MODEL_ID       = e.g. qwen2.5:7b | claude-sonnet-5 | gpt-5.6-sol
    OLLAMA_BASE_URL= http://localhost:11434

Latest provider models (as of 2026-07-10, recheck on a schedule, see ADR-004): Anthropic: Opus 4.8
(`claude-opus-4-8`), Sonnet 5 (`claude-sonnet-5`), Haiku 4.5 (`claude-haiku-4-5-20251001`). OpenAI:
GPT-5.6 Sol (`gpt-5.6-sol`, alias `gpt-5.6`) frontier, plus `gpt-5.6-terra`, `gpt-5.6-luna` (GPT-5.5
was frontier until GPT-5.6 shipped 2026-07-09). The agent defaults to Sonnet, the judge to a GPT
model (cross family, ADR-004), and both flow through the gateway, so swaps are env only. These are
live defaults, not part of any committed cassette: a provider bump here never changes what REPLAY
serves, only what a fresh RECORD/LIVE run would call.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_MODEL_IDS = {"ollama": "qwen2.5:7b", "anthropic": "claude-sonnet-5", "openai": "gpt-5.6-sol"}  # 2026-07-10


def provider_tag(provider: str | None = None, model_id: str | None = None) -> str:
    """The stable `model_id` that keys the cassette: `provider:model`, so a provider swap is visible."""
    provider = provider or os.environ.get("MODEL_PROVIDER", "ollama")
    model_id = model_id or os.environ.get("MODEL_ID", DEFAULT_MODEL_IDS.get(provider, ""))
    return f"{provider}:{model_id}"


def build_chat_model(provider: str | None = None, model_id: str | None = None, **kwargs: Any) -> BaseChatModel:
    """Construct the live provider model for the gateway's `inner`.

    No fixed `temperature` here: determinism is preserved by replay, never by the provider (see the
    module docstring), and a pinned `temperature=0` is no longer even a request every model accepts.
    Confirmed live: Anthropic's `claude-sonnet-5` hard-rejects an explicit `temperature` at all
    (`400 temperature is deprecated for this model`), and OpenAI's GPT-5.6 tier rejects `temperature=0`
    specifically (only its default of 1 is accepted). Newer reasoning-tuned models increasingly own
    their own sampling, so ask for the provider's default rather than assuming every model takes an
    explicit one.

    `**kwargs` forwards straight to the provider constructor (e.g. `max_tokens=64` for a smoke check
    that wants a tiny, cheap completion); every existing caller passes none, so this is additive.
    """
    provider = provider or os.environ.get("MODEL_PROVIDER", "ollama")
    model_id = model_id or os.environ.get("MODEL_ID", DEFAULT_MODEL_IDS.get(provider))
    if model_id is None:
        raise ValueError(f"no MODEL_ID for provider {provider!r}")

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_id, base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"), **kwargs
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_id, **kwargs)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_id, **kwargs)
    raise ValueError(f"unknown MODEL_PROVIDER {provider!r} (use ollama | anthropic | openai)")
