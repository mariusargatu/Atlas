"""Provider selection for the gateway's `inner` model (record / eval lanes only).

The gateway (ADR-007) is the provider seam: replay needs no model, record wraps a real
`BaseChatModel`. This factory builds that inner model from env, so swapping Anthropic ↔ Ollama is a
one line change with no edits to the graph. Determinism is preserved by replay, never by the
provider. Ollama is not byte stable, so the rule stays: record once, replay forever; live models
(any provider) never gate a merge.

We construct provider classes directly (no LiteLLM, no `init_chat_model` umbrella): the gateway is
already our provider abstraction layer, so a second one would be redundant. LiteLLM earns its place
only when you need cross vendor routing / budgets / a hosted proxy, a P9 reference concern (ADR-025).

    MODEL_PROVIDER = ollama | anthropic | openai   (default: ollama)
    MODEL_ID       = e.g. qwen2.5:7b | claude-sonnet-4-6 | gpt-5.5
    OLLAMA_BASE_URL= http://localhost:11434

Latest provider models (mid 2026): Anthropic: Opus 4.8 (`claude-opus-4-8`), Sonnet 4.6
(`claude-sonnet-4-6`), Haiku 4.5 (`claude-haiku-4-5-20251001`). OpenAI: GPT-5.5 (`gpt-5.5`) frontier,
plus `gpt-5.5-pro`, `gpt-5-mini`, `gpt-5-nano`. The agent defaults to Sonnet, the judge to GPT-5.5
(cross family, ADR-004); both flow through the gateway, so swaps are env only.
"""
from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

_DEFAULTS = {"ollama": "qwen2.5:7b", "anthropic": "claude-sonnet-4-6", "openai": "gpt-5.5"}  # latest mid 2026


def provider_tag(provider: str | None = None, model_id: str | None = None) -> str:
    """The stable `model_id` that keys the cassette: `provider:model`, so a provider swap is visible."""
    provider = provider or os.environ.get("MODEL_PROVIDER", "ollama")
    model_id = model_id or os.environ.get("MODEL_ID", _DEFAULTS.get(provider, ""))
    return f"{provider}:{model_id}"


def build_chat_model(provider: str | None = None, model_id: str | None = None) -> BaseChatModel:
    """Construct the live provider model for the gateway's `inner` (temperature 0)."""
    provider = provider or os.environ.get("MODEL_PROVIDER", "ollama")
    model_id = model_id or os.environ.get("MODEL_ID", _DEFAULTS.get(provider))
    if model_id is None:
        raise ValueError(f"no MODEL_ID for provider {provider!r}")

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_id,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_id, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_id, temperature=0)
    raise ValueError(f"unknown MODEL_PROVIDER {provider!r} (use ollama | anthropic | openai)")
