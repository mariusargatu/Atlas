"""SP9 task 7: the Ollama qwen2.5:7b LOCAL generator cell, made real.

The staged matrix's generator axis (`matrix.generators`' own module docstring: `{Claude, GPT,
qwen2.5:7b}`) has, until this task, no code anywhere that actually builds a `GeneratorComponent`
for any of the three -- the axis existed only in prose and in `models.lock`'s pinned entries
(SP9 task 3). Per the standing "code first + spend posture" directive, the Claude/GPT cells stay
flag gated behind a spend decision (deferred to the batched live capture session); the Ollama cell
is local, free, and `matrix.spend_gate.ALWAYS_RUNS`, so it is the one cell this task makes real now.
ADR-030 documents WHY this is `qwen2.5:7b` on Ollama CPU decode rather than D28's literal vLLM +
Qwen3-8B GPU burst arm (no RunPod key/infra provisioned for this repo): the substitution is named
explicitly, never a silent swap.

Keyless in tests, the SAME discipline `backend.atlas.adapters.openai_embedding` already
established for its own live SDK seam: `build_ollama_generator_component`'s `inner` argument lets a
hermetic test inject a stub `BaseChatModel` (mirroring `test_gateway.py`'s/`test_matrix_spend_gate.
py`'s own `_StubProvider`), so `_live_ollama_model` -- the only place `replay.providers` (itself
lazy on `langchain_ollama`) is ever imported here -- is never reached by the hermetic PR lane.
`test_matrix_ollama_generator.py`'s own
`test_module_never_imports_replay_providers_at_module_scope` guards this directly, the same AST
technique `test_embedding_port.py` already uses for `openai`.

Every call still routes through `matrix.spend_gate.build_generator_gateway` (RECORD mode, D19's
seam, unchanged): an unchanged cell rerun replays for free, exactly like the Claude/GPT cells will
once they are wired. `estimated_usd` is always `0.0` here -- `qwen2.5:7b` is priced at zero by
`matrix.spend_gate.GENERATION_PRICE_PER_1M["ollama"]` and never rationed against a remaining
balance (`ALWAYS_RUNS`), so a live matrix run's spend gate never has anything to check for this
cell regardless of what `estimated_usd` said.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from matrix.generators import GeneratorComponent
from matrix.spend_gate import build_generator_gateway

_REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_LOCK_PATH = _REPO_ROOT / "models.lock"

PROVIDER = "ollama"


def _ollama_entry(models_lock_path: Path) -> dict:
    """The `models.lock` `generator` entry for `provider="ollama"` (SP9 task 3's pinned axis:
    `qwen2.5:7b`, the API model id shape, `revision == model_id`). Fails closed on a missing entry
    (a models.lock edited to drop the axis) rather than silently building a component with a made
    up identity."""
    data = json.loads(Path(models_lock_path).read_text())
    entry = next((e for e in data.get("generator", []) if e["provider"] == PROVIDER), None)
    if entry is None:
        raise ValueError(f"models.lock ({models_lock_path}) has no generator entry for provider={PROVIDER!r}")
    return entry


def _live_ollama_model(model_id: str) -> BaseChatModel:  # pragma: no cover - live only, needs a running Ollama daemon
    """Reached only when a caller asks for a REAL local generation call (no `inner` injected): a
    lazy import of `replay.providers.build_chat_model`, itself the one place `langchain_ollama` is
    imported (also lazily) -- neither this module nor `replay.providers` needs the `ollama`
    dependency group installed to be collected or unit tested, matching `task install`'s own bare
    `uv sync` (no optional groups) in the hermetic PR lane."""
    from replay.providers import build_chat_model

    return build_chat_model(PROVIDER, model_id)


def build_ollama_generator_component(
    *,
    cassette_dir: Path,
    inner: Optional[BaseChatModel] = None,
    models_lock_path: Path = MODELS_LOCK_PATH,
) -> GeneratorComponent:
    """The matrix's real local generator cell: `models.lock`'s pinned `ollama` entry becomes one
    `GeneratorComponent`, ready to hand `matrix.runner.run_matrix` alongside the (still deferred)
    Claude/GPT cells.

    `inner`, when given (every hermetic test), is used as is -- a stub `BaseChatModel`, never a
    real Ollama daemon call. `inner=None` (an operator's own live capture session) lazily builds
    the real `ChatOllama` via `_live_ollama_model`. Either way the gateway is always RECORD mode
    (`matrix.spend_gate.build_generator_gateway`), never REPLAY/LIVE, matching every other generator
    cell's own contract. `estimated_usd` is always `0.0` -- see the module docstring."""
    entry = _ollama_entry(models_lock_path)
    model_id = entry["model_id"]
    live_inner = inner if inner is not None else _live_ollama_model(model_id)
    gateway = build_generator_gateway(
        provider=PROVIDER, model_id=model_id, inner=live_inner, cassette_dir=Path(cassette_dir),
    )
    return GeneratorComponent(
        component_id=f"{PROVIDER}-{model_id}",
        model_snapshot={"provider": entry["provider"], "model_id": entry["model_id"], "revision": entry["revision"]},
        gateway=gateway,
        estimated_usd=0.0,
    )


__all__ = ["MODELS_LOCK_PATH", "PROVIDER", "build_ollama_generator_component"]
