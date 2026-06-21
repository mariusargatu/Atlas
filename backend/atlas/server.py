"""ASGI entrypoint for the Atlas product API, wired for the hermetic/replay lane.

`uv run uvicorn atlas.server:app` serves the chat + auth edge against the REPLAYED gateway and the
in memory actions backend, no keys, no live calls. This is what the Playwright E2E lane boots:
deterministic, byte stable, safe to drive a thousand times. The cassettes for the E2E prompts live
under `testing/harness/cassettes/e2e/` (see `testing/harness/recording/seed_e2e_cassettes.py`).
"""
from __future__ import annotations

import os

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory, fixture_kit
from replay.gateway import GatewayChatModel

from atlas.chat_app import make_chat_app
from atlas.domain.accounts import apply_write
from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph

_CASSETTES = os.environ.get("ATLAS_CASSETTES", "testing/harness/cassettes/e2e")
# ATLAS_MODE maps 1:1 to GatewayMode, no translation, so the env value never lies about behaviour:
#   replay (default): zero egress, only committed prompts answer (a miss hard fails)
#   record           : answer via the live provider (default Ollama) AND persist to _CASSETTES
#   live             : answer via the live provider, persist nothing (the eval lane)
_MODE = os.environ.get("ATLAS_MODE", "replay")


def _gateway():
    if _MODE == "replay":
        return GatewayChatModel(model_id="claude-test", cassette_dir=_CASSETTES, mode="replay")
    from replay.providers import build_chat_model, provider_tag

    return GatewayChatModel(model_id=provider_tag(), cassette_dir=_CASSETTES, mode=_MODE, inner=build_chat_model())


def create_app():
    kit = fixture_kit()
    # write through: a confirmed action mutates account state, so a later read reflects it
    backend = ActionsBackend(IdFactory("ref"), writer=apply_write)
    graph = build_atlas_graph(_gateway(), IdFactory("idem"), backend, new_checkpointer())
    return make_chat_app(kit.clock, graph, cors_origins=["http://localhost:5173"])


app = create_app()
