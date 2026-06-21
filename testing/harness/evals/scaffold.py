"""Shared REPLAY wiring for the eval lanes.

The eval and drift lanes all need the same thing: a pinned gateway Atlas graph plus the tracer
wired into it. Retyping that `build_atlas_graph(gateway, IdFactory(...), ActionsBackend(...),
new_checkpointer(), tracer=...)` incantation in every demo and test is the kind of near duplicate
that drifts, the same reason conftest centralized `seed_cassette`. This is the one definition. A
change to the graph's wiring (a new dependency, a different fake) lands here, not in N call sites.
"""
from __future__ import annotations

from determinism.checkpointer import new_checkpointer
from determinism.sources import IdFactory
from replay.gateway import GatewayChatModel
from tracing import InMemoryTracer

from atlas.domain import accounts
from atlas.domain.actions import ActionsBackend
from atlas.orchestration.atlas_graph import build_atlas_graph

DEFAULT_MODEL_ID = "claude-test"


def build_replay_graph(cassette_dir, *, model_id: str = DEFAULT_MODEL_ID):
    """Return ``(graph, tracer)``, a REPLAY pinned Atlas graph and the tracer it writes to.

    Deterministic id factories and a fresh in memory checkpointer. The gateway replays from
    ``cassette_dir`` (a miss is a hard fail). Account state is reset to the pristine seed so each
    trial starts clean, and the runner that repeats trials relies on this.
    """
    accounts.reset_state()
    tracer = InMemoryTracer()
    gateway = GatewayChatModel(model_id=model_id, cassette_dir=cassette_dir, mode="replay")
    graph = build_atlas_graph(
        gateway, IdFactory("idem"), ActionsBackend(IdFactory("ref")), new_checkpointer(), tracer=tracer
    )
    return graph, tracer


__all__ = ["DEFAULT_MODEL_ID", "build_replay_graph"]
