"""Shared fixtures + a deterministic hypothesis profile.

`seed_cassette` writes a cassette through the REAL `Cassette` schema and `FileCassetteStore`, the
same path the gateway reads, replacing six near identical `_seed` helpers that had drifted across
the graph/app tests. One helper, one contract, exercised by the writer it documents.

The hypothesis profile is derandomized so property tests reproduce exactly like the rest of the
regression lane. Variance is the eval lane's job (live model). The PR lane never flickers.
"""
from __future__ import annotations

import os

import pytest
from hypothesis import settings

from replay.cassette_store import seed_cassette as _seed_cassette

from atlas.domain import accounts

# DeepEval phones home (a PostHog event + an api.ipify.org public-IP fetch) on every metric.measure()
# unless opted out. The hermetic gate must make NO network call, and `importorskip("deepeval")` only
# skips on deepeval's ABSENCE — which `uv run` does not guarantee once an operator installed the
# rageval group — so the opt-out is set HERE, before any test imports deepeval, not left to the skip.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

settings.register_profile("ci", derandomize=True, database=None, deadline=None)
settings.load_profile("ci")


@pytest.fixture(autouse=True)
def _reset_account_state():
    """Every test starts from the pristine seed. Write through tests never leak state into reads."""
    accounts.reset_state()
    yield
    accounts.reset_state()


@pytest.fixture
def seed_cassette():
    """Return the shared `seed(cassette_dir, messages, response, model_id="claude-test")` that commits
    a replayable cassette under its content addressed key (replay.cassette_store.seed_cassette)."""
    return _seed_cassette


@pytest.fixture
def build_replay_graph(tmp_path):
    """One wiring for trace-reading tests: a replay-mode atlas_graph over the test's tmp cassette dir.
    Returns a builder `(*, writer=None) -> (graph, tracer, backend)` so the InMemoryTracer +
    GatewayChatModel(replay) + ActionsBackend + build_atlas_graph incantation lives in one place
    instead of being re-inlined per test (and drifting when build_atlas_graph's signature changes).
    Imports are local so tests that never build a graph don't pull in langgraph."""
    from determinism.checkpointer import new_checkpointer
    from determinism.sources import IdFactory
    from replay.gateway import GatewayChatModel
    from tracing import InMemoryTracer

    from atlas.domain.actions import ActionsBackend
    from atlas.orchestration.atlas_graph import build_atlas_graph

    def _build(*, writer=None):
        tracer = InMemoryTracer()
        gw = GatewayChatModel(model_id="claude-test", cassette_dir=tmp_path, mode="replay")
        backend = ActionsBackend(IdFactory("ref"), writer=writer)
        graph = build_atlas_graph(gw, IdFactory("idem"), backend, new_checkpointer(), tracer=tracer)
        return graph, tracer, backend

    return _build
