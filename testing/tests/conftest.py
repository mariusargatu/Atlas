"""Shared fixtures + a deterministic hypothesis profile.

`seed_cassette` writes a cassette through the REAL `Cassette` schema and `FileCassetteStore`, the
same path the gateway reads, replacing six near identical `_seed` helpers that had drifted across
the graph/app tests. One helper, one contract, exercised by the writer it documents.

The hypothesis profile is derandomized so property tests reproduce exactly like the rest of the
regression lane. Variance is the eval lane's job (live model); the PR lane never flickers.
"""
from __future__ import annotations

import pytest
from hypothesis import settings

from replay.cassette_store import seed_cassette as _seed_cassette

from atlas.domain import accounts

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
