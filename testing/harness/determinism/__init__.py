"""Determinism: pin every source of non-reproducibility so a run repeats byte for byte.

Replay (../replay/) pins the model. This pins everything else, so the only thing that
can differ between two runs is the thing under test (principle 7).

  canonical.py     The digest contract: how any value becomes canonical JSON and a
                   sha256. The cassette key and every run digest come from here, so
                   changing it invalidates cassettes on purpose.
  sources.py       The injectable fixtures (a frozen clock, a seeded RNG, monotonic
                   id and span order counters) bundled as DeterminismKit. Dev/prod
                   inject real ones at the same call sites (duck typed, no base class).
  checkpointer.py  A fresh in memory LangGraph saver per test. The engine's checkpoint
                   id/timestamp stay out of every digest.

Imports use the explicit path (`from determinism.sources import fixture_kit`) so a
reader sees which file a symbol lives in.
"""
