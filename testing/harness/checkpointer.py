"""The deterministic checkpointer (Spike A outcome).

LangGraph mints the checkpoint id and `ts` itself (a `BaseCheckpointSaver` only stores what
it is handed), so the determinism decision is **exclude everywhere**: use a fresh in memory
saver per test and never hash or assert on `checkpoint_id`/`ts`. Trajectory tests assert on
content (the action result, the trace tree), which is deterministic; the engine's ids stay
out of every digest. This is robust across LangGraph versions, unlike monkeypatching the id
source.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from langgraph.checkpoint.memory import InMemorySaver as _InMemory
except ImportError:  # older LangGraph
    from langgraph.checkpoint.memory import MemorySaver as _InMemory

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def new_checkpointer() -> "BaseCheckpointSaver":
    """A fresh, per test in memory checkpointer (reset == a new instance)."""
    return _InMemory()


__all__ = ["new_checkpointer"]
