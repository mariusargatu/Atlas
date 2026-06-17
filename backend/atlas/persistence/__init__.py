"""Checkpointer persistence (SP4 task 2): the Alembic migration that creates the langgraph
checkpointer schema (`versions/0001_checkpointer.py`) and the env driven selection between the
hermetic `InMemorySaver` (default, every test and eval lane) and the real `AsyncPostgresSaver`
(`ATLAS_CHECKPOINTER=postgres`, `checkpointer.py`). See `checkpointer.py`'s module docstring for
the full rationale.
"""
from __future__ import annotations
