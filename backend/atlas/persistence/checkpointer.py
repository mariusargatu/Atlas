"""Checkpointer selection for the graph's persistence layer (SP4 task 2).

`ATLAS_CHECKPOINTER` picks which saver backs the graph, mirroring `atlas_graph.select_retriever`'s
env driven adapter selection (D36 tier 2 pattern): unset (or "inmemory") keeps every hermetic test
and eval lane on the harness's deterministic in memory saver
(`testing/harness/determinism/checkpointer.new_checkpointer`, already the ONLY checkpointer
`server.py` has ever constructed); "postgres" selects `AsyncPostgresSaver`. This repo invokes the
graph exclusively via `ainvoke`/`aget_state` (`chat_app.py`'s `/chat` and `/chat/resume`), so the
ASYNC saver is the one that matches the call style; the sync `PostgresSaver` would need its blocking
calls run in a thread executor for no benefit here.

`AsyncPostgresSaver` captures `asyncio.get_running_loop()` at construction (see
`langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.__init__`), so it MUST be built on the event
loop that will actually serve requests, never at cold `create_app()` import time (uvicorn's loop
does not exist yet then). `server.py` therefore compiles the graph with a placeholder
`InMemorySaver` (the exact same one an unset `ATLAS_CHECKPOINTER` produces) and swaps in the real
saver from inside the FastAPI lifespan, once uvicorn's loop is running. That swap is safe because
`CompiledStateGraph.checkpointer` is a plain instance attribute langgraph reads fresh on every
invoke (`self.checkpointer`, see `langgraph.pregel.Pregel`), never a value captured once at compile
time, so reassigning it post compile but pre first-request changes nothing about how the graph reads
it.

`AsyncPostgresSaver` is imported at MODULE level (like `atlas_graph.py` imports `PgvectorRetriever`
at module level), so a hermetic test can monkeypatch this module's `AsyncPostgresSaver` name to
explode and prove the default (`ATLAS_CHECKPOINTER` unset) path never constructs it.
"""
from __future__ import annotations

import os

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row

_KNOWN_CHECKPOINTERS = ("inmemory", "postgres")


def checkpointer_kind(kind: str | None = None) -> str:
    """`ATLAS_CHECKPOINTER` selection, fail fast on a typo (mirrors server.py's
    `_resolve_mode`/`atlas_graph.select_retriever`'s discipline: an unrecognized value is a config
    error, never a silent fallback to the hermetic adapter)."""
    kind = kind or os.environ.get("ATLAS_CHECKPOINTER", "inmemory")
    if kind not in _KNOWN_CHECKPOINTERS:
        raise RuntimeError(
            f"unknown ATLAS_CHECKPOINTER={kind!r}; expected one of {'|'.join(_KNOWN_CHECKPOINTERS)}"
        )
    return kind


def postgres_dsn() -> str:
    """`ATLAS_PG_DSN`, required whenever the postgres checkpointer is selected. Mirrors the alembic
    `env.py` refusal: unset must fail loudly at startup, not fall back to a default that could
    silently point at nothing (or, worse, at another environment's database)."""
    dsn = os.environ.get("ATLAS_PG_DSN")
    if not dsn:
        raise RuntimeError(
            "ATLAS_CHECKPOINTER=postgres needs ATLAS_PG_DSN set, e.g. "
            "postgresql://atlas:atlas-dev-password@localhost:5433/atlas "
            "(compose's in network DSN is postgresql://atlas:atlas-dev-password@postgres:5432/atlas)."
        )
    return dsn


async def open_postgres_checkpointer(dsn: str) -> AsyncPostgresSaver:
    """Open the `AsyncPostgresSaver`'s connection ON the running loop. Must be awaited from inside
    the lifespan (never at cold `create_app()` time, see module docstring). The returned saver's
    connection is `saver.conn`; the caller closes it at shutdown, mirroring `PgvectorRetriever`'s
    explicit `close()` (the only other adapter this app owns a resource lifecycle for). `autocommit`
    + `prepare_threshold=0` + `dict_row` match the defaults `AsyncPostgresSaver.from_conn_string`
    itself uses (its short lived, single connection convenience constructor); this function is the
    long lived equivalent a FastAPI lifespan needs, per the langgraph docs' own guidance for
    integrating the Postgres saver into a long running server."""
    conn = await AsyncConnection.connect(dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    return AsyncPostgresSaver(conn=conn)


__all__ = ["checkpointer_kind", "postgres_dsn", "open_postgres_checkpointer"]
