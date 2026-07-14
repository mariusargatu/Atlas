"""Postgres checkpointer persistence, hermetic (SP4 task 2).

Four concerns, all provable with no Postgres reachable:

1. `persistence.checkpointer`'s env driven selection (`checkpointer_kind`, `postgres_dsn`) fails
   fast on a typo/unset var, mirroring `server.py`'s `_resolve_mode` discipline.
2. `versions/0001_checkpointer.py`'s `upgrade()` calls `AsyncPostgresSaver(...).setup()` -- never
   hand written DDL (digest rec 3) -- proven by monkeypatching the saver class entirely.
3. The alembic `env.py` refuses to run (offline or online) when `ATLAS_PG_DSN` is unset, before any
   database connection is attempted.
4. The regression proof mirroring SP3's monkeypatch that explodes pattern
   (`test_chat_app.py::test_create_app_never_touches_pgvector_when_atlas_retriever_is_unset`): the
   served app, with `ATLAS_CHECKPOINTER` unset, never constructs `AsyncPostgresSaver` even across a
   full lifespan cycle AND a real chat turn -- and, symmetrically, DOES wire + close it when the env
   var opts in.
"""
from __future__ import annotations

import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "backend" / "atlas" / "persistence" / "alembic.ini"
MIGRATION_PATH = REPO_ROOT / "backend" / "atlas" / "persistence" / "versions" / "0001_checkpointer.py"


def _load_migration_module():
    # 0001_checkpointer.py cannot be `import`ed by its literal name (leading digit is not a valid
    # Python identifier); alembic itself loads revision files this same way internally.
    spec = importlib.util.spec_from_file_location("atlas_persistence_migration_0001", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- 1. selection logic ----


def test_checkpointer_kind_defaults_to_inmemory_when_env_is_unset(monkeypatch):
    from atlas.persistence.checkpointer import checkpointer_kind

    monkeypatch.delenv("ATLAS_CHECKPOINTER", raising=False)
    assert checkpointer_kind() == "inmemory"


def test_checkpointer_kind_reads_postgres_from_env(monkeypatch):
    from atlas.persistence.checkpointer import checkpointer_kind

    monkeypatch.setenv("ATLAS_CHECKPOINTER", "postgres")
    assert checkpointer_kind() == "postgres"


def test_checkpointer_kind_rejects_an_unknown_value_fail_fast(monkeypatch):
    from atlas.persistence.checkpointer import checkpointer_kind

    monkeypatch.setenv("ATLAS_CHECKPOINTER", "sqlite")  # a typo/unsupported value
    with pytest.raises(RuntimeError, match="ATLAS_CHECKPOINTER"):
        checkpointer_kind()


def test_postgres_dsn_requires_atlas_pg_dsn_with_a_worded_error(monkeypatch):
    from atlas.persistence.checkpointer import postgres_dsn

    monkeypatch.delenv("ATLAS_PG_DSN", raising=False)
    with pytest.raises(RuntimeError, match="ATLAS_PG_DSN"):
        postgres_dsn()


def test_postgres_dsn_passes_through_when_set(monkeypatch):
    from atlas.persistence.checkpointer import postgres_dsn

    monkeypatch.setenv("ATLAS_PG_DSN", "postgresql://atlas:pw@localhost:5433/atlas")
    assert postgres_dsn() == "postgresql://atlas:pw@localhost:5433/atlas"


# ---- 2. the migration calls setup(), never hand written DDL ----


def test_migration_upgrade_calls_asyncpostgressaver_setup(monkeypatch):
    module = _load_migration_module()
    calls: list[str] = []

    class _FakeSaver:
        async def setup(self):
            calls.append("setup")

    class _FakeAsyncPostgresSaver:
        @classmethod
        @asynccontextmanager
        async def from_conn_string(cls, dsn):
            assert dsn == "postgresql://stub-dsn"
            calls.append(f"connect:{dsn}")
            yield _FakeSaver()

    monkeypatch.setattr(module, "AsyncPostgresSaver", _FakeAsyncPostgresSaver)
    monkeypatch.setenv("ATLAS_PG_DSN", "postgresql://stub-dsn")
    module.upgrade()
    assert calls == ["connect:postgresql://stub-dsn", "setup"]


def test_migration_downgrade_is_intentionally_unsupported():
    module = _load_migration_module()
    with pytest.raises(NotImplementedError, match="hand mirroring"):
        module.downgrade()


# ---- 3. env.py refuses to run with no DSN, before touching a database ----


def test_alembic_env_refuses_to_run_when_atlas_pg_dsn_is_unset(monkeypatch):
    from alembic import command
    from alembic.config import Config

    monkeypatch.delenv("ATLAS_PG_DSN", raising=False)
    cfg = Config(str(ALEMBIC_INI))
    with pytest.raises(RuntimeError, match="ATLAS_PG_DSN"):
        command.upgrade(cfg, "head")


# ---- 4. the hermetic default path never constructs AsyncPostgresSaver (SP3 pattern mirror) ----


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _login(client, customer_id):
    r = await client.post("/auth/login", json={"customer_id": customer_id})
    assert r.status_code == 200
    return r.json()["access_token"]


def _server_app(monkeypatch, cassette_dir):
    from atlas import server

    monkeypatch.setenv("ATLAS_MODE", "replay")
    monkeypatch.setenv("ATLAS_CASSETTES", str(cassette_dir))
    return server.create_app()


@pytest.mark.asyncio
async def test_create_app_never_constructs_postgres_saver_when_atlas_checkpointer_is_unset(
    tmp_path, monkeypatch, seed_cassette
):
    """Mirrors test_chat_app.py's
    test_create_app_never_touches_pgvector_when_atlas_retriever_is_unset: `AsyncPostgresSaver` is
    monkeypatched to blow up if constructed at all, so a regression that flips the default would
    fail this test even though a plain chat turn happens to still work. Runs a FULL lifespan cycle
    (TestClient's `with` block) AND an actual chat turn, not just app construction, per the SP4
    task 2 implementation note."""
    from atlas.persistence import checkpointer as persistence_checkpointer
    from starlette.testclient import TestClient

    def _boom(*_a, **_k):
        raise AssertionError("AsyncPostgresSaver must not be constructed when ATLAS_CHECKPOINTER is unset")

    monkeypatch.delenv("ATLAS_CHECKPOINTER", raising=False)
    monkeypatch.setattr(persistence_checkpointer, "AsyncPostgresSaver", _boom)

    from langchain_core.messages import HumanMessage

    seed_cassette(tmp_path, [HumanMessage("What is your name?")], {"content": "Hi.", "tool_calls": []})
    app = _server_app(monkeypatch, tmp_path)  # must not raise

    with TestClient(app):  # fires the real ASGI lifespan startup/shutdown cycle
        pass

    async with _client(app) as client:
        token = await _login(client, "cust_current")
        r = await client.post(
            "/chat", json={"message": "What is your name?", "thread_id": "t1"},
            headers={"authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200
        assert r.json()["final_response"] == "Hi."


@pytest.mark.asyncio
async def test_create_app_wires_postgres_saver_and_closes_its_connection_on_shutdown(tmp_path, monkeypatch):
    """Symmetric to the "never touches" proof above: ATLAS_CHECKPOINTER=postgres DOES swap the
    graph's checkpointer and closes its connection exactly once at shutdown. Hermetic:
    `open_postgres_checkpointer` itself is monkeypatched (its real implementation needs a live
    Postgres), so this proves only the served app's WIRING -- checkpointer_kind() is consulted, the
    constructed saver reaches the graph, and its connection's close() fires on shutdown. Patched on
    `atlas.server` (not `atlas.persistence.checkpointer`): server.py imports the name directly
    (`from atlas.persistence.checkpointer import open_postgres_checkpointer`), which binds its own
    reference in server's namespace, the same reason atlas_graph.py's `PgvectorRetriever` monkeypatch
    targets `atlas_graph.PgvectorRetriever`, not the adapter module it was imported from."""
    from atlas import server
    from starlette.testclient import TestClient

    closed = []

    class _FakeConn:
        async def close(self):
            closed.append(True)

    class _FakeSaver:
        def __init__(self):
            self.conn = _FakeConn()

    async def _fake_open(dsn):
        assert dsn == "postgresql://stub-dsn"
        return _FakeSaver()

    monkeypatch.setenv("ATLAS_CHECKPOINTER", "postgres")
    monkeypatch.setenv("ATLAS_PG_DSN", "postgresql://stub-dsn")
    monkeypatch.setattr(server, "open_postgres_checkpointer", _fake_open)
    app = _server_app(monkeypatch, tmp_path)

    with TestClient(app):
        pass  # startup then shutdown lifespan events fire around this block
    assert closed == [True]
