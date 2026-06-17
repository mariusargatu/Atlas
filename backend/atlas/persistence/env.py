"""Alembic environment for the checkpointer schema (SP4 task 2).

`ATLAS_PG_DSN` is the ONLY source of the database URL: `alembic.ini` deliberately ships no
`sqlalchemy.url` (a connection string committed there, even a local dev one, is how these leak into
version control), and this file refuses to run with a worded error when the env var is unset,
mirroring `server.py`'s `_resolve_mode`/`_require_provider_sdk` fail fast discipline and
`persistence.checkpointer.postgres_dsn`'s identical refusal at runtime.

SQLAlchemy's bare `postgresql://` scheme defaults to psycopg2, which this repo does not install (it
pins psycopg3, `psycopg[binary]>=3.3.4` in pyproject.toml); the DSN is rewritten to the
`postgresql+psycopg://` dialect before alembic's own bookkeeping connection (the `alembic_version`
table) is opened. The migration itself (`versions/0001_checkpointer.py`) needs its OWN connection to
call the async checkpointer library's `setup()`; it manages that with psycopg3 directly and reads
`ATLAS_PG_DSN` again on its own, unrelated to the SQLAlchemy engine built here. This file's engine
exists only so alembic can track "has 0001 already run" in its own table.
"""
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

if config.config_file_name is not None:
    # disable_existing_loggers=False (SP4 final fix wave carryover): fileConfig's own default
    # (True) sets .disabled = True on every logger that already existed at import time and is not
    # itself named in alembic.ini's own [loggers] section -- "atlas.chat_app" among them, since it
    # is created at plain module import, well before this file ever runs. In a real deployment
    # alembic and the served app are different processes, so this is invisible there; inside one
    # pytest process (this repo's own test_persistence.py exercises alembic directly), the disabled
    # flag otherwise leaks past this file and silently disables every "atlas" logger for the rest
    # of the session, independent of level (see the reproduction test_sse_contract.py's own
    # `test_stream_cassette_miss_emits_error_then_terminal_message_end` relies on: it runs AFTER
    # test_persistence.py in the SAME process, alphabetically).
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# No ORM models back this migration (it only calls the checkpointer library's own setup()), so
# alembic has nothing to autogenerate against.
target_metadata = None


def _require_dsn() -> str:
    dsn = os.environ.get("ATLAS_PG_DSN")
    if not dsn:
        raise RuntimeError(
            "ATLAS_PG_DSN is not set. Alembic needs a Postgres DSN, e.g. "
            "postgresql://atlas:atlas-dev-password@localhost:5433/atlas "
            "(compose's in network DSN is postgresql://atlas:atlas-dev-password@postgres:5432/atlas)."
        )
    return dsn


def _sqlalchemy_url(dsn: str) -> str:
    # This repo pins psycopg3, never psycopg2, so SQLAlchemy must be told the `psycopg` (v3) dialect
    # explicitly: its bare `postgresql://` scheme resolves to psycopg2 by default.
    if dsn.startswith("postgresql://"):
        return "postgresql+psycopg://" + dsn[len("postgresql://") :]
    return dsn


def run_migrations_offline() -> None:
    """Emit SQL without a live connection (`alembic upgrade head --sql`). Unused by `task
    db:upgrade`, kept for completeness since it is the standard alembic entrypoint pair."""
    url = _sqlalchemy_url(_require_dsn())
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"}
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    config.set_main_option("sqlalchemy.url", _sqlalchemy_url(_require_dsn()))
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}), prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
