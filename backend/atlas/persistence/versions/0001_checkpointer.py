"""Create the langgraph checkpointer schema via AsyncPostgresSaver.setup().

Revision ID: 0001
Revises:
Create Date: 2026-07-19

Digest rec 3: never hand mirror langgraph's checkpoint DDL. Every table this migration creates
(`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`) is owned entirely
by `langgraph-checkpoint-postgres`; this file calls its OWN `setup()` instead of copying CREATE
TABLE statements that would silently drift the next time the dependency bumps its schema. `setup()`
is itself idempotent (it tracks its own version in `checkpoint_migrations`), so running `task
db:upgrade` again against an already migrated database changes nothing, matching every other one shot job
in this repo's compose stack (rag-init's `IF NOT EXISTS` / `ON CONFLICT DO NOTHING` discipline).

`upgrade()` reads `ATLAS_PG_DSN` directly (env.py already refused to get this far if it were unset)
and opens its OWN short lived psycopg3 async connection via `AsyncPostgresSaver.from_conn_string`,
separate from alembic's own SQLAlchemy bookkeeping connection (env.py); alembic here is only the
runner that tracks "has 0001 already run" in `alembic_version`, not the thing issuing the
checkpointer's DDL.

`downgrade()` is intentionally unimplemented: reversing this migration would mean dropping tables
owned by an external library's private schema by hand, exactly the hand mirroring this migration's
`upgrade()` exists to avoid.
"""
from __future__ import annotations

import asyncio
import os

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# alembic's revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


async def _create_checkpointer_schema(dsn: str) -> None:
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()


def upgrade() -> None:
    dsn = os.environ["ATLAS_PG_DSN"]  # env.py already refused to run with this unset
    asyncio.run(_create_checkpointer_schema(dsn))


def downgrade() -> None:
    raise NotImplementedError(
        "downgrade is intentionally unsupported: reversing it would mean hand mirroring the "
        "checkpointer library's private schema, exactly what upgrade() avoids by calling setup()."
    )
