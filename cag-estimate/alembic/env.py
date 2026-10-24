import asyncio
from logging.config import fileConfig

import pgvector.sqlalchemy
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from cag_estimate.config import get_settings

# The models module is imported here in P2 so autogenerate sees every table.
from cag_estimate.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_database_url() -> str:
    url = get_settings().database_url
    if url is None:
        raise RuntimeError("DATABASE_URL is not set; Alembic needs it to run migrations.")
    return url


def run_migrations_offline() -> None:
    """Emit SQL without a database connection (`alembic upgrade head --sql`)."""
    context.configure(
        url=get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    # Reflection and autogenerate need to know the pgvector column type.
    connection.dialect.ischema_names["vector"] = pgvector.sqlalchemy.Vector
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = create_async_engine(get_database_url(), poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
