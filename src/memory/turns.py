"""Durable turn metrics for the runtime dashboard.

One row per finished turn, written through the same DATABASE_TYPE choice as the
checkpointer so SQLite and PostgreSQL deployments behave alike. MongoDB has no
turn store and falls back to the in-process collector in core.metrics.
"""

from contextlib import asynccontextmanager
from typing import Any

import aiosqlite
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from core.metrics import RECENT_LIMIT
from core.settings import settings
from memory.postgres import get_postgres_connection_string, validate_postgres_config

_COLUMNS = (
    "thread_id",
    "user_id",
    "status",
    "started_at",
    "finished_at",
    "model",
    "query_summary",
    "report_title",
    "sources_count",
    "error_detail",
    "elapsed",
    "llm_calls",
    "tokens_in",
    "tokens_out",
    "search_calls",
    "fetch_calls",
    "search_failed",
    "fetch_failed",
)

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS turn_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT '',
    finished_at TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '-',
    query_summary TEXT NOT NULL DEFAULT '',
    report_title TEXT NOT NULL DEFAULT '',
    sources_count INTEGER NOT NULL DEFAULT 0,
    error_detail TEXT NOT NULL DEFAULT '',
    elapsed REAL NOT NULL,
    llm_calls INTEGER NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    search_calls INTEGER NOT NULL,
    fetch_calls INTEGER NOT NULL,
    search_failed INTEGER NOT NULL,
    fetch_failed INTEGER NOT NULL
)
"""

_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS turn_metrics (
    id BIGSERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT '',
    finished_at TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '-',
    query_summary TEXT NOT NULL DEFAULT '',
    report_title TEXT NOT NULL DEFAULT '',
    sources_count INTEGER NOT NULL DEFAULT 0,
    error_detail TEXT NOT NULL DEFAULT '',
    elapsed DOUBLE PRECISION NOT NULL,
    llm_calls INTEGER NOT NULL,
    tokens_in INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    search_calls INTEGER NOT NULL,
    fetch_calls INTEGER NOT NULL,
    search_failed INTEGER NOT NULL,
    fetch_failed INTEGER NOT NULL
)
"""


def _public(row: Any) -> dict[str, Any]:
    return {column: row[column] for column in _COLUMNS if column != "user_id"}


async def _ensure_sqlite_columns(connection: aiosqlite.Connection) -> None:
    cursor = await connection.execute("PRAGMA table_info(turn_metrics)")
    existing = {row[1] for row in await cursor.fetchall()}
    definitions = {
        "started_at": "TEXT NOT NULL DEFAULT ''",
        "finished_at": "TEXT NOT NULL DEFAULT ''",
        "model": "TEXT NOT NULL DEFAULT '-'",
        "query_summary": "TEXT NOT NULL DEFAULT ''",
        "report_title": "TEXT NOT NULL DEFAULT ''",
        "sources_count": "INTEGER NOT NULL DEFAULT 0",
        "error_detail": "TEXT NOT NULL DEFAULT ''",
    }
    for column, definition in definitions.items():
        if column not in existing:
            await connection.execute(f"ALTER TABLE turn_metrics ADD COLUMN {column} {definition}")
    await connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_turn_metrics_user_id ON turn_metrics(user_id)"
    )


async def _ensure_postgres_columns(connection: Any) -> None:
    definitions = {
        "started_at": "TEXT NOT NULL DEFAULT ''",
        "finished_at": "TEXT NOT NULL DEFAULT ''",
        "model": "TEXT NOT NULL DEFAULT '-'",
        "query_summary": "TEXT NOT NULL DEFAULT ''",
        "report_title": "TEXT NOT NULL DEFAULT ''",
        "sources_count": "INTEGER NOT NULL DEFAULT 0",
        "error_detail": "TEXT NOT NULL DEFAULT ''",
    }
    for column, definition in definitions.items():
        await connection.execute(
            f"ALTER TABLE turn_metrics ADD COLUMN IF NOT EXISTS {column} {definition}"
        )
    await connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_turn_metrics_user_id ON turn_metrics(user_id)"
    )


class SqliteTurnStore:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self.connection = connection

    async def append(self, snapshot: dict[str, Any]) -> None:
        placeholders = ", ".join("?" for _ in _COLUMNS)
        await self.connection.execute(
            f"INSERT INTO turn_metrics ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
            [snapshot[column] for column in _COLUMNS],
        )
        await self.connection.commit()

    async def recent(
        self, user_id: str | None = None, limit: int = RECENT_LIMIT
    ) -> list[dict[str, Any]]:
        sql = f"SELECT {', '.join(_COLUMNS)} FROM turn_metrics"
        params: list[Any] = []
        if user_id is not None:
            sql += " WHERE user_id = ?"
            params.append(user_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        cursor = await self.connection.execute(sql, params)
        return [_public(row) for row in await cursor.fetchall()]


class PostgresTurnStore:
    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self.pool = pool

    async def append(self, snapshot: dict[str, Any]) -> None:
        placeholders = ", ".join("%s" for _ in _COLUMNS)
        async with self.pool.connection() as connection:
            await connection.execute(
                f"INSERT INTO turn_metrics ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
                [snapshot[column] for column in _COLUMNS],
            )

    async def recent(
        self, user_id: str | None = None, limit: int = RECENT_LIMIT
    ) -> list[dict[str, Any]]:
        sql = f"SELECT {', '.join(_COLUMNS)} FROM turn_metrics"
        params: list[Any] = []
        if user_id is not None:
            sql += " WHERE user_id = %s"
            params.append(user_id)
        sql += " ORDER BY id DESC LIMIT %s"
        params.append(limit)
        async with self.pool.connection() as connection:
            cursor = await connection.execute(sql, params)
            return [_public(row) for row in await cursor.fetchall()]


@asynccontextmanager
async def get_sqlite_turn_store():
    async with aiosqlite.connect(settings.SQLITE_DB_PATH) as connection:
        connection.row_factory = aiosqlite.Row
        await connection.execute(_SQLITE_SCHEMA)
        await _ensure_sqlite_columns(connection)
        await connection.commit()
        yield SqliteTurnStore(connection)


@asynccontextmanager
async def get_postgres_turn_store():
    validate_postgres_config()
    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
        max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "application_name": settings.POSTGRES_APPLICATION_NAME + "-turns",
        },
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        async with pool.connection() as connection:
            await connection.execute(_POSTGRES_SCHEMA)
            await _ensure_postgres_columns(connection)
        yield PostgresTurnStore(pool)
