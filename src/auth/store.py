from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from core.settings import settings
from memory.postgres import get_postgres_connection_string, validate_postgres_config

from .security import hash_password, normalize_username, verify_password


class AuthUser:
    def __init__(
        self, user_id: UUID, username: str, token_version: int, created_at: datetime | None = None
    ) -> None:
        self.id = user_id
        self.username = username
        self.token_version = token_version
        self.created_at = created_at


class AuthStore:
    def __init__(self, pool: AsyncConnectionPool[Any]) -> None:
        self.pool = pool

    async def setup(self) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_users (
                    id UUID PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    token_version INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    is_active BOOLEAN NOT NULL DEFAULT TRUE
                )
                """
            )

    async def create_user(self, username: str, password: str) -> AuthUser:
        normalized = normalize_username(username)
        password_hash = hash_password(password)
        user_id = uuid4()
        try:
            async with self.pool.connection() as connection:
                row = await connection.execute(
                    """
                    INSERT INTO app_users (id, username, password_hash)
                    VALUES (%s, %s, %s)
                    RETURNING id, username, token_version, created_at
                    """,
                    (user_id, normalized, password_hash),
                )
                result = await row.fetchone()
        except UniqueViolation as exc:
            raise ValueError("用户名已注册。") from exc
        if result is None:
            raise RuntimeError("创建用户后未返回用户记录。")
        return _user_from_row(result)

    async def authenticate(self, username: str, password: str) -> AuthUser | None:
        normalized = normalize_username(username)
        async with self.pool.connection() as connection:
            result = await connection.execute(
                """
                SELECT id, username, password_hash, token_version, created_at
                FROM app_users
                WHERE username = %s AND is_active = TRUE
                """,
                (normalized,),
            )
            row = await result.fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return _user_from_row(row)

    async def get_user(self, user_id: UUID) -> AuthUser | None:
        async with self.pool.connection() as connection:
            result = await connection.execute(
                """
                SELECT id, username, token_version, created_at
                FROM app_users
                WHERE id = %s AND is_active = TRUE
                """,
                (user_id,),
            )
            row = await result.fetchone()
        return _user_from_row(row) if row else None

    async def rotate_token_version(self, user_id: UUID, expected_version: int) -> AuthUser | None:
        async with self.pool.connection() as connection:
            result = await connection.execute(
                """
                UPDATE app_users
                SET token_version = token_version + 1
                WHERE id = %s AND token_version = %s AND is_active = TRUE
                RETURNING id, username, token_version, created_at
                """,
                (user_id, expected_version),
            )
            row = await result.fetchone()
        return _user_from_row(row) if row else None

    async def revoke_refresh_tokens(self, user_id: UUID) -> None:
        async with self.pool.connection() as connection:
            await connection.execute(
                "UPDATE app_users SET token_version = token_version + 1 WHERE id = %s",
                (user_id,),
            )


def _user_from_row(row: dict[str, Any]) -> AuthUser:
    return AuthUser(
        user_id=UUID(str(row["id"])),
        username=row["username"],
        token_version=row["token_version"],
        created_at=row.get("created_at"),
    )


@asynccontextmanager
async def get_auth_store():
    validate_postgres_config()
    async with AsyncConnectionPool(
        get_postgres_connection_string(),
        min_size=settings.POSTGRES_MIN_CONNECTIONS_PER_POOL,
        max_size=settings.POSTGRES_MAX_CONNECTIONS_PER_POOL,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "application_name": settings.POSTGRES_APPLICATION_NAME + "-auth",
        },
        check=AsyncConnectionPool.check_connection,
    ) as pool:
        store = AuthStore(pool)
        await store.setup()
        yield store
