import pytest

from core import settings
from core.settings import DatabaseType
from memory import initialize_turn_store
from memory.turns import get_sqlite_turn_store


def _snapshot(thread_id: str, user_id: str = "alice") -> dict:
    return {
        "thread_id": thread_id,
        "user_id": user_id,
        "status": "ok",
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "model": "fake-model",
        "query_summary": "研究储能市场",
        "report_title": "储能市场研究",
        "sources_count": 3,
        "error_detail": "",
        "elapsed": 1.5,
        "llm_calls": 2,
        "tokens_in": 100,
        "tokens_out": 20,
        "search_calls": 1,
        "fetch_calls": 1,
        "search_failed": 0,
        "fetch_failed": 0,
    }


@pytest.mark.asyncio
async def test_sqlite_turn_store_roundtrip_filter_and_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "turns.db"))

    async with get_sqlite_turn_store() as store:
        await store.append(_snapshot("t1"))
        await store.append(_snapshot("t2", user_id="bob"))
        await store.append(_snapshot("t3"))

        rows = await store.recent()
        assert [row["thread_id"] for row in rows] == ["t3", "t2", "t1"]
        assert all("user_id" not in row for row in rows)
        assert rows[0]["tokens_in"] == 100
        assert rows[0]["model"] == "fake-model"
        assert rows[0]["sources_count"] == 3
        assert rows[0]["query_summary"] == "研究储能市场"
        assert [row["thread_id"] for row in await store.recent("alice")] == ["t3", "t1"]
        assert [row["thread_id"] for row in await store.recent(limit=1)] == ["t3"]

    # Reopening the file keeps the rows: this is the restart-survival guarantee.
    async with get_sqlite_turn_store() as store:
        assert [row["thread_id"] for row in await store.recent()] == ["t3", "t2", "t1"]


@pytest.mark.asyncio
async def test_turn_store_dispatch_follows_database_type(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SQLITE_DB_PATH", str(tmp_path / "turns.db"))
    monkeypatch.setattr(settings, "DATABASE_TYPE", DatabaseType.SQLITE)
    async with initialize_turn_store() as store:
        assert store is not None

    monkeypatch.setattr(settings, "DATABASE_TYPE", DatabaseType.MONGO)
    async with initialize_turn_store() as store:
        assert store is None
