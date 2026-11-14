import pytest
from pydantic import SecretStr

from core import settings
from memory.postgres import get_postgres_connection_string


@pytest.fixture
def pg_base(monkeypatch):
    monkeypatch.setattr(settings, "POSTGRES_HOST", "localhost")
    monkeypatch.setattr(settings, "POSTGRES_PORT", 5432)
    monkeypatch.setattr(settings, "POSTGRES_DB", "agent_service")


def test_connection_string_escapes_special_chars(monkeypatch, pg_base):
    """Credentials with URI-reserved chars must be percent-encoded, not interpolated raw."""
    monkeypatch.setattr(settings, "POSTGRES_USER", "user@name")
    monkeypatch.setattr(settings, "POSTGRES_PASSWORD", SecretStr("p@ss:word"))

    conn = get_postgres_connection_string()

    assert conn == "postgresql://user%40name:p%40ss%3Aword@localhost:5432/agent_service"
    assert "p@ss" not in conn


def test_connection_string_plain_credentials_unchanged(monkeypatch, pg_base):
    monkeypatch.setattr(settings, "POSTGRES_USER", "agent")
    monkeypatch.setattr(settings, "POSTGRES_PASSWORD", SecretStr("secret"))

    conn = get_postgres_connection_string()

    assert conn == "postgresql://agent:secret@localhost:5432/agent_service"


def test_connection_string_requires_user(monkeypatch, pg_base):
    monkeypatch.setattr(settings, "POSTGRES_USER", None)
    monkeypatch.setattr(settings, "POSTGRES_PASSWORD", SecretStr("secret"))

    with pytest.raises(ValueError, match="POSTGRES_USER"):
        get_postgres_connection_string()
