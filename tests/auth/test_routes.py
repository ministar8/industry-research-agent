from datetime import UTC, datetime
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr

from auth import routes
from auth.store import AuthUser
from core import settings
from service import app


class FakeAuthStore:
    def __init__(self) -> None:
        self.user = AuthUser(uuid4(), "alice", 0, datetime.now(UTC))

    async def create_user(self, username: str, password: str) -> AuthUser:
        return self.user

    async def authenticate(self, username: str, password: str) -> AuthUser | None:
        if (username, password) == ("alice", "correct horse battery staple"):
            return self.user
        return None

    async def get_user(self, user_id):
        return self.user if user_id == self.user.id else None


def test_register_sets_httponly_auth_cookies(monkeypatch) -> None:
    store = FakeAuthStore()
    monkeypatch.setattr(settings, "JWT_SECRET", SecretStr("test-secret"))
    app.dependency_overrides[routes.get_auth_store] = lambda: store
    try:
        client = TestClient(app)
        response = client.post(
            "/auth/register",
            json={"username": "alice", "password": "correct horse battery staple"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 201
    assert "access_token=" in response.headers["set-cookie"]
    assert "HttpOnly" in response.headers["set-cookie"]


def test_login_rejects_cross_origin(monkeypatch) -> None:
    store = FakeAuthStore()
    monkeypatch.setattr(settings, "JWT_SECRET", SecretStr("test-secret"))
    app.dependency_overrides[routes.get_auth_store] = lambda: store
    try:
        client = TestClient(app)
        response = client.post(
            "/auth/login",
            json={"username": "alice", "password": "correct horse battery staple"},
            headers={"Origin": "https://evil.example"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 403
    assert "set-cookie" not in response.headers


def test_login_allows_matching_origin(monkeypatch) -> None:
    store = FakeAuthStore()
    monkeypatch.setattr(settings, "JWT_SECRET", SecretStr("test-secret"))
    app.dependency_overrides[routes.get_auth_store] = lambda: store
    try:
        client = TestClient(app, base_url="https://testserver")
        response = client.post(
            "/auth/login",
            json={"username": "alice", "password": "correct horse battery staple"},
            headers={"Origin": "https://testserver"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "access_token=" in response.headers["set-cookie"]
