from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import SecretStr

from auth.security import (
    decode_token,
    hash_password,
    issue_refresh_token,
    normalize_username,
    verify_password,
)
from core import settings


def test_password_hash_is_not_reversible_and_verifies() -> None:
    encoded = hash_password("correct horse battery staple")

    assert encoded != "correct horse battery staple"
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong password", encoded)


def test_username_and_password_validation() -> None:
    assert normalize_username("  Alice_1 ") == "alice_1"
    with pytest.raises(ValueError):
        normalize_username("no spaces")
    with pytest.raises(ValueError):
        hash_password("short")


def test_refresh_token_has_type_and_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "JWT_SECRET", SecretStr("test-secret"))
    token = issue_refresh_token(uuid4(), "alice", 0)

    claims = decode_token(token, "refresh")

    assert claims["type"] == "refresh"
    assert claims["version"] == 0
    assert (
        claims["exp"] - claims["iat"]
        == timedelta(days=settings.JWT_REFRESH_TTL_DAYS).total_seconds()
    )
