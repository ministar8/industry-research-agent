import base64
import hashlib
import hmac
import re
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import jwt
from pydantic import SecretStr

from core.settings import settings

PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,32}$")
_SCRYPT_N = 16_384
_SCRYPT_R = 8
_SCRYPT_P = 1


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def normalize_username(username: str) -> str:
    value = username.strip().lower()
    if not USERNAME_PATTERN.fullmatch(value):
        raise ValueError("用户名需为 3-32 位字母、数字或下划线。")
    return value


def validate_password(password: str) -> str:
    if not PASSWORD_MIN_LENGTH <= len(password) <= PASSWORD_MAX_LENGTH:
        raise ValueError(f"密码长度需为 {PASSWORD_MIN_LENGTH}-{PASSWORD_MAX_LENGTH} 位。")
    return password


def hash_password(password: str) -> str:
    validate_password(password)
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_encode(salt)}${_encode(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, n, r, p, salt_value, digest_value = encoded.split("$")
        if algorithm != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_decode(salt_value),
            n=int(n),
            r=int(r),
            p=int(p),
        )
        return hmac.compare_digest(digest, _decode(digest_value))
    except (ValueError, TypeError):
        return False


def _secret() -> str:
    if not isinstance(settings.JWT_SECRET, SecretStr):
        raise RuntimeError("JWT_SECRET is not configured.")
    return settings.JWT_SECRET.get_secret_value()


def jwt_enabled() -> bool:
    return isinstance(settings.JWT_SECRET, SecretStr) and bool(
        settings.JWT_SECRET.get_secret_value()
    )


def issue_token(
    user_id: UUID, username: str, token_type: str, lifetime: timedelta, version: int
) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "type": token_type,
        "version": version,
        "iat": now,
        "exp": now + lifetime,
    }
    return jwt.encode(payload, _secret(), algorithm="HS256")


def issue_access_token(user_id: UUID, username: str, version: int) -> str:
    return issue_token(
        user_id,
        username,
        "access",
        timedelta(minutes=settings.JWT_ACCESS_TTL_MINUTES),
        version,
    )


def issue_refresh_token(user_id: UUID, username: str, version: int) -> str:
    return issue_token(
        user_id,
        username,
        "refresh",
        timedelta(days=settings.JWT_REFRESH_TTL_DAYS),
        version,
    )


def decode_token(token: str, expected_type: str) -> dict[str, Any]:
    try:
        claims = jwt.decode(token, _secret(), algorithms=["HS256"])
    except (jwt.InvalidTokenError, RuntimeError) as exc:
        raise ValueError("令牌无效或已过期。") from exc
    if claims.get("type") != expected_type or not claims.get("sub"):
        raise ValueError("令牌类型无效。")
    return claims
