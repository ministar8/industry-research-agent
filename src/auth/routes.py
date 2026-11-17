import time
from collections import deque
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from core.settings import settings
from schema import AuthResponse, LoginInput, RegisterInput, UserResponse

from .security import decode_token, issue_access_token, issue_refresh_token
from .store import AuthStore, AuthUser

router = APIRouter(prefix="/auth", tags=["auth"])

_LOGIN_WINDOW_SECONDS = 60
_LOGIN_FAILURE_LIMIT = 5
_MAX_LOGIN_KEYS = 10_000
_login_failures: dict[str, deque[float]] = {}


def _check_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
        raise HTTPException(status_code=403, detail="请求来源不被允许。")


def _login_key(request: Request, username: str) -> str:
    client_host = request.client.host if request.client else "unknown"
    return f"{client_host}:{username.strip().lower()}"


def _check_login_limit(key: str) -> None:
    now = time.monotonic()
    failures = _login_failures.get(key)
    if failures is None:
        return
    while failures and now - failures[0] >= _LOGIN_WINDOW_SECONDS:
        failures.popleft()
    if not failures:
        _login_failures.pop(key, None)
        return
    if len(failures) >= _LOGIN_FAILURE_LIMIT:
        retry_after = max(1, int(_LOGIN_WINDOW_SECONDS - (now - failures[0])))
        raise HTTPException(
            status_code=429,
            detail="登录尝试过于频繁，请稍后再试。",
            headers={"Retry-After": str(retry_after)},
        )


def _record_login_failure(key: str) -> None:
    if key not in _login_failures and len(_login_failures) >= _MAX_LOGIN_KEYS:
        oldest_key = min(_login_failures, key=lambda item: _login_failures[item][-1])
        _login_failures.pop(oldest_key, None)
    _login_failures.setdefault(key, deque()).append(time.monotonic())


def _clear_login_failures(key: str) -> None:
    _login_failures.pop(key, None)


def get_auth_store(request: Request) -> AuthStore:
    store = getattr(request.app.state, "auth_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="用户认证服务未配置。")
    return store


def _set_auth_cookies(response: Response, user: AuthUser) -> None:
    response.set_cookie(
        "access_token",
        issue_access_token(user.id, user.username, user.token_version),
        max_age=settings.JWT_ACCESS_TTL_MINUTES * 60,
        httponly=True,
        secure=settings.AUTH_COOKIE_SECURE,
        samesite="lax",
    )
    response.set_cookie(
        "refresh_token",
        issue_refresh_token(user.id, user.username, user.token_version),
        max_age=settings.JWT_REFRESH_TTL_DAYS * 86400,
        httponly=True,
        secure=settings.AUTH_COOKIE_SECURE,
        samesite="lax",
    )


def _user_response(user: AuthUser) -> UserResponse:
    return UserResponse(id=str(user.id), username=user.username, created_at=user.created_at)


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterInput,
    request: Request,
    response: Response,
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> AuthResponse:
    _check_origin(request)
    try:
        user = await store.create_user(payload.username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _set_auth_cookies(response, user)
    return AuthResponse(user=_user_response(user))


@router.post("/login", response_model=AuthResponse)
async def login(
    payload: LoginInput,
    request: Request,
    response: Response,
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> AuthResponse:
    _check_origin(request)
    login_key = _login_key(request, payload.username)
    _check_login_limit(login_key)
    try:
        user = await store.authenticate(payload.username, payload.password)
    except ValueError as exc:
        _record_login_failure(login_key)
        raise HTTPException(status_code=401, detail="用户名或密码错误。") from exc
    if user is None:
        _record_login_failure(login_key)
        raise HTTPException(status_code=401, detail="用户名或密码错误。")
    _clear_login_failures(login_key)
    _set_auth_cookies(response, user)
    return AuthResponse(user=_user_response(user))


@router.post("/refresh", response_model=AuthResponse)
async def refresh(
    request: Request,
    response: Response,
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> AuthResponse:
    _check_origin(request)
    token = request.cookies.get("refresh_token")
    if not token:
        raise HTTPException(status_code=401, detail="缺少刷新令牌。")
    try:
        claims = decode_token(token, "refresh")
        user = await store.get_user(UUID(claims["sub"]))
        if user is None or user.token_version != claims.get("version"):
            raise ValueError("令牌已撤销。")
        rotated_user = await store.rotate_token_version(user.id, user.token_version)
        if rotated_user is None:
            raise ValueError("令牌已撤销。")
        user = rotated_user
    except (ValueError, KeyError):
        raise HTTPException(status_code=401, detail="刷新令牌无效或已过期。")
    _set_auth_cookies(response, user)
    return AuthResponse(user=_user_response(user))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> None:
    _check_origin(request)
    token = request.cookies.get("refresh_token")
    revoked = False
    if token:
        try:
            claims = decode_token(token, "refresh")
            await store.revoke_refresh_tokens(UUID(claims["sub"]))
            revoked = True
        except (ValueError, KeyError):
            pass
    if not revoked:
        access_token = request.cookies.get("access_token")
        if access_token:
            try:
                claims = decode_token(access_token, "access")
                await store.revoke_refresh_tokens(UUID(claims["sub"]))
            except (ValueError, KeyError):
                pass
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")


@router.get("/me", response_model=UserResponse)
async def me(
    request: Request,
    store: Annotated[AuthStore, Depends(get_auth_store)],
) -> UserResponse:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="未登录。")
    try:
        claims = decode_token(token, "access")
        user = await store.get_user(UUID(claims["sub"]))
        if user is None or user.token_version != claims.get("version"):
            raise ValueError("令牌已撤销。")
    except (ValueError, KeyError):
        raise HTTPException(status_code=401, detail="登录状态无效或已过期。")
    return _user_response(user)
