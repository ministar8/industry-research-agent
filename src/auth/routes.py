from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from core.settings import settings
from schema import AuthResponse, LoginInput, RegisterInput, UserResponse

from .security import decode_token, issue_access_token, issue_refresh_token
from .store import AuthStore, AuthUser

router = APIRouter(prefix="/auth", tags=["auth"])


def _check_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
        raise HTTPException(status_code=403, detail="请求来源不被允许。")


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
    try:
        user = await store.authenticate(payload.username, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误。")
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
    if token:
        try:
            claims = decode_token(token, "refresh")
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
