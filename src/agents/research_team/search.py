"""DeepSeek Search and Fetch adapters for the researcher sub-agent.

Search discovers pages, Fetch reads one page as Markdown. Both are billed per
successful request (HTTP 200 with a non-empty payload), so failed calls are free
to retry.
"""

import asyncio
import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from core import settings
from core.metrics import record_fetch, record_fetch_skipped, record_search

logger = logging.getLogger(__name__)

BASE_URL = "https://api.deepseek.com/v1/tools"
MAX_ATTEMPTS = 3
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_QUERY_CHARS = 128
MAX_URL_CHARS = 8192
DEFAULT_COUNT = 10


class SearchUnavailable(RuntimeError):
    """Raised when the Search/Fetch API keeps failing after retries."""


class SearchRejected(RuntimeError):
    """Raised when the API refuses the request outright.

    Balance exhausted, a bad key or a malformed query all land here. These are not
    transient: retrying cannot help, and treating them as "no results" hides the real
    fault behind an empty source list.
    """


class SearchResponseError(RuntimeError):
    """Raised when a successful response violates the Search API shape."""


def _headers() -> dict[str, str]:
    if not settings.DEEPSEEK_API_KEY:
        raise SearchUnavailable("DEEPSEEK_API_KEY is not configured")
    return {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY.get_secret_value()}",
        "Content-Type": "application/json",
    }


def is_fetchable(url: str) -> bool:
    if not url or len(url) > MAX_URL_CHARS:
        return False
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname) and "@" not in parsed.netloc


_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None


async def _close_quietly(client: httpx.AsyncClient) -> None:
    """Close a pool without letting a dead loop turn cleanup into a failure."""
    if client.is_closed:
        return
    try:
        await client.aclose()
    except Exception:
        # A loop that is gone cannot clean up after itself; dropping the pool silently
        # is still better than failing the request that triggered the switch.
        logger.debug("Closing an HTTP client failed", exc_info=True)


async def _shared_client(timeout: float) -> httpx.AsyncClient:
    """Reuse one connection pool per event loop.

    A round issues four searches plus a handful of fetches; a fresh client per call
    pays a TLS handshake every time. The pool belongs to the loop that created it, so a
    different loop gets a fresh client - and the stale one is closed rather than
    dropped, otherwise every loop switch leaks its sockets.
    """
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is not None and _client_loop is not loop:
        await _close_quietly(_client)
        _client = None
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=timeout)
        _client_loop = loop
    return _client


async def close_shared_client() -> None:
    """Release the pool on shutdown; nothing else closes it before the process exits."""
    global _client
    client, _client = _client, None
    if client is not None:
        await _close_quietly(client)


async def _post(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    headers = _headers()
    last_error = "unknown"
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = await (await _shared_client(timeout)).post(
                f"{BASE_URL}/{path}", headers=headers, json=payload, timeout=timeout
            )
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200:
                return response.json()
            if response.status_code not in RETRY_STATUS:
                # 4xx means the request itself is malformed; retrying cannot help.
                logger.error(
                    "DeepSeek %s rejected (%s): %s",
                    path,
                    response.status_code,
                    response.text[:200],
                )
                raise SearchRejected(f"HTTP {response.status_code}: {response.text[:160]}")
            last_error = f"HTTP {response.status_code}"
        if attempt < MAX_ATTEMPTS - 1:
            await asyncio.sleep(1.5 * (attempt + 1))
    raise SearchUnavailable(f"DeepSeek {path} failed after {MAX_ATTEMPTS} attempts: {last_error}")


async def search(
    query: str, count: int = DEFAULT_COUNT, ori_question: str | None = None
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {
        "query": query.strip()[:MAX_QUERY_CHARS],
        "count": max(1, min(count, 50)),
        "need_content": False,
    }
    if ori_question:
        payload["ori_question"] = ori_question.strip()[:MAX_QUERY_CHARS]

    data = await _post("search", payload, timeout=60.0)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise SearchResponseError("DeepSeek search returned an invalid results payload")
    results = data["results"]
    # Counts the request, not the hit rate: an empty page still consumed a call.
    record_search(ok=True)
    return results


async def fetch(url: str) -> dict[str, str] | None:
    if not is_fetchable(url):
        logger.warning("Skipping unfetchable url: %s", url[:120])
        record_fetch_skipped()
        return None

    data = await _post("fetch", {"url": url}, timeout=60.0)
    if not isinstance(data, dict):
        raise SearchResponseError("DeepSeek fetch returned an invalid response payload")
    content = str(data.get("content") or "")
    if not content:
        record_fetch(ok=False)
        return None
    record_fetch(ok=True)
    return {"title": str(data.get("title") or ""), "content": content}
