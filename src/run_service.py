import asyncio
import logging
import sys

import uvicorn
from dotenv import load_dotenv

from core import settings
from core.settings import DatabaseType

load_dotenv()

if __name__ == "__main__":
    root_logger = logging.getLogger()
    if root_logger.handlers:
        print(
            f"Warning: Root logger already has {len(root_logger.handlers)} handler(s) configured. "
            f"basicConfig() will be ignored. Current level: {logging.getLevelName(root_logger.level)}"
        )

    logging.basicConfig(level=settings.LOG_LEVEL.to_logging_level())

    # psycopg's async mode cannot run on ProactorEventLoop, and uvicorn >= 0.54 hard-codes
    # ProactorEventLoop on Windows through its loop_factory (it ignores the event loop
    # policy), so the server has to be built here to hand it a Selector loop. This path
    # gives up hot reload, which is a fair trade for a working database connection.
    needs_selector_loop = (
        sys.platform == "win32" and settings.DATABASE_TYPE == DatabaseType.POSTGRES
    )
    if needs_selector_loop:
        if settings.is_dev():
            print(
                "Warning: hot reload is disabled because PostgreSQL on Windows requires a Selector loop."
            )
        config = uvicorn.Config(
            "service:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=False,
            timeout_graceful_shutdown=settings.GRACEFUL_SHUTDOWN_TIMEOUT,
            forwarded_allow_ips=settings.FORWARDED_ALLOW_IPS,
        )
        asyncio.run(uvicorn.Server(config).serve(), loop_factory=asyncio.SelectorEventLoop)
    else:
        uvicorn.run(
            "service:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.is_dev(),
            timeout_graceful_shutdown=settings.GRACEFUL_SHUTDOWN_TIMEOUT,
            forwarded_allow_ips=settings.FORWARDED_ALLOW_IPS,
        )
