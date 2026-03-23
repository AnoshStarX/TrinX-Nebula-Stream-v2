import logging
from typing import Optional
import aiohttp

logger = logging.getLogger("trinx")

_session: Optional[aiohttp.ClientSession] = None


async def init_http_session() -> aiohttp.ClientSession:
    """Initialize a shared aiohttp session with pooled connections."""
    global _session
    if _session and not _session.closed:
        return _session

    connector = aiohttp.TCPConnector(
        limit=200,
        limit_per_host=50,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=15, connect=5, sock_connect=5, sock_read=10)
    _session = aiohttp.ClientSession(connector=connector, timeout=timeout)
    logger.info("Shared aiohttp session initialized")
    return _session


async def get_http_session() -> aiohttp.ClientSession:
    if _session is None or _session.closed:
        return await init_http_session()
    return _session


async def close_http_session():
    global _session
    if _session and not _session.closed:
        await _session.close()
        logger.info("Shared aiohttp session closed")
    _session = None
