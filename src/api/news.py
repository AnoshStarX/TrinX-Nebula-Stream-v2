import asyncio
from ..config import (
    CRYPTOPANIC_API_KEY, CRYPTOPANIC_API_URL,
    CRYPTOCOMPARE_API_KEY, CRYPTOCOMPARE_API_URL,
    GNEWS_API_KEY, GNEWS_API_URL
)
from ..utils import logger, _safe_json
from ..http_client import get_http_session


async def _fetch_cryptopanic(session, query, limit):
    """Fetch crypto news from CryptoPanic."""
    if not CRYPTOPANIC_API_KEY:
        return None
    try:
        params = {"auth_token": CRYPTOPANIC_API_KEY, "filter": "hot", "currencies": query, "limit": limit}
        async with session.get(CRYPTOPANIC_API_URL, params=params, timeout=8) as resp:
            if resp.status == 200:
                articles = (await _safe_json(resp)).get("results", [])
                if articles:
                    formatted = [f"• {a.get('title', 'N/A')} (Source: {a.get('source', {}).get('title', 'N/A')})" for a in articles[:limit]]
                    return "Latest Crypto News:\n" + "\n".join(formatted)
    except Exception as e:
        logger.warning(f"CryptoPanic failed: {e}")
    return None


async def _fetch_cryptocompare(session, query, limit):
    """Fetch crypto news from CryptoCompare."""
    if not CRYPTOCOMPARE_API_KEY:
        return None
    try:
        params = {"lang": "EN", "api_key": CRYPTOCOMPARE_API_KEY, "categories": query.upper(), "limit": limit}
        async with session.get(CRYPTOCOMPARE_API_URL, params=params, timeout=8) as resp:
            if resp.status == 200:
                articles = (await _safe_json(resp)).get("Data", [])
                if articles:
                    formatted = [f"• {a.get('title', 'N/A')} (Source: {a.get('source', 'N/A')})" for a in articles[:limit]]
                    return "Latest Crypto News:\n" + "\n".join(formatted)
    except Exception as e:
        logger.warning(f"CryptoCompare failed: {e}")
    return None


async def _fetch_gnews(session, query, limit, is_general_news):
    """Fetch news from GNews."""
    if not GNEWS_API_KEY:
        return None
    try:
        q = query if is_general_news or 'crypto' in query.lower() else f"{query} cryptocurrency"
        params = {"q": q, "lang": "en", "max": limit, "token": GNEWS_API_KEY}
        async with session.get(GNEWS_API_URL, params=params, timeout=8) as resp:
            if resp.status == 200:
                articles = (await _safe_json(resp)).get("articles", [])
                if articles:
                    formatted = [f"• {a.get('title', 'N/A')} (Source: {a.get('source', {}).get('name', 'N/A')})" for a in articles[:limit]]
                    return "Latest News:\n" + "\n".join(formatted)
    except Exception as e:
        logger.warning(f"GNews failed: {e}")
    return None


async def fetch_news_enhanced(query: str, limit: int = 5, is_general_news: bool = False) -> tuple[str, bool]:
    """Fetch news from multiple sources in parallel, return first successful result.

    Parallel fetching reduces latency vs the previous sequential fallback approach.
    Priority order is preserved: CryptoPanic > CryptoCompare > GNews.
    """
    if not any([CRYPTOPANIC_API_KEY, CRYPTOCOMPARE_API_KEY, GNEWS_API_KEY]):
        return "News APIs not configured.", False

    session = await get_http_session()
    if is_general_news:
        result = await _fetch_gnews(session, query, limit, is_general_news=True)
        if result:
            return result, True
    else:
        # Fire all crypto news sources in parallel
        results = await asyncio.gather(
            _fetch_cryptopanic(session, query, limit),
            _fetch_cryptocompare(session, query, limit),
            _fetch_gnews(session, query, limit, is_general_news=False),
            return_exceptions=True,
        )

        # Return first successful result (preserves priority order)
        for result in results:
            if isinstance(result, str):
                return result, True

    return f"No news available for '{query}'.", False
