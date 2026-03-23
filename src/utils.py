import re
import json
import logging
from hashlib import md5
from typing import Optional, List, Any
try:
    import redis.asyncio as redis
except Exception:  # pragma: no cover - optional dependency fallback
    redis = None
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from .config import (
    SYMBOL_MAP,
    REDIS_URL,
    CACHE_TTL_SECONDS,
    SESSION_TTL_SECONDS,
    SESSION_HISTORY_LIMIT,
)

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("trinx")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# --- Redis-backed cache and session state ---
_redis_client: Optional[Any] = None
_local_response_cache: dict[str, str] = {}
_local_session_store: dict[str, list[dict[str, str]]] = {}
_CACHE_PREFIX = "cache:resp:"
_SESSION_PREFIX = "session:messages:"


async def init_redis() -> bool:
    """Initialize Redis client for shared cache and session history."""
    global _redis_client
    if _redis_client:
        return True
    if redis is None:
        logger.warning("redis package is not installed. Falling back to in-process storage.")
        return False
    if not REDIS_URL:
        logger.warning("REDIS_URL is not set. Falling back to in-process storage.")
        return False

    try:
        _redis_client = redis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            health_check_interval=30,
        )
        await _redis_client.ping()
        logger.info("Redis connected for shared state/cache")
        return True
    except Exception as e:
        logger.warning(f"Redis initialization failed, using in-process fallback: {e}")
        _redis_client = None
        return False


async def close_redis():
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


def _serialize_message(msg: AnyMessage) -> dict[str, str]:
    if isinstance(msg, HumanMessage):
        return {"role": "human", "content": msg.content}
    return {"role": "ai", "content": msg.content}


def _deserialize_message(item: dict[str, str]) -> AnyMessage:
    role = item.get("role")
    content = item.get("content", "")
    if role == "human":
        return HumanMessage(content=content)
    return AIMessage(content=content)

def get_cache_key(query: str, intent: str, scope: Optional[str] = None) -> str:
    """Generate cache key from query, intent, and optional scope."""
    raw = f"{query.lower()}:{intent}:{(scope or 'global').lower()}"
    return md5(raw.encode()).hexdigest()

async def get_cached_response(key: str) -> Optional[str]:
    """Retrieve response from Redis cache."""
    if _redis_client:
        try:
            cached = await _redis_client.get(f"{_CACHE_PREFIX}{key}")
            if cached:
                logger.info(f"Cache hit for key: {key}")
            return cached
        except Exception as e:
            logger.warning(f"Redis get cache failed: {e}")
    return _local_response_cache.get(key)

async def set_cache(key: str, response: str):
    """Store response in Redis cache."""
    if _redis_client:
        try:
            await _redis_client.setex(f"{_CACHE_PREFIX}{key}", CACHE_TTL_SECONDS, response)
            return
        except Exception as e:
            logger.warning(f"Redis set cache failed: {e}")
    _local_response_cache[key] = response


async def get_cache_size() -> int:
    """Return cache key count for health reporting."""
    if _redis_client:
        try:
            count = 0
            async for _ in _redis_client.scan_iter(match=f"{_CACHE_PREFIX}*"):
                count += 1
            return count
        except Exception as e:
            logger.warning(f"Redis cache size scan failed: {e}")
    return len(_local_response_cache)


async def get_session_messages(session_id: str) -> list[AnyMessage]:
    """Load chat history from Redis."""
    key = f"{_SESSION_PREFIX}{session_id}"
    if _redis_client:
        try:
            raw = await _redis_client.get(key)
            if not raw:
                return []
            items = json.loads(raw)
            if not isinstance(items, list):
                return []
            return [_deserialize_message(item) for item in items]
        except Exception as e:
            logger.warning(f"Redis get session failed: {e}")

    items = _local_session_store.get(session_id, [])
    return [_deserialize_message(item) for item in items]


async def append_session_turn(session_id: str, user_prompt: str, ai_response: str):
    """Persist one chat turn to shared session history."""
    history = await get_session_messages(session_id)
    history.append(HumanMessage(content=user_prompt))
    history.append(AIMessage(content=ai_response))
    history = history[-SESSION_HISTORY_LIMIT:]

    payload = [_serialize_message(msg) for msg in history]
    key = f"{_SESSION_PREFIX}{session_id}"
    if _redis_client:
        try:
            await _redis_client.setex(key, SESSION_TTL_SECONDS, json.dumps(payload, ensure_ascii=False))
            return
        except Exception as e:
            logger.warning(f"Redis save session failed: {e}")

    _local_session_store[session_id] = payload

# --- Helper Functions ---
def detect_crypto_symbols(text: str) -> List[str]:
    """Detect crypto symbols from text."""
    t = text.lower()
    found = set()
    for k, v in SYMBOL_MAP.items():
        if re.search(rf'\b{re.escape(k)}\b', t):
            found.add(v)
    return list(found)

def is_news_related_query(text: str) -> bool:
    """Check if query is news-related using keywords."""
    news_keywords = [
        'news', 'latest', 'update', 'updates', 'recent', 'announce', 'announced',
        'announcement', 'development', 'developments', 'happening', 'trend',
        'trends', 'event', 'events', 'report', 'reports', 'breaking', 'what happened'
    ]
    t = text.lower()
    return any(keyword in t for keyword in news_keywords)

def is_wallet_query(text: str) -> bool:
    """Check if query is wallet/account related."""
    keywords = ['balance', 'wallet', 'coins', 't coin', 'tix coin', 'tap', 'taps', 'premium', 'account', 'my', 'how much']
    t = text.lower()
    return any(keyword in t for keyword in keywords)

def is_web_search_query(text: str) -> bool:
    """Check if query requires web search (real-time info)."""
    web_keywords = [
        'search', 'google', 'current', 'now', 'today', 'right now', 'happening', 
        'trending', 'real-time', 'latest', 'recent', 'breaking', 'news', 'what is',
        'who is', 'where is', 'when did', 'check', 'find', 'lookup', 'look up',
        'research', 'find out', 'tell me about', 'info on', 'info about',
        'information on', 'information about', 'details on', 'details about',
        'election', 'weather', 'stock', 'market', 'price', 'rate', 'covid',
        'pandemic', 'virus', 'disease', 'tech', 'technology', 'ai', 'update'
    ]
    t = text.lower()
    return any(keyword in t for keyword in web_keywords)

def extract_urls(text: str) -> list[str]:
    """Extract URLs from text."""
    return re.findall(r'https?://[^\s<>"\')\]]+', text)


def extract_youtube_video_id(url: str) -> str | None:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def is_web_scrape_query(text: str) -> tuple[bool, list[str]]:
    """Check if query is a web scraping request. Returns (is_scrape, urls)."""
    urls = extract_urls(text)
    if urls:
        return (True, urls)
    scrape_keywords = [
        'scrape', 'crawl', 'summarize this page', 'summarize this website',
        'summarize this article', 'read this page', 'read this website',
        'read this article', 'extract from',
    ]
    t = text.lower()
    if any(kw in t for kw in scrape_keywords):
        return (True, [])
    return (False, [])


def extract_query_fast(text: str) -> str:
    """OPTIMIZATION: Fast query extraction without LLM."""
    # Try to find first crypto symbol
    for word in text.lower().split():
        clean_word = word.strip('.,!?;:')
        if clean_word in SYMBOL_MAP:
            return SYMBOL_MAP[clean_word]
    # Return original text if no symbol found
    return text

async def _safe_json(resp):
    """Safely parse JSON from response."""
    try:
        return await resp.json()
    except Exception:
        text = await resp.text()
        try:
            return json.loads(text or "{}")
        except Exception:
            logger.warning(f"Could not parse JSON from response")
            return {"_raw": text or ""}
