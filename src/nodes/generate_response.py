import json
from hashlib import md5
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from ..state import GraphState
from ..api.groq import llm
from ..config import GENERATE_RESPONSE_PROMPT, SYSTEM_PROMPT, LONG_TERM_MEMORY_MAX_ITEMS
from ..utils import get_cache_key, get_cached_response, set_cache, logger

try:
    import httpx
except Exception:  # pragma: no cover - optional dependency shape guard
    httpx = None

try:
    import httpcore
except Exception:  # pragma: no cover - optional dependency shape guard
    httpcore = None


def _extract_token_usage(response) -> dict:
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "from_cache": False,
        "source": "unknown",
    }
    usage_meta = getattr(response, "usage_metadata", None) or {}
    if usage_meta:
        in_t = int(usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens") or 0)
        out_t = int(usage_meta.get("output_tokens") or usage_meta.get("completion_tokens") or 0)
        tot_t = int(usage_meta.get("total_tokens") or (in_t + out_t))
        usage.update(
            {
                "input_tokens": in_t,
                "output_tokens": out_t,
                "total_tokens": tot_t,
                "source": "usage_metadata",
            }
        )
        return usage

    response_meta = getattr(response, "response_metadata", None) or {}
    token_usage = response_meta.get("token_usage") or response_meta.get("usage") or {}
    if token_usage:
        in_t = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
        out_t = int(token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0)
        tot_t = int(token_usage.get("total_tokens") or (in_t + out_t))
        usage.update(
            {
                "input_tokens": in_t,
                "output_tokens": out_t,
                "total_tokens": tot_t,
                "source": "response_metadata",
            }
        )
    return usage


def _is_timeout_error(exc: Exception) -> bool:
    timeout_types = []
    if httpx is not None:
        timeout_types.append(httpx.TimeoutException)
    if httpcore is not None:
        timeout_types.append(httpcore.TimeoutException)
    typed_timeout = tuple(timeout_types)

    seen: set[int] = set()
    current: Exception | None = exc
    while current and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError):
            return True
        if typed_timeout and isinstance(current, typed_timeout):
            return True
        if "timeout" in current.__class__.__name__.lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def _shorter_retry_messages(llm_messages: list) -> list:
    if len(llm_messages) <= 9:
        return llm_messages
    # Keep system prompt plus most recent turns for a single retry.
    return [llm_messages[0], *llm_messages[-8:]]


async def generate_response_node(state: GraphState):
    """Generate final response using LLM via ChatOpenAI (auto-traced by LangSmith).

    Cache is checked first. History window is bounded for latency.
    """
    messages = state["messages"]
    user_input = messages[-1].content
    intent = state.get("intent", "GENERAL_CHAT")
    user_id = str(state.get("user_id") or "anon")
    long_term_memories = state.get("long_term_memories") or []
    session_summary = state.get("session_summary")

    # Avoid stale cache on volatile intents (live/news/price lookups).
    volatile_intents = {
        "NEWS_QUERY",
        "GENERAL_NEWS_QUERY",
        "CRYPTO_PRICE_QUERY",
        "PRICE_AND_NEWS_QUERY",
        "WEB_SEARCH_QUERY",
        "WEB_SCRAPE_QUERY",
    }
    cache_allowed = intent not in volatile_intents
    memory_sig = ""
    if long_term_memories:
        memory_sig = md5("|".join(long_term_memories).encode()).hexdigest()[:10]
    cache_scope = f"user:{user_id}:mem:{memory_sig or 'none'}"

    # Check cache
    cache_key = get_cache_key(user_input, intent, scope=cache_scope)
    if cache_allowed:
        cached = await get_cached_response(cache_key)
        if cached:
            return {
                "messages": [AIMessage(content=cached)],
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "from_cache": True,
                    "source": "cache",
                },
            }

    crypto_prices = state.get("crypto_prices", "Not requested or available.")
    news_results = state.get("news_results", "Not requested or available.")
    scraped_content = state.get("scraped_content", "Not requested or available.")
    scraped_url = state.get("scraped_url") or ""
    user_data = state.get("user_data")

    user_wallet_data = "Not requested or available."
    if user_data and not user_data.get("error"):
        wallet_info = {
            "user_has_premium": user_data.get("is_premium", False),
            "words_typed": user_data.get("words_typed", 0),
            "tix": user_data.get("tix", 0),
        }
        user_wallet_data = json.dumps(wallet_info, indent=2)

    prompt_with_context = GENERATE_RESPONSE_PROMPT.format(
        user_question=user_input,
        user_wallet_data=user_wallet_data,
        crypto_data=crypto_prices,
        news_content=news_results,
        scraped_content=scraped_content,
        scraped_url=scraped_url,
    )
    if long_term_memories:
        memory_block = "\n".join(f"- {m}" for m in long_term_memories[:LONG_TERM_MEMORY_MAX_ITEMS])
        prompt_with_context += f"\n\nRelevant User Memory:\n{memory_block}"
    if session_summary:
        prompt_with_context += f"\n\nSession Summary:\n{session_summary}"

    # Build LangChain message list
    llm_messages = [SystemMessage(content=SYSTEM_PROMPT)]
    history_window = messages[-14:]  # Better continuity while staying bounded.
    for msg in history_window:
        if msg == messages[-1]:
            llm_messages.append(HumanMessage(content=prompt_with_context))
        elif isinstance(msg, HumanMessage):
            llm_messages.append(msg)
        elif isinstance(msg, AIMessage):
            llm_messages.append(msg)

    try:
        if not llm:
            raise RuntimeError("LLM not configured - check GROQ_API_KEYS")

        try:
            response = await llm.ainvoke(llm_messages)
        except Exception as first_error:
            if not _is_timeout_error(first_error):
                raise
            logger.warning("LLM request timed out; retrying once with shorter context.")
            response = await llm.ainvoke(_shorter_retry_messages(llm_messages))
        response_content = response.content.strip()

        if not response_content:
            return {
                "messages": [AIMessage(content="I'm having trouble formulating a response. Please try again.")],
                "token_usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "from_cache": False,
                    "source": "empty_response",
                },
            }

        if cache_allowed:
            await set_cache(cache_key, response_content)
        token_usage = _extract_token_usage(response)
        return {"messages": [AIMessage(content=response_content)], "token_usage": token_usage}
    except Exception as e:
        logger.error(f"Response generation failed: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content="I'm experiencing a technical issue. Please try again later.")],
            "token_usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "from_cache": False,
                "source": "error",
            },
        }
