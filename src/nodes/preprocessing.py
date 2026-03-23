from ..state import GraphState
from ..utils import (
    detect_crypto_symbols, is_news_related_query, is_wallet_query,
    is_web_search_query, is_web_scrape_query, SYMBOL_MAP, logger
)

async def preprocessing_node(state: GraphState):
    """OPTIMIZATION: Heuristics-first intent detection, only use LLM for ambiguous cases."""
    messages = state.get("messages", [])
    user_input = messages[-1].content if messages else ""
    user_id = state.get("user_id")

    if user_id and not isinstance(user_id, str):
        user_id = str(user_id)

    if not user_input:
        return {"intent": "GENERAL_CHAT", "crypto_symbols": [], "error_message": "Empty input"}

    crypto_symbols = detect_crypto_symbols(user_input)
    is_news = is_news_related_query(user_input)
    is_wallet = is_wallet_query(user_input)
    is_web_search = is_web_search_query(user_input)
    is_scrape, scrape_urls = is_web_scrape_query(user_input)

    # OPTIMIZATION: Use heuristics to determine intent without LLM call
    # Priority: wallet > web_scrape(URL) > web_search(no crypto) > price_and_news > crypto_price > news > web_scrape(keyword) > web_search > general_chat
    if user_id and is_wallet:
        intent = "WALLET_QUERY"
    elif is_scrape and scrape_urls:
        intent = "WEB_SCRAPE_QUERY"
    elif is_web_search and not crypto_symbols:  # Web search takes priority if no crypto mentioned
        intent = "WEB_SEARCH_QUERY"
    elif crypto_symbols and is_news:
        intent = "PRICE_AND_NEWS_QUERY"
    elif crypto_symbols:
        intent = "CRYPTO_PRICE_QUERY"
    elif is_news and not crypto_symbols:
        # Distinguish between crypto news and general news
        if any(coin in user_input.lower() for coin in SYMBOL_MAP.keys()):
            intent = "NEWS_QUERY"
        else:
            # Check if it's general news or crypto news
            general_news_keywords = ['world', 'politics', 'government', 'stock', 'market', 'economy', 'tech', 'ai', 'science']
            if any(kw in user_input.lower() for kw in general_news_keywords):
                intent = "GENERAL_NEWS_QUERY"
            else:
                intent = "NEWS_QUERY"
    elif is_scrape:
        intent = "WEB_SCRAPE_QUERY"
    elif is_web_search:
        intent = "WEB_SEARCH_QUERY"
    else:
        intent = "GENERAL_CHAT"

    logger.info(f"Intent: {intent} | Symbols: {crypto_symbols} | Web Search: {is_web_search} | Scrape: {is_scrape} | User: {user_id}")
    return {
        "intent": intent,
        "crypto_symbols": crypto_symbols,
        "error_message": None,
        "user_id": user_id,
        "scraped_url": scrape_urls[0] if scrape_urls else None,
    }
