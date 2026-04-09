import asyncio
import json
from ..state import GraphState
from ..api.crypto import fetch_crypto_data
from ..api.news import fetch_news_enhanced
from ..utils import extract_query_fast

async def parallel_fetch_node(state: GraphState):
    """OPTIMIZATION: Fetch crypto and news data in parallel."""
    symbols = state.get("crypto_symbols", [])
    is_news = state.get("intent") == "PRICE_AND_NEWS_QUERY"
    
    tasks = []
    
    # Always fetch crypto data if symbols exist
    if symbols:
        tasks.append(fetch_crypto_data(symbols))
    else:
        tasks.append(None)
    
    # Fetch news if needed
    if is_news:
        query = extract_query_fast(state["messages"][-1].content)
        tasks.append(fetch_news_enhanced(query, is_general_news=False))
    else:
        tasks.append(None)
    
    # Run in parallel
    results = await asyncio.gather(*[t for t in tasks if t], return_exceptions=True)
    
    result_dict = {}
    result_idx = 0
    
    if symbols and result_idx < len(results):
        crypto_data = results[result_idx] if not isinstance(results[result_idx], Exception) else json.dumps({})
        result_dict["crypto_prices"] = crypto_data
        result_idx += 1
    
    if is_news and result_idx < len(results):
        news_result = results[result_idx]
        if not isinstance(news_result, Exception):
            news_content, news_success = news_result
            result_dict["news_results"] = news_content
            result_dict["news_available"] = news_success
        result_idx += 1
    
    return result_dict
