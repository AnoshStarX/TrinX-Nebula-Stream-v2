from ..state import GraphState
from ..api.news import fetch_news_enhanced
from ..utils import extract_query_fast, logger

async def news_search_node(state: GraphState):
    """OPTIMIZATION: Use fast query extraction instead of LLM."""
    messages = state.get("messages", [])
    user_input = messages[-1].content
    intent = state.get("intent", "")
    
    # OPTIMIZATION: Fast extraction without LLM
    extracted_query = extract_query_fast(user_input)
    
    is_general = intent == "GENERAL_NEWS_QUERY"
    logger.info(f"News search for: '{extracted_query}' (General: {is_general})")
    
    news_content, news_success = await fetch_news_enhanced(extracted_query, is_general_news=is_general)
    return {"news_query": extracted_query, "news_results": news_content, "news_available": news_success}
