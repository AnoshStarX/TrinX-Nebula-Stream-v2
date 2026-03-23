from ..state import GraphState
from ..api.tavily import fetch_tavily_search
from ..utils import logger

async def web_search_node(state: GraphState):
    """OPTIMIZATION: Use fast query extraction instead of LLM."""
    messages = state.get("messages", [])
    user_input = messages[-1].content
    
    # OPTIMIZATION: Fast extraction without LLM
    # For web search, use the full user input as query since it's more contextual
    extracted_query = user_input.strip()
    
    logger.info(f"WEB_SEARCH_NODE triggered | Query: '{extracted_query}'")
    search_results, search_success = await fetch_tavily_search(extracted_query, max_results=5)
    
    logger.info(f"Web search completed | Success: {search_success} | Results length: {len(search_results)}")
    
    return {
        "news_query": extracted_query,
        "news_results": search_results,
        "news_available": search_success
    }
