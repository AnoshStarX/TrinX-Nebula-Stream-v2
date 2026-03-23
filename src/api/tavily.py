import json
import asyncio
from ..config import TAVILY_API_KEY, TAVILY_API_URL
from ..utils import logger
from ..http_client import get_http_session

async def fetch_tavily_search(query: str, max_results: int = 5) -> tuple[str, bool]:
    """OPTIMIZATION: Reduced timeout to 10s."""
    if not TAVILY_API_KEY:
        logger.warning("Tavily API key not configured")
        return "Web search is not available. Tavily API key is not configured.", False
    
    logger.info(f"Tavily API call initiated for query: '{query}'")
    
    try:
        session = await get_http_session()
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False,
            "include_domains": [],
            "exclude_domains": []
        }
        
        logger.info(f"Tavily payload: {json.dumps({k: v for k, v in payload.items() if k != 'api_key'})}")
        
        async with session.post(TAVILY_API_URL, json=payload, timeout=10) as resp:
            logger.info(f"Tavily response status: {resp.status}")
            
            if resp.status == 200:
                data = await resp.json()
                logger.info(f"Tavily response data keys: {data.keys()}")
                results = []
                
                if data.get("answer"):
                    results.append(f"Quick Answer: {data['answer']}\n")
                
                if data.get("results"):
                    results.append("Search Results:")
                    for idx, result in enumerate(data["results"][:max_results], 1):
                        title = result.get("title", "N/A")
                        url = result.get("url", "")
                        content = result.get("content", "")
                        
                        if len(content) > 200:
                            content = content[:200] + "..."
                        
                        results.append(f"{idx}. {title}")
                        if content:
                            results.append(f"   {content}")
                        results.append(f"   Source: {url}\n")
                
                if results:
                    formatted_output = "\n".join(results)
                    logger.info(f"Tavily search successful for: {query}")
                    return formatted_output, True
                else:
                    logger.warning(f"Tavily returned no results for: {query}")
                    return f"No results found for '{query}'.", False
            elif resp.status == 429:
                logger.warning("Tavily rate limit exceeded")
                return "Web search rate limit exceeded. Please try again later.", False
            else:
                error_text = await resp.text()
                logger.error(f"Tavily API error {resp.status}: {error_text}")
                return f"Web search encountered an error (status {resp.status}).", False
                    
    except asyncio.TimeoutError:
        logger.error(f"Tavily API timeout for query: {query}")
        return "Web search request timed out. Please try again.", False
    except Exception as e:
        logger.error(f"Tavily API exception for query '{query}': {e}", exc_info=True)
        return f"Web search failed: {str(e)}", False
