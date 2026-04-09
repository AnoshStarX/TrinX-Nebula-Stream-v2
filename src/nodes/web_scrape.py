import asyncio
from ..state import GraphState
from ..api.scraper import scrape_url
from ..utils import extract_urls, logger
from ..config import SCRAPE_TIMEOUT_SECONDS


async def web_scrape_node(state: GraphState) -> dict:
    """Scrape a URL and return markdown content for the LLM."""
    url = state.get("scraped_url")

    # If preprocessing didn't set a URL, try extracting from the message
    if not url:
        messages = state.get("messages", [])
        user_input = messages[-1].content if messages else ""
        urls = extract_urls(user_input)
        url = urls[0] if urls else None

    if not url:
        logger.info("WEB_SCRAPE_NODE triggered but no URL found")
        return {
            "scraped_content": "No URL was provided. Please paste a link you'd like me to scrape or summarize.",
            "scraped_url": None,
        }

    logger.info(f"WEB_SCRAPE_NODE triggered | URL: {url}")
    try:
        content, success = await asyncio.wait_for(
            scrape_url(url),
            timeout=SCRAPE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Web scrape timed out for {url}")
        content = f"Scraping timed out for {url}. The page may be too large or slow to respond."
        success = False

    logger.info(f"Web scrape completed | Success: {success} | URL: {url} | Content length: {len(content)}")
    return {
        "scraped_content": content,
        "scraped_url": url,
    }
