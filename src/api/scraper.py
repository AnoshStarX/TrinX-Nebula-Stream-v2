import asyncio
import logging
import aiohttp
from ..config import SCRAPE_MAX_CONTENT_CHARS
from ..http_client import get_http_session

logger = logging.getLogger("trinx")


def _truncate_content(text: str, max_chars: int = SCRAPE_MAX_CONTENT_CHARS) -> str:
    """Truncate content at a paragraph boundary to fit LLM context."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to cut at last paragraph break
    last_break = truncated.rfind("\n\n")
    if last_break > max_chars // 2:
        truncated = truncated[:last_break]
    return truncated.rstrip() + "\n\n[Content truncated]"


async def _fetch_youtube_transcript(video_id: str) -> tuple[str, bool]:
    """Fetch YouTube transcript using youtube-transcript-api (sync, run in executor)."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        loop = asyncio.get_running_loop()

        def _get_transcript():
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)
            return " ".join(snippet.text for snippet in transcript)

        text = await asyncio.wait_for(
            loop.run_in_executor(None, _get_transcript),
            timeout=8,
        )
        if text:
            return _truncate_content(f"**YouTube Transcript:**\n\n{text}"), True
        return "Could not retrieve transcript for this video.", False
    except asyncio.TimeoutError:
        logger.warning(f"YouTube transcript fetch timed out for {video_id}")
        return "YouTube transcript fetch timed out.", False
    except Exception as e:
        logger.warning(f"YouTube transcript fetch failed for {video_id}: {e}")
        return f"Could not retrieve YouTube transcript: {e}", False


async def _fetch_with_crawl4ai(url: str) -> tuple[str, bool]:
    """Fetch page content using crawl4ai (async Playwright)."""
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await asyncio.wait_for(
                crawler.arun(url=url),
                timeout=12,
            )
            markdown = result.markdown if hasattr(result, "markdown") else ""
            if markdown:
                return _truncate_content(markdown), True
            return "Page was fetched but no content could be extracted.", False
    except asyncio.TimeoutError:
        logger.warning(f"crawl4ai fetch timed out for {url}")
        return "", False  # Empty string signals fallback to Jina
    except Exception as e:
        logger.warning(f"crawl4ai fetch failed for {url}: {e}")
        return "", False


async def _fetch_with_jina_reader(url: str) -> tuple[str, bool]:
    """Fallback: fetch page content via Jina Reader API."""
    try:
        session = await get_http_session()
        jina_url = f"https://r.jina.ai/{url}"
        jina_timeout = aiohttp.ClientTimeout(total=15, sock_read=12)
        async with session.get(jina_url, timeout=jina_timeout) as resp:
            if resp.status == 200:
                text = await resp.text()
                if text:
                    return _truncate_content(text), True
            logger.warning(f"Jina Reader returned status {resp.status} for {url}")
            return f"Could not fetch content from {url}.", False
    except asyncio.TimeoutError:
        logger.warning(f"Jina Reader fetch timed out for {url}")
        return f"Page fetch timed out for {url}.", False
    except Exception as e:
        logger.warning(f"Jina Reader fetch failed for {url}: {e}")
        return f"Could not fetch content from {url}: {e}", False


async def scrape_url(url: str) -> tuple[str, bool]:
    """Scrape a URL and return (markdown_content, success).

    Tiers:
    1. YouTube URLs → youtube-transcript-api
    2. Other URLs → crawl4ai
    3. Fallback → Jina Reader API
    """
    from ..utils import extract_youtube_video_id

    video_id = extract_youtube_video_id(url)
    if video_id:
        return await _fetch_youtube_transcript(video_id)

    # Tier 2: crawl4ai
    content, success = await _fetch_with_crawl4ai(url)
    if success:
        return content, True

    # Tier 3: Jina Reader fallback
    logger.info(f"Falling back to Jina Reader for {url}")
    return await _fetch_with_jina_reader(url)
