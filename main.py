import uuid
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessageChunk

from src.graph import chat_graph
from src.nodes.preprocessing import preprocessing_node
from src.utils import (
    logger, is_news_related_query,
    is_web_search_query, is_web_scrape_query, is_wallet_query, detect_crypto_symbols,
    init_redis, close_redis, get_cache_size,
)
from src.api.tavily import fetch_tavily_search
from src.api.user import init_mongo, close_mongo
from src.http_client import init_http_session, close_http_session
from src.memory.session_store import (
    init_session_store,
    close_session_store,
    get_session_messages,
    get_session_summary,
    append_session_turn,
    get_memory_metrics,
)
from src.memory.user_memory_store import (
    init_user_memory_store,
    close_user_memory_store,
    retrieve_user_memories,
    store_user_memory,
    get_user_memory_metrics,
)
from src.memory.retrieval_policy import (
    should_retrieve_long_term_memory,
    adaptive_memory_limit,
    apply_memory_budget,
    mark_retrieval_attempt,
    mark_retrieval_gate_skip,
    mark_retrieval_timeout,
    mark_retrieval_failure,
    get_retrieval_policy_metrics,
)
from src.config import (
    LONG_TERM_MEMORY_MAX_ITEMS,
    LONG_TERM_MEMORY_MAX_CHARS,
    LONG_TERM_MEMORY_MAX_CHARS_PER_ITEM,
    LONG_TERM_MEMORY_READ_TIMEOUT_MS,
    LONG_TERM_MEMORY_WRITE_TIMEOUT_MS,
)
from dotenv import load_dotenv
import os

load_dotenv()

# Verify LangSmith Configuration
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    if os.getenv("LANGCHAIN_API_KEY"):
        logger.info(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT', 'default')}")
    else:
        logger.warning("LANGCHAIN_TRACING_V2 is true but LANGCHAIN_API_KEY is missing!")
else:
    logger.info("LangSmith tracing is NOT enabled. Set LANGCHAIN_TRACING_V2=true to enable.")

# === FastAPI App ===
app = FastAPI(title="TrinX Nebula Stream", version="14.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    news_available: Optional[bool] = None
    error: Optional[str] = None
    token_usage: Optional[dict] = None


_runtime_metrics = {
    "session_persist_failures": 0,
    "memory_write_timeouts": 0,
    "memory_write_failures": 0,
}

_llm_usage_metrics = {
    "requests_recorded": 0,
    "provider_records": 0,
    "cache_records": 0,
    "provider_input_tokens": 0,
    "provider_output_tokens": 0,
    "provider_total_tokens": 0,
    "stream_estimated_output_tokens": 0.0,
    "stream_unknown_records": 0,
}


def _record_llm_usage(token_usage: Optional[dict], *, endpoint: str, estimated_output_chars: int = 0) -> None:
    _llm_usage_metrics["requests_recorded"] += 1
    usage = token_usage or {}
    if usage.get("from_cache"):
        _llm_usage_metrics["cache_records"] += 1
        return

    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    if input_tokens > 0 or output_tokens > 0 or total_tokens > 0:
        _llm_usage_metrics["provider_records"] += 1
        _llm_usage_metrics["provider_input_tokens"] += input_tokens
        _llm_usage_metrics["provider_output_tokens"] += output_tokens
        _llm_usage_metrics["provider_total_tokens"] += total_tokens
        return

    if endpoint == "/chat/stream":
        _llm_usage_metrics["stream_unknown_records"] += 1
        _llm_usage_metrics["stream_estimated_output_tokens"] += float(max(0, estimated_output_chars) / 4.0)


async def _load_long_term_memories(
    user_id: Optional[str],
    session_id: str,
    prompt: str,
) -> list[str]:
    if not should_retrieve_long_term_memory(prompt):
        mark_retrieval_gate_skip()
        return []

    mark_retrieval_attempt()
    limit = min(
        LONG_TERM_MEMORY_MAX_ITEMS,
        adaptive_memory_limit(prompt, base=LONG_TERM_MEMORY_MAX_ITEMS),
    )
    try:
        async with asyncio.timeout(max(0.05, LONG_TERM_MEMORY_READ_TIMEOUT_MS / 1000.0)):
            candidates = await retrieve_user_memories(
                user_id,
                session_id,
                prompt,
                limit=max(limit, LONG_TERM_MEMORY_MAX_ITEMS),
            )
    except TimeoutError:
        mark_retrieval_timeout()
        logger.warning(f"Long-term memory retrieval timeout for session {session_id}")
        return []
    except Exception:
        mark_retrieval_failure()
        logger.warning(f"Long-term memory retrieval failed for session {session_id}", exc_info=True)
        return []

    return apply_memory_budget(
        prompt,
        candidates,
        max_items=limit,
        max_chars=LONG_TERM_MEMORY_MAX_CHARS,
        max_chars_per_item=LONG_TERM_MEMORY_MAX_CHARS_PER_ITEM,
    )


async def _persist_turn_and_memory(
    session_id: str,
    user_id: Optional[str],
    user_prompt: str,
    ai_response: str,
) -> None:
    try:
        # Session continuity is critical for next-turn context.
        # append_session_turn already has DB-level timeout/fallback handling.
        await append_session_turn(session_id, user_prompt, ai_response)
    except Exception:
        _runtime_metrics["session_persist_failures"] += 1
        logger.warning(f"Session persist failed for session {session_id}", exc_info=True)

    try:
        async with asyncio.timeout(max(0.5, LONG_TERM_MEMORY_WRITE_TIMEOUT_MS / 1000.0)):
            await store_user_memory(user_id, session_id, user_prompt)
    except TimeoutError:
        _runtime_metrics["memory_write_timeouts"] += 1
        logger.warning(f"Long-term memory write timeout for session {session_id}")
    except Exception:
        _runtime_metrics["memory_write_failures"] += 1
        logger.warning(f"Long-term memory write failed for session {session_id}", exc_info=True)


@app.on_event("startup")
async def startup_event():
    await init_http_session()
    await init_mongo()
    await init_redis()
    await init_session_store()
    await init_user_memory_store()


@app.on_event("shutdown")
async def shutdown_event():
    await close_http_session()
    await close_mongo()
    await close_redis()
    await close_session_store()
    await close_user_memory_store()


@app.post("/chat", response_model=ChatResponse)
async def chat_api(req: ChatRequest):
    """Chat endpoint - full graph invocation with LangSmith tracing."""
    session_id = req.session_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": session_id},
        "run_name": "chat_api",
        "metadata": {"session_id": session_id, "endpoint": "/chat"},
    }

    try:
        logger.info(f"Chat request for session {session_id}")

        user_id_str = str(req.user_id) if req.user_id else None
        new_user_message = HumanMessage(content=req.prompt)
        history_task = asyncio.create_task(get_session_messages(session_id))
        summary_task = asyncio.create_task(get_session_summary(session_id))
        long_term_task = asyncio.create_task(_load_long_term_memories(user_id_str, session_id, req.prompt))
        history, session_summary, long_term_memories = await asyncio.gather(
            history_task, summary_task, long_term_task
        )
        input_messages = history + [new_user_message]

        final_state = await chat_graph.ainvoke(
            {
                "messages": input_messages,
                "user_id": user_id_str,
                "session_summary": session_summary,
                "long_term_memories": long_term_memories,
            },
            config=config,
        )

        ai_response = final_state["messages"][-1].content
        token_usage = final_state.get("token_usage") if isinstance(final_state, dict) else None
        _record_llm_usage(token_usage, endpoint="/chat")
        await _persist_turn_and_memory(session_id, user_id_str, req.prompt, ai_response)

        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            intent=final_state.get("intent", "UNKNOWN"),
            news_available=final_state.get("news_available"),
            error=final_state.get("error_message"),
            token_usage=token_usage,
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={
            "response": "I'm getting a lot of requests. Please try again.",
            "session_id": session_id,
            "intent": "ERROR",
            "error": str(e),
        })

@app.post("/chat/stream")
async def chat_stream_api(req: ChatRequest):
    """Streaming chat endpoint optimized for lower latency.

    Uses the same chat_graph as /chat but streams tokens via `stream_mode="messages"`
    to reduce per-event overhead vs full event streaming.
    """
    session_id = req.session_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": session_id},
        "run_name": "chat_stream",
        "metadata": {"session_id": session_id, "endpoint": "/chat/stream"},
    }

    def _extract_text(content: object) -> str:
        """Normalize chunk content to plain text for SSE payloads."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            return "".join(parts)
        return str(content)

    async def response_generator():
        try:
            logger.info(f"Streaming chat request for session {session_id}")

            user_id_str = str(req.user_id) if req.user_id else None
            user_message = HumanMessage(content=req.prompt)
            history_task = asyncio.create_task(get_session_messages(session_id))
            summary_task = asyncio.create_task(get_session_summary(session_id))
            long_term_task = asyncio.create_task(_load_long_term_memories(user_id_str, session_id, req.prompt))
            history, session_summary, long_term_memories = await asyncio.gather(
                history_task, summary_task, long_term_task
            )
            input_messages = history + [user_message]

            # Push a lightweight SSE comment so proxies/clients flush headers early.
            yield b": connected\n\n"

            token_streamed = False
            full_message_streamed = False
            streamed_tokens = []
            final_full_message = ""
            stream_timed_out = False
            final_state_token_usage = None

            async def _consume_stream():
                nonlocal token_streamed, full_message_streamed, final_full_message
                async for message, _metadata in chat_graph.astream(
                    {
                        "messages": input_messages,
                        "user_id": user_id_str,
                        "session_summary": session_summary,
                        "long_term_memories": long_term_memories,
                    },
                    config=config,
                    stream_mode="messages",
                ):
                    content = _extract_text(getattr(message, "content", ""))
                    if not content:
                        continue

                    is_chunk = isinstance(message, AIMessageChunk)

                    # Skip final full-message emission when token chunks were already streamed.
                    if token_streamed and not is_chunk:
                        continue
                    if full_message_streamed and not is_chunk:
                        continue

                    if is_chunk:
                        token_streamed = True
                        streamed_tokens.append(content)
                    else:
                        full_message_streamed = True
                        final_full_message = content

                    sse_data = json.dumps({"token": content}, ensure_ascii=False, separators=(",", ":"))
                    yield f"data: {sse_data}\n\n".encode("utf-8")

            try:
                async with asyncio.timeout(45):
                    async for payload in _consume_stream():
                        yield payload
            except TimeoutError:
                stream_timed_out = True
                logger.warning(f"Stream timeout for session {session_id}; falling back to ainvoke")

            if token_streamed:
                final_response_text = "".join(streamed_tokens)
            else:
                final_response_text = final_full_message

            # Fallback path for cache-hit/no-token cases and timeout edge cases.
            if not final_response_text or stream_timed_out:
                final_state = await chat_graph.ainvoke(
                    {
                        "messages": input_messages,
                        "user_id": user_id_str,
                        "session_summary": session_summary,
                        "long_term_memories": long_term_memories,
                    },
                    config=config,
                )
                final_response_text = _extract_text(final_state["messages"][-1].content)
                final_state_token_usage = final_state.get("token_usage") if isinstance(final_state, dict) else None
                if final_response_text:
                    sse_data = json.dumps({"token": final_response_text}, ensure_ascii=False, separators=(",", ":"))
                    yield f"data: {sse_data}\n\n".encode("utf-8")

            if final_response_text:
                _record_llm_usage(
                    final_state_token_usage,
                    endpoint="/chat/stream",
                    estimated_output_chars=len(final_response_text),
                )
                # Persist before DONE so immediate next prompt sees prior turn context.
                await _persist_turn_and_memory(session_id, user_id_str, req.prompt, final_response_text)

            yield b"data: [DONE]\n\n"

        except asyncio.CancelledError:
            logger.warning(f"Stream cancelled for session {session_id}")
            yield b"data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            error_msg = json.dumps({"error": str(e)}, ensure_ascii=False, separators=(",", ":"))
            yield f"data: {error_msg}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@app.get("/debug/test-intent")
async def test_intent(query: str = "What is happening in AI today?"):
    """Debug endpoint to test intent detection."""
    messages = [HumanMessage(content=query)]
    state = {
        "messages": messages,
        "user_id": None,
        "crypto_symbols": [],
        "intent": None
    }

    result = await preprocessing_node(state)

    return {
        "query": query,
        "intent": result["intent"],
        "crypto_symbols": result["crypto_symbols"],
        "is_news": is_news_related_query(query),
        "is_web_search": is_web_search_query(query),
        "is_web_scrape": is_web_scrape_query(query),
        "is_wallet": is_wallet_query(query),
        "detected_symbols": detect_crypto_symbols(query)
    }

@app.post("/debug/test-scrape")
async def test_scrape(url: str = "https://example.com"):
    """Debug endpoint to test web scraping directly."""
    from src.api.scraper import scrape_url
    content, success = await scrape_url(url)
    return {
        "url": url,
        "success": success,
        "content_length": len(content),
        "preview": content[:500],
    }


@app.post("/debug/test-web-search")
async def test_web_search(query: str = "What is the latest AI news?"):
    """Debug endpoint to test web search directly."""
    results, success = await fetch_tavily_search(query, max_results=5)
    return {
        "query": query,
        "success": success,
        "results": results
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    cache_size = await get_cache_size()
    session_memory_metrics = await get_memory_metrics()
    user_memory_metrics = await get_user_memory_metrics()
    retrieval_policy_metrics = get_retrieval_policy_metrics()
    return {
        "status": "healthy",
        "version": "14.0",
        "optimizations": [
            "ChatOpenAI with auto LangSmith tracing",
            "Redis-backed shared cache",
            "Redis-backed shared session history",
            "Heuristics-first intent detection",
            "Parallel crypto+news fetching",
            "Parallel news source fetching",
            "Shared aiohttp connection pooling",
            "Reduced API timeouts",
            "LangGraph messages-mode token streaming",
        ],
        "cache_size": cache_size,
        "memory": {
            "session": session_memory_metrics,
            "long_term": user_memory_metrics,
            "retrieval_policy": retrieval_policy_metrics,
            "runtime": dict(_runtime_metrics),
            "llm_usage": dict(_llm_usage_metrics),
        },
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "TrinX Nebula Stream",
        "version": "14.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat (POST)",
            "stream": "/chat/stream (POST)",
            "health": "/health (GET)",
            "root": "/ (GET)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
