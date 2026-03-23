from langchain_openai import ChatOpenAI
from ..config import GROQ_API_KEYS, GROQ_BASE_URL, DEFAULT_MODEL
from ..utils import logger


def _build_llm():
    """Build ChatOpenAI with Groq backend and automatic key fallback.

    Using ChatOpenAI pointed at Groq's OpenAI-compatible endpoint gives us:
    - Automatic LangSmith tracing for every LLM call
    - Token counting and metadata in traces
    - Built-in retry logic with max_retries
    - httpx connection pooling for lower latency
    - Native streaming support via astream_events
    """
    if not GROQ_API_KEYS:
        logger.error("No GROQ_API_KEYS configured.")
        return None

    llms = []
    for key in GROQ_API_KEYS:
        llm = ChatOpenAI(
            base_url=GROQ_BASE_URL,
            api_key=key,
            model=DEFAULT_MODEL,
            temperature=0.3,
            max_tokens=2000,
            streaming=True,
            request_timeout=12,
            max_retries=2,
        )
        llms.append(llm)

    primary = llms[0]
    if len(llms) > 1:
        primary = primary.with_fallbacks(llms[1:])

    logger.info(f"Groq LLM initialized with {len(llms)} key(s), model: {DEFAULT_MODEL}")
    return primary


llm = _build_llm()
