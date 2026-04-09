from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    intent: Optional[str]
    crypto_symbols: List[str]
    news_query: Optional[str]
    news_results: Optional[str]
    news_available: Optional[bool]
    crypto_prices: Optional[str]
    error_message: Optional[str]
    user_id: Optional[str]
    user_data: Optional[Dict[str, Any]]
    rewards_data: Optional[Dict[str, Any]]
    web_search_results: Optional[str]
    scraped_content: Optional[str]
    scraped_url: Optional[str]
    long_term_memories: Optional[List[str]]
    session_summary: Optional[str]
    token_usage: Optional[Dict[str, Any]]
