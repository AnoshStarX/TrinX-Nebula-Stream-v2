from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes.preprocessing import preprocessing_node
from .nodes.web_search import web_search_node
from .nodes.news_search import news_search_node
from .nodes.crypto_data import fetch_crypto_data_node
from .nodes.user_data import fetch_user_data_node
from .nodes.parallel_fetch import parallel_fetch_node
from .nodes.web_scrape import web_scrape_node
from .nodes.generate_response import generate_response_node
from .utils import logger

def router(state: GraphState):
    """Route to appropriate node based on intent."""
    intent = state.get("intent")
    logger.info(f"Router: {intent}")

    if intent == "WALLET_QUERY":
        return "fetch_user_data_node"
    elif intent == "CRYPTO_PRICE_QUERY":
        return "fetch_crypto_data_node"
    elif intent in ["NEWS_QUERY", "GENERAL_NEWS_QUERY"]:
        return "news_search_node"
    elif intent == "PRICE_AND_NEWS_QUERY":
        return "parallel_fetch_node"
    elif intent == "WEB_SEARCH_QUERY":
        return "web_search_node"
    elif intent == "WEB_SCRAPE_QUERY":
        return "web_scrape_node"
    else:
        return "generate_response_node"


def build_chat_graph():
    """Build the unified chat graph (used by both /chat and /chat/stream)."""
    graph = StateGraph(GraphState)

    graph.add_node("preprocessing", preprocessing_node)
    graph.add_node("news_search_node", news_search_node)
    graph.add_node("fetch_crypto_data_node", fetch_crypto_data_node)
    graph.add_node("fetch_user_data_node", fetch_user_data_node)
    graph.add_node("web_search_node", web_search_node)
    graph.add_node("web_scrape_node", web_scrape_node)
    graph.add_node("parallel_fetch_node", parallel_fetch_node)
    graph.add_node("generate_response_node", generate_response_node)

    graph.set_entry_point("preprocessing")
    graph.add_conditional_edges("preprocessing", router)

    graph.add_edge("news_search_node", "generate_response_node")
    graph.add_edge("fetch_user_data_node", "generate_response_node")
    graph.add_edge("web_search_node", "generate_response_node")
    graph.add_edge("web_scrape_node", "generate_response_node")
    graph.add_edge("fetch_crypto_data_node", "generate_response_node")
    graph.add_edge("parallel_fetch_node", "generate_response_node")
    graph.add_edge("generate_response_node", END)

    return graph.compile()


chat_graph = build_chat_graph()
