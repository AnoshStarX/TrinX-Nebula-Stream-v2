import json
from ..state import GraphState
from ..api.crypto import fetch_crypto_data

async def fetch_crypto_data_node(state: GraphState):
    """Fetch cryptocurrency price data."""
    symbols = state.get("crypto_symbols", [])
    if not symbols:
        return {"crypto_prices": json.dumps({"info": "No symbols mentioned."})}
    crypto_data = await fetch_crypto_data(symbols)
    return {"crypto_prices": crypto_data}
