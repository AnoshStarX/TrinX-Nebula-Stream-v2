import json
from typing import List, Dict, Any
from ..config import COINGECKO_API, COINGECKO_MAP, USD_INR_RATE
from ..utils import logger
from ..http_client import get_http_session

async def fetch_crypto_data(symbols: List[str]) -> str:
    """OPTIMIZATION: Reduced timeout to 5s."""
    if not symbols:
        return json.dumps({})
    
    results: Dict[str, Any] = {}
    coingecko_ids = [COINGECKO_MAP[s] for s in symbols if s in COINGECKO_MAP]
    if not coingecko_ids:
        return json.dumps({"error": "No valid symbols."})

    try:
        session = await get_http_session()
        params = {"ids": ",".join(coingecko_ids), "vs_currencies": "usd", "include_24hr_change": "true"}
        async with session.get(COINGECKO_API, params=params, timeout=5) as r:
            r.raise_for_status()
            data = await r.json()

            for sym in symbols:
                cid = COINGECKO_MAP.get(sym)
                if cid and cid in data:
                    price_usd = data[cid].get("usd")
                    if price_usd is not None:
                        price_inr = price_usd * USD_INR_RATE
                        results[sym] = {
                            "symbol": sym,
                            "current_price_usd": price_usd,
                            "current_price_inr": round(price_inr, 2),
                            "change_24h_percent": round(data[cid].get("usd_24h_change", 0.0), 2),
                        }
                    else:
                        results[sym] = {"error": "Price data unavailable."}
                else:
                    results[sym] = {"error": "Symbol not found."}
    except Exception as e:
        logger.error(f"Crypto data fetch failed: {e}")
        return json.dumps({"error": f"API call failed: {e}"})
    return json.dumps(results, indent=2)
