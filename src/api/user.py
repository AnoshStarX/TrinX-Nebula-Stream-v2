import asyncio
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson.objectid import ObjectId
from ..config import MONGO_URI, DATABASE_NAME
from ..utils import logger

_mongo_client: Optional[MongoClient] = None
_DB_OP_TIMEOUT_S = 2.5


def _exc_label(exc: Exception) -> str:
    msg = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {msg}" if msg else name


def _disable_mongo(reason: str):
    global _mongo_client
    if _mongo_client:
        try:
            _mongo_client.close()
        except Exception:
            pass
    _mongo_client = None
    logger.warning(f"MongoDB client disabled: {reason}")


async def init_mongo() -> bool:
    """Initialize shared MongoDB client during app startup."""
    global _mongo_client
    if _mongo_client:
        return True
    if not MONGO_URI:
        logger.warning("MONGODB_CONNECTION_STRING not set. Wallet queries will be disabled.")
        return False

    try:
        _mongo_client = MongoClient(
            MONGO_URI,
            maxPoolSize=30,
            minPoolSize=0,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            socketTimeoutMS=5000,
            retryWrites=True,
            connect=False,
        )
        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(None, lambda: _mongo_client.admin.command("ping")),
            timeout=_DB_OP_TIMEOUT_S,
        )
        logger.info("MongoDB client initialized")
        return True
    except Exception as e:
        logger.warning(f"MongoDB initialization failed: {_exc_label(e)}")
        _disable_mongo(f"init failed: {_exc_label(e)}")
        return False


async def close_mongo():
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None
        logger.info("MongoDB client closed")

async def get_user_and_rewards_data_async(user_id: str):
    """OPTIMIZATION: Fetch user and rewards in parallel."""
    if not _mongo_client:
        return {"error": "MongoDB client not configured."}

    try:
        loop = asyncio.get_running_loop()
        db = _mongo_client[DATABASE_NAME]
        users_collection = db['users']
        
        try:
            object_id = ObjectId(user_id)
        except Exception:
            return {"error": f"Invalid user ID format: {user_id}"}

        # Fetch user document
        user_document = await loop.run_in_executor(
            None, users_collection.find_one, {"_id": object_id}
        )
        if not user_document:
            return {"error": f"User not found"}

        result = {
            "user_id": str(user_document["_id"]),
            "is_premium": user_document.get("is_premium_user", "N/A"),
            "words_typed": user_document.get("total_words_typed", 0),
            "tix": user_document.get("total_tix_tokens", 0)
        }

        return result

    except Exception as e:
        if isinstance(e, (TimeoutError, PyMongoError)):
            _disable_mongo(f"runtime failure: {_exc_label(e)}")
        logger.error(f"Error fetching user data: {_exc_label(e)}")
        return {"error": f"Error fetching data: {_exc_label(e)}"}
