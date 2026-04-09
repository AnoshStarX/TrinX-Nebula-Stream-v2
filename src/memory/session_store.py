import asyncio
from collections import deque
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional

from pymongo import DESCENDING, MongoClient

from ..config import DATABASE_NAME, MONGO_URI, SESSION_HISTORY_LIMIT
from ..utils import logger

_client: Optional[MongoClient] = None
_connected = False
_fallback_store: dict[str, deque[dict[str, str]]] = {}
_fallback_summaries: dict[str, str] = {}

_metrics = {
    "backend": "memory_fallback",
    "connected": False,
    "loads": 0,
    "writes": 0,
    "summary_updates": 0,
    "load_failures": 0,
    "write_failures": 0,
    "avg_load_ms": 0.0,
    "avg_write_ms": 0.0,
}

_SUMMARY_MAX_CHARS = 1200
_SUMMARY_SOURCE_MSGS = 12
_SUMMARY_EVERY_TURNS = 4
_DB_OP_TIMEOUT_S = 2.5


async def _run_db_call(fn):
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(loop.run_in_executor(None, fn), timeout=_DB_OP_TIMEOUT_S)


def _update_avg(metric_key: str, elapsed_ms: float, count_key: str):
    count = _metrics[count_key]
    current = _metrics[metric_key]
    if count <= 1:
        _metrics[metric_key] = elapsed_ms
    else:
        _metrics[metric_key] = current + ((elapsed_ms - current) / count)


def _build_summary_from_messages(messages: list[dict]) -> str:
    """Create a compact deterministic summary from recent turns."""
    lines: list[str] = []
    for msg in messages[-_SUMMARY_SOURCE_MSGS:]:
        role = msg.get("role", "assistant")
        prefix = "User" if role == "human" else "Assistant"
        content = str(msg.get("content", "")).strip().replace("\n", " ")
        if not content:
            continue
        if len(content) > 160:
            content = content[:157] + "..."
        lines.append(f"{prefix}: {content}")

    summary = " | ".join(lines)
    if len(summary) > _SUMMARY_MAX_CHARS:
        summary = summary[-_SUMMARY_MAX_CHARS:]
    return summary


def _exc_label(exc: Exception) -> str:
    msg = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {msg}" if msg else name


def _degrade_to_fallback(reason: str, *, failure_metric: Optional[str] = None) -> None:
    global _client, _connected
    if failure_metric:
        _metrics[failure_metric] += 1
    if _client:
        try:
            _client.close()
        except Exception:
            pass
    _client = None
    _connected = False
    _metrics["backend"] = "memory_fallback"
    _metrics["connected"] = False
    logger.warning(f"Session store switched to in-process fallback: {reason}")


async def init_session_store() -> bool:
    """Initialize session persistence backend.

    Uses MongoDB when available. Falls back to in-process storage if unavailable.
    """
    global _client, _connected
    if _connected and _client:
        return True
    if not MONGO_URI:
        logger.warning("Session store: MONGODB_CONNECTION_STRING not set; using in-process fallback.")
        _metrics["backend"] = "memory_fallback"
        _metrics["connected"] = False
        return False

    try:
        _client = MongoClient(
            MONGO_URI,
            maxPoolSize=30,
            minPoolSize=0,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
            socketTimeoutMS=5000,
            retryWrites=True,
            connect=False,
        )
        await _run_db_call(lambda: _client.admin.command("ping"))
        _connected = True
        _metrics["backend"] = "mongodb"
        _metrics["connected"] = True
        logger.info("Session store initialized with MongoDB backend")
        return True
    except Exception as e:
        logger.warning(f"Session store init failed; using in-process fallback: {_exc_label(e)}")
        if _client:
            try:
                _client.close()
            except Exception:
                pass
        _client = None
        _connected = False
        _metrics["backend"] = "memory_fallback"
        _metrics["connected"] = False
        return False


async def close_session_store():
    global _client, _connected
    if _client:
        _client.close()
    _client = None
    _connected = False
    _metrics["connected"] = False


async def get_session_messages(session_id: str) -> list:
    """Load recent session messages as LangChain-compatible message objects."""
    global _connected
    from langchain_core.messages import AIMessage, HumanMessage

    started = perf_counter()
    _metrics["loads"] += 1

    if _connected and _client:
        try:
            db = _client[DATABASE_NAME]
            coll = db["session_messages"]
            docs = await _run_db_call(
                lambda: list(
                    coll.find({"session_id": session_id}, {"role": 1, "content": 1, "_id": 0})
                    .sort("ts", DESCENDING)
                    .limit(SESSION_HISTORY_LIMIT)
                )
            )
            docs.reverse()

            out = []
            for doc in docs:
                role = doc.get("role", "assistant")
                content = doc.get("content", "")
                if role == "human":
                    out.append(HumanMessage(content=content))
                else:
                    out.append(AIMessage(content=content))

            elapsed_ms = (perf_counter() - started) * 1000
            _update_avg("avg_load_ms", elapsed_ms, "loads")
            return out
        except TimeoutError:
            _degrade_to_fallback("Session store load timed out.", failure_metric="load_failures")
        except Exception as e:
            _degrade_to_fallback(
                f"Session store load failed: {_exc_label(e)}",
                failure_metric="load_failures",
            )

    items = list(_fallback_store.get(session_id, deque()))
    out = []
    for item in items[-SESSION_HISTORY_LIMIT:]:
        role = item.get("role", "assistant")
        content = item.get("content", "")
        if role == "human":
            out.append(HumanMessage(content=content))
        else:
            out.append(AIMessage(content=content))

    elapsed_ms = (perf_counter() - started) * 1000
    _update_avg("avg_load_ms", elapsed_ms, "loads")
    return out


async def append_session_turn(session_id: str, user_prompt: str, ai_response: str):
    """Persist a user+assistant turn and periodically update session summary."""
    global _connected
    started = perf_counter()
    _metrics["writes"] += 1

    now = datetime.now(timezone.utc)
    docs = [
        {"session_id": session_id, "role": "human", "content": user_prompt, "ts": now},
        {"session_id": session_id, "role": "assistant", "content": ai_response, "ts": now},
    ]

    if _connected and _client:
        try:
            db = _client[DATABASE_NAME]
            messages = db["session_messages"]
            summaries = db["session_summaries"]
            await _run_db_call(lambda: messages.insert_many(docs, ordered=True))

            # Refresh summary every N turns.
            count = await _run_db_call(
                lambda: messages.count_documents({"session_id": session_id})
            )
            turns = count // 2
            if turns > 0 and (turns % _SUMMARY_EVERY_TURNS == 0):
                recent = await _run_db_call(
                    lambda: list(
                        messages.find({"session_id": session_id}, {"role": 1, "content": 1, "_id": 0})
                        .sort("ts", DESCENDING)
                        .limit(_SUMMARY_SOURCE_MSGS)
                    )
                )
                recent.reverse()
                summary = _build_summary_from_messages(recent)
                await _run_db_call(
                    lambda: summaries.update_one(
                        {"session_id": session_id},
                        {
                            "$set": {
                                "summary": summary,
                                "turn_count": turns,
                                "updated_at": now,
                            }
                        },
                        upsert=True,
                    ),
                )
                _metrics["summary_updates"] += 1

            elapsed_ms = (perf_counter() - started) * 1000
            _update_avg("avg_write_ms", elapsed_ms, "writes")
            return
        except TimeoutError:
            _degrade_to_fallback("Session store write timed out.", failure_metric="write_failures")
        except Exception as e:
            _degrade_to_fallback(
                f"Session store write failed: {_exc_label(e)}",
                failure_metric="write_failures",
            )

    bucket = _fallback_store.setdefault(session_id, deque(maxlen=SESSION_HISTORY_LIMIT))
    bucket.append({"role": "human", "content": user_prompt})
    bucket.append({"role": "assistant", "content": ai_response})
    if len(bucket) >= 2:
        _fallback_summaries[session_id] = _build_summary_from_messages(list(bucket))
        _metrics["summary_updates"] += 1
    elapsed_ms = (perf_counter() - started) * 1000
    _update_avg("avg_write_ms", elapsed_ms, "writes")


async def get_session_summary(session_id: str) -> Optional[str]:
    """Return the latest persisted summary for the session (if available)."""
    if _connected and _client:
        try:
            db = _client[DATABASE_NAME]
            summaries = db["session_summaries"]
            doc = await _run_db_call(
                lambda: summaries.find_one({"session_id": session_id}, {"summary": 1, "_id": 0}),
            )
            if doc:
                return doc.get("summary")
        except TimeoutError:
            _degrade_to_fallback("Session summary read timed out.", failure_metric="load_failures")
        except Exception as e:
            _degrade_to_fallback(
                f"Session summary read failed: {_exc_label(e)}",
                failure_metric="load_failures",
            )
    return _fallback_summaries.get(session_id)


async def get_memory_metrics() -> dict:
    return {
        "backend": _metrics["backend"],
        "connected": _metrics["connected"],
        "loads": _metrics["loads"],
        "writes": _metrics["writes"],
        "summary_updates": _metrics["summary_updates"],
        "load_failures": _metrics["load_failures"],
        "write_failures": _metrics["write_failures"],
        "avg_load_ms": round(float(_metrics["avg_load_ms"]), 2),
        "avg_write_ms": round(float(_metrics["avg_write_ms"]), 2),
    }
