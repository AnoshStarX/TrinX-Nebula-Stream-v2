import asyncio
import math
import re
from datetime import datetime, timezone
from hashlib import md5
from typing import Optional

from pymongo import DESCENDING, MongoClient

from ..config import DATABASE_NAME, MONGO_URI, PGVECTOR_DSN, USER_MEMORY_BACKEND
from ..utils import logger

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency
    psycopg = None
    dict_row = None

try:
    from pgvector.psycopg import register_vector
except Exception:  # pragma: no cover - optional dependency
    register_vector = None

_mongo_client: Optional[MongoClient] = None
_mongo_connected = False
_backend = "memory_fallback"
_fallback_memories: dict[str, list[dict]] = {}

_VECTOR_DIM = 192
_MAX_MEMORY_TEXT = 260
_RECENCY_DAYS = 30
_DB_OP_TIMEOUT_S = 2.5

_metrics = {
    "backend": "memory_fallback",
    "connected": False,
    "stores": 0,
    "retrievals": 0,
    "store_failures": 0,
    "retrieval_failures": 0,
    "db_timeouts": 0,
    "conflicts_resolved": 0,
    "rerank_calls": 0,
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _vectorize(text: str) -> list[float]:
    vec = [0.0] * _VECTOR_DIM
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for token in tokens:
        idx = hash(token) % _VECTOR_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _scope_key(user_id: Optional[str], session_id: Optional[str]) -> str:
    if user_id:
        return f"user:{user_id}"
    return f"session:{session_id or 'anonymous'}"


def _extract_candidate_memories(user_text: str) -> list[dict]:
    """Rule-based memory extraction.

    Extracts multiple durable user facts/preferences/tasks from one turn.
    """
    if not user_text:
        return []

    raw = " ".join(user_text.strip().split())
    if not raw:
        return []
    lower = raw.lower()

    candidates: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(memory_type: str, text: str, confidence: float, entity_key: Optional[str]):
        cleaned = " ".join(str(text or "").split()).strip(" .,:;")
        if not cleaned:
            return
        cleaned = cleaned[:_MAX_MEMORY_TEXT]
        key = (memory_type, cleaned.lower(), str(entity_key or ""))
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "memory_type": memory_type,
                "text": cleaned,
                "confidence": float(confidence),
                "entity_key": entity_key,
            }
        )

    triggers = [
        "remember that",
        "remember this",
        "dont forget",
        "don't forget",
        "note that",
    ]
    for trig in triggers:
        if trig in lower:
            idx = lower.find(trig) + len(trig)
            _add("explicit", raw[idx:], 0.95, None)
            break

    fav_match = re.search(r"\bmy favorite ([a-z0-9_ ]{1,40}) is ([^.!?,]{1,60})", raw, flags=re.IGNORECASE)
    if fav_match:
        subject = fav_match.group(1).strip().lower().replace(" ", "_")
        value = fav_match.group(2).strip()
        _add("preference", f"My favorite {fav_match.group(1).strip()} is {value}", 0.88, f"favorite_{subject}")

    prefer_match = re.search(r"\bi prefer ([^.!?,]{1,80})", raw, flags=re.IGNORECASE)
    if prefer_match:
        _add("preference", f"I prefer {prefer_match.group(1).strip()}", 0.84, "preference_general")
    elif any(re.search(p, lower) for p in [r"\bi like\b", r"\bi love\b", r"\bmy favorite\b"]):
        _add("preference", raw, 0.78, "preference_general")

    specific_profile = 0
    name_match = re.search(
        r"\bmy name is ([a-z][a-z .'-]{0,40}?)(?:\s*(?:,|\.|!|\?| and\b|$))",
        raw,
        flags=re.IGNORECASE,
    )
    if name_match:
        _add("profile", f"My name is {name_match.group(1).strip()}", 0.9, "profile_name")
        specific_profile += 1

    live_match = re.search(
        r"\bi live in ([a-z0-9 .,'-]{1,60}?)(?:\s*(?:,|\.|!|\?| and\b|$))",
        raw,
        flags=re.IGNORECASE,
    )
    if live_match:
        _add("profile", f"I live in {live_match.group(1).strip()}", 0.84, "profile_location")
        specific_profile += 1

    work_match = re.search(
        r"\bi work as ([a-z0-9 .,'-]{1,60}?)(?:\s*(?:,|\.|!|\?| and\b|$))",
        raw,
        flags=re.IGNORECASE,
    )
    if work_match:
        _add("profile", f"I work as {work_match.group(1).strip()}", 0.84, "profile_occupation")
        specific_profile += 1

    if specific_profile == 0 and any(
        re.search(p, lower) for p in [r"\bmy name is\b", r"\bi am\b", r"\bi'm\b", r"\bi work as\b", r"\bi live in\b"]
    ):
        _add("profile", raw, 0.72, None)

    task_match = re.search(
        r"\b(remind me|i need to|i plan to|i want to) ([^.!?]{1,120})",
        raw,
        flags=re.IGNORECASE,
    )
    if task_match:
        prefix = task_match.group(1).strip()
        detail = task_match.group(2).strip()
        _add("task", f"{prefix} {detail}", 0.7, None)

    # Keep writes bounded per turn.
    return candidates[:4]


def _set_backend(name: str, connected: bool):
    global _backend
    _backend = name
    _metrics["backend"] = name
    _metrics["connected"] = connected


def _exc_label(exc: Exception) -> str:
    msg = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {msg}" if msg else name


def _degrade_to_fallback(reason: str):
    global _mongo_client, _mongo_connected
    if _mongo_client:
        try:
            _mongo_client.close()
        except Exception:
            pass
    _mongo_client = None
    _mongo_connected = False
    _set_backend("memory_fallback", False)
    logger.warning(f"User memory store switched to in-process fallback: {reason}")


async def _run_mongo_call(fn):
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(loop.run_in_executor(None, fn), timeout=_DB_OP_TIMEOUT_S)


def _pgvector_available() -> bool:
    return bool(PGVECTOR_DSN and psycopg and register_vector)


def _init_pgvector_schema_sync():
    if not PGVECTOR_DSN:
        raise RuntimeError("PGVECTOR_DSN is not configured")
    with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS user_memories_pg (
                    scope TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    text TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    entity_key TEXT,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    confidence REAL NOT NULL,
                    embedding VECTOR({_VECTOR_DIM}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    resolved_at TIMESTAMPTZ,
                    PRIMARY KEY (scope, fingerprint)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_mem_scope_updated ON user_memories_pg (scope, updated_at DESC)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_mem_scope_active ON user_memories_pg (scope, active)"
            )
            # Optional advanced index (works on newer pgvector releases).
            try:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_user_mem_embedding_hnsw ON user_memories_pg USING hnsw (embedding vector_cosine_ops)"
                )
            except Exception:
                pass
            # Add columns for existing deployments.
            for ddl in [
                "ALTER TABLE user_memories_pg ADD COLUMN IF NOT EXISTS entity_key TEXT",
                "ALTER TABLE user_memories_pg ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE",
                "ALTER TABLE user_memories_pg ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ",
            ]:
                try:
                    cur.execute(ddl)
                except Exception:
                    pass


def _pg_upsert_memory_sync(
    scope: str,
    fingerprint: str,
    text: str,
    memory_type: str,
    entity_key: Optional[str],
    confidence: float,
    vector: list[float],
    now: datetime,
):
    with psycopg.connect(PGVECTOR_DSN, autocommit=True) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            if entity_key:
                cur.execute(
                    """
                    UPDATE user_memories_pg
                    SET active = FALSE, resolved_at = %s, updated_at = %s
                    WHERE scope = %s
                      AND memory_type = %s
                      AND entity_key = %s
                      AND active = TRUE
                      AND text <> %s
                    """,
                    (now, now, scope, memory_type, entity_key, text),
                )
            cur.execute(
                """
                INSERT INTO user_memories_pg (
                    scope, fingerprint, text, memory_type, entity_key, active, confidence, embedding, created_at, updated_at, resolved_at
                ) VALUES (%s, %s, %s, %s, %s, TRUE, %s, %s, %s, %s, NULL)
                ON CONFLICT (scope, fingerprint)
                DO UPDATE SET
                    text = EXCLUDED.text,
                    memory_type = EXCLUDED.memory_type,
                    entity_key = EXCLUDED.entity_key,
                    active = TRUE,
                    confidence = EXCLUDED.confidence,
                    embedding = EXCLUDED.embedding,
                    updated_at = EXCLUDED.updated_at,
                    resolved_at = NULL
                """,
                (scope, fingerprint, text, memory_type, entity_key, confidence, vector, now, now),
            )


def _pg_fetch_memories_sync(scope: str, query_vec: list[float], raw_limit: int) -> list[dict]:
    with psycopg.connect(PGVECTOR_DSN, row_factory=dict_row) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT text, memory_type, entity_key, confidence, updated_at, (1 - (embedding <=> %s)) AS similarity
                FROM user_memories_pg
                WHERE scope = %s
                  AND active = TRUE
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_vec, scope, query_vec, raw_limit),
            )
            return list(cur.fetchall())


async def init_user_memory_store() -> bool:
    global _mongo_client, _mongo_connected

    preferred = USER_MEMORY_BACKEND
    if preferred in ("auto", "pgvector"):
        if _pgvector_available():
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _init_pgvector_schema_sync)
                _set_backend("pgvector", True)
                logger.info("User memory store initialized with pgvector backend")
                return True
            except Exception as e:
                logger.warning(f"User memory pgvector init failed: {_exc_label(e)}")
                if preferred == "pgvector":
                    _set_backend("memory_fallback", False)
                    return False
        elif preferred == "pgvector":
            logger.warning(
                "USER_MEMORY_BACKEND=pgvector but pgvector dependencies/config missing; falling back."
            )

    if preferred in ("auto", "mongo", "mongodb"):
        if not MONGO_URI:
            if preferred in ("mongo", "mongodb"):
                logger.warning(
                    "USER_MEMORY_BACKEND=mongo but MONGODB_CONNECTION_STRING missing; using in-process fallback."
                )
            _set_backend("memory_fallback", False)
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
            await _run_mongo_call(lambda: _mongo_client.admin.command("ping"))
            _mongo_connected = True
            _set_backend("mongodb", True)
            logger.info("User memory store initialized with MongoDB backend")
            return True
        except Exception as e:
            logger.warning(f"User memory Mongo init failed; using in-process fallback: {_exc_label(e)}")
            _degrade_to_fallback(f"Mongo init failed: {_exc_label(e)}")

    _set_backend("memory_fallback", False)
    return False


async def close_user_memory_store():
    global _mongo_client, _mongo_connected
    if _mongo_client:
        _mongo_client.close()
    _mongo_client = None
    _mongo_connected = False
    _metrics["connected"] = False


async def store_user_memory(user_id: Optional[str], session_id: Optional[str], user_text: str):
    _metrics["stores"] += 1
    candidates = _extract_candidate_memories(user_text)
    if not candidates:
        return

    scope = _scope_key(user_id, session_id)
    now = datetime.now(timezone.utc)

    for candidate in candidates:
        text = candidate["text"]
        vector = _vectorize(text)
        fingerprint = md5(f"{scope}:{text.lower()}".encode()).hexdigest()
        doc = {
            "scope": scope,
            "text": text,
            "memory_type": candidate["memory_type"],
            "entity_key": candidate.get("entity_key"),
            "active": True,
            "confidence": float(candidate["confidence"]),
            "vector": vector,
            "fingerprint": fingerprint,
            "created_at": now,
            "updated_at": now,
            "resolved_at": None,
        }

        stored = False

        if _backend == "pgvector":
            try:
                loop = asyncio.get_running_loop()
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: _pg_upsert_memory_sync(
                            scope,
                            fingerprint,
                            text,
                            candidate["memory_type"],
                            candidate.get("entity_key"),
                            float(candidate["confidence"]),
                            vector,
                            now,
                        ),
                    ),
                    timeout=_DB_OP_TIMEOUT_S,
                )
                stored = True
            except TimeoutError:
                _metrics["db_timeouts"] += 1
                _metrics["store_failures"] += 1
                _set_backend("memory_fallback", False)
                logger.warning("User memory pgvector store timed out; switched to in-process fallback.")
            except Exception as e:
                _metrics["store_failures"] += 1
                _set_backend("memory_fallback", False)
                logger.warning(f"User memory pgvector store failed, using fallback: {_exc_label(e)}")

        if not stored and _backend == "mongodb" and _mongo_connected and _mongo_client:
            try:
                db = _mongo_client[DATABASE_NAME]
                coll = db["user_memories"]
                entity_key = candidate.get("entity_key")
                if entity_key:
                    modified = await _run_mongo_call(
                        lambda: coll.update_many(
                            {
                                "scope": scope,
                                "memory_type": candidate["memory_type"],
                                "entity_key": entity_key,
                                "active": True,
                                "text": {"$ne": text},
                            },
                            {"$set": {"active": False, "resolved_at": now, "updated_at": now}},
                        ).modified_count,
                    )
                    if modified:
                        _metrics["conflicts_resolved"] += int(modified)

                await _run_mongo_call(
                    lambda: coll.update_one(
                        {"scope": scope, "fingerprint": fingerprint},
                        {
                            "$set": {
                                "text": text,
                                "memory_type": candidate["memory_type"],
                                "entity_key": candidate.get("entity_key"),
                                "active": True,
                                "confidence": float(candidate["confidence"]),
                                "vector": vector,
                                "updated_at": now,
                                "resolved_at": None,
                            },
                            "$setOnInsert": {"created_at": now, "scope": scope},
                        },
                        upsert=True,
                    ),
                )
                stored = True
            except TimeoutError:
                _metrics["db_timeouts"] += 1
                _metrics["store_failures"] += 1
                _degrade_to_fallback("Mongo store timed out")
            except Exception as e:
                _metrics["store_failures"] += 1
                _degrade_to_fallback(f"Mongo store failed: {_exc_label(e)}")

        if stored:
            continue

        bucket = _fallback_memories.setdefault(scope, [])
        existing = None
        for idx, item in enumerate(bucket):
            if item.get("fingerprint") == fingerprint:
                existing = idx
                break
            entity_key = candidate.get("entity_key")
            if entity_key and item.get("entity_key") == entity_key and item.get("memory_type") == candidate["memory_type"]:
                if item.get("active", True) and item.get("text") != text:
                    item["active"] = False
                    item["resolved_at"] = now
                    item["updated_at"] = now
                    _metrics["conflicts_resolved"] += 1
        if existing is not None:
            bucket[existing] = doc
        else:
            bucket.append(doc)
        if len(bucket) > 200:
            del bucket[:-200]


def _rank_docs(query: str, docs: list[dict], limit: int) -> list[str]:
    now = datetime.now(timezone.utc)
    qvec = _vectorize(query)
    qtokens = set(_tokenize(query))

    scored: list[tuple[float, dict]] = []
    for doc in docs:
        if doc.get("active") is False:
            continue
        dvec = doc.get("vector") or []
        sim = _cosine(qvec, dvec)
        dtext = str(doc.get("text", ""))
        dtokens = set(_tokenize(dtext))
        lexical = 0.0
        if qtokens and dtokens:
            lexical = len(qtokens.intersection(dtokens)) / max(1, len(qtokens))

        updated = doc.get("updated_at")
        recency = 0.0
        if isinstance(updated, datetime):
            days = max(0.0, (now - updated).total_seconds() / 86400.0)
            recency = max(0.0, 1.0 - (days / _RECENCY_DAYS))

        confidence = float(doc.get("confidence", 0.5))
        score = (0.62 * sim) + (0.18 * lexical) + (0.12 * confidence) + (0.08 * recency)
        if score > 0.12:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    _metrics["rerank_calls"] += 1
    top = scored[: max(limit * 2, limit)]

    # Final rerank pass with query intent hints.
    boosted: list[tuple[float, str]] = []
    for rank, (score, item) in enumerate(top):
        text = str(item.get("text", ""))
        entity_key = str(item.get("entity_key") or "")
        mtype = str(item.get("memory_type") or "")

        bonus = 0.0
        if "prefer" in query.lower() and mtype == "preference":
            bonus += 0.08
        if "favorite" in query.lower() and entity_key.startswith("favorite_"):
            bonus += 0.1
        if "my name" in query.lower() and entity_key == "profile_name":
            bonus += 0.1
        if mtype == "explicit":
            bonus += 0.04

        # Slightly favor already higher-ranked candidates.
        bonus += max(0.0, 0.03 - (rank * 0.005))
        boosted.append((score + bonus, text))

    boosted.sort(key=lambda x: x[0], reverse=True)
    out = [text for _, text in boosted[:limit]]
    if out:
        return out

    # Similarity can be brittle for short follow-ups; fallback to recent active memories.
    recent = [d for d in docs if d.get("active") is not False]
    recent.sort(key=lambda d: d.get("updated_at") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return [str(d.get("text", "")) for d in recent[:limit] if str(d.get("text", "")).strip()]


def _rank_pg_rows(query: str, rows: list[dict], limit: int) -> list[str]:
    now = datetime.now(timezone.utc)
    qtokens = set(_tokenize(query))
    scored: list[tuple[float, str]] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        sim = float(row.get("similarity") or 0.0)
        confidence = float(row.get("confidence") or 0.5)
        updated = row.get("updated_at")
        mtype = str(row.get("memory_type") or "")
        entity_key = str(row.get("entity_key") or "")

        recency = 0.0
        if isinstance(updated, datetime):
            days = max(0.0, (now - updated).total_seconds() / 86400.0)
            recency = max(0.0, 1.0 - (days / _RECENCY_DAYS))

        # lexical overlap is injected by pre-computed token hint when available
        dtokens = set(_tokenize(text))
        lexical = 0.0
        if qtokens and dtokens:
            lexical = len(qtokens.intersection(dtokens)) / max(1, len(qtokens))
        score = (0.72 * sim) + (0.14 * lexical) + (0.08 * confidence) + (0.06 * recency)
        if "prefer" in query.lower() and mtype == "preference":
            score += 0.08
        if "favorite" in query.lower() and entity_key.startswith("favorite_"):
            score += 0.1
        if "my name" in query.lower() and entity_key == "profile_name":
            score += 0.1
        if mtype == "explicit":
            score += 0.04
        if score > 0.1:
            scored.append((score, text))

    scored.sort(key=lambda x: x[0], reverse=True)
    _metrics["rerank_calls"] += 1
    out = [text for _, text in scored[:limit]]
    if out:
        return out
    # pgvector already returns nearest neighbors ordered by distance.
    fallback: list[str] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if text:
            fallback.append(text)
        if len(fallback) >= limit:
            break
    return fallback


async def retrieve_user_memories(
    user_id: Optional[str],
    session_id: Optional[str],
    query: str,
    limit: int = 4,
) -> list[str]:
    _metrics["retrievals"] += 1
    scope = _scope_key(user_id, session_id)

    if _backend == "pgvector":
        try:
            qvec = _vectorize(query)
            loop = asyncio.get_running_loop()
            rows = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: _pg_fetch_memories_sync(scope, qvec, 120)),
                timeout=_DB_OP_TIMEOUT_S,
            )
            return _rank_pg_rows(query, rows, limit)
        except TimeoutError:
            _metrics["db_timeouts"] += 1
            _metrics["retrieval_failures"] += 1
            _set_backend("memory_fallback", False)
            logger.warning("User memory pgvector retrieval timed out; switched to in-process fallback.")
        except Exception as e:
            _metrics["retrieval_failures"] += 1
            _set_backend("memory_fallback", False)
            logger.warning(f"User memory pgvector retrieval failed, using fallback: {_exc_label(e)}")

    if _backend == "mongodb" and _mongo_connected and _mongo_client:
        try:
            db = _mongo_client[DATABASE_NAME]
            coll = db["user_memories"]
            docs = await _run_mongo_call(
                lambda: list(
                    coll.find({"scope": scope, "active": {"$ne": False}}, {"_id": 0})
                    .sort("updated_at", DESCENDING)
                    .limit(120)
                ),
            )
            return _rank_docs(query, docs, limit)
        except TimeoutError:
            _metrics["db_timeouts"] += 1
            _metrics["retrieval_failures"] += 1
            _degrade_to_fallback("Mongo retrieval timed out")
        except Exception as e:
            _metrics["retrieval_failures"] += 1
            _degrade_to_fallback(f"Mongo retrieval failed: {_exc_label(e)}")

    docs = _fallback_memories.get(scope, [])
    return _rank_docs(query, docs, limit)


async def get_user_memory_metrics() -> dict:
    return {
        "backend": _metrics["backend"],
        "connected": _metrics["connected"],
        "stores": _metrics["stores"],
        "retrievals": _metrics["retrievals"],
        "store_failures": _metrics["store_failures"],
        "retrieval_failures": _metrics["retrieval_failures"],
        "db_timeouts": _metrics["db_timeouts"],
        "conflicts_resolved": _metrics["conflicts_resolved"],
        "rerank_calls": _metrics["rerank_calls"],
    }
