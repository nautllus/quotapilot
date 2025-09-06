import os
import logging
from datetime import datetime
from typing import Optional, Tuple
from urllib.parse import urlparse

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


def _get_db_name_from_uri(uri: str) -> str:
    # Try to parse db name from URI path; fallback to env or default
    parsed = urlparse(uri)
    # path like "/quotapilot" or possibly empty
    if parsed.path and len(parsed.path) > 1:
        return parsed.path.lstrip("/")
    return os.getenv("MONGODB_DB", "quotapilot")


def get_client() -> AsyncIOMotorClient:
    if _client is None:
        raise RuntimeError("Mongo client is not initialized. Call init_mongo() first.")
    return _client


def get_db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Mongo DB not initialized. Call init_mongo() first.")
    return _db


async def init_mongo() -> Tuple[AsyncIOMotorClient, AsyncIOMotorDatabase]:
    """Initialize Mongo connection and ensure indexes.

    Reads MONGODB_URI and optional pool tuning from environment.
    """
    global _client, _db

    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/quotapilot")
    max_pool = int(os.getenv("MONGODB_MAX_POOL_SIZE", "100"))
    min_pool = int(os.getenv("MONGODB_MIN_POOL_SIZE", "0"))
    connect_timeout_ms = int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "10000"))
    socket_timeout_ms = int(os.getenv("MONGODB_SOCKET_TIMEOUT_MS", "20000"))
    wait_queue_timeout_ms = int(os.getenv("MONGODB_WAIT_QUEUE_TIMEOUT_MS", "0"))

    try:
        _client = AsyncIOMotorClient(
            uri,
            maxPoolSize=max_pool,
            minPoolSize=min_pool,
            connectTimeoutMS=connect_timeout_ms,
            socketTimeoutMS=socket_timeout_ms,
            waitQueueTimeoutMS=wait_queue_timeout_ms,
        )
        db_name = _get_db_name_from_uri(uri)
        _db = _client[db_name]

        # simple connectivity check (non-fatal if fails)
        try:
            await _db.command("ping")
            logger.info("Connected to MongoDB database '%s'", db_name)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("MongoDB ping failed: %s", e)

        await _ensure_indexes(_db)
        return _client, _db
    except Exception as e:
        logger.exception("Failed to initialize MongoDB: %s", e)
        raise


async def close_mongo() -> None:
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None


async def _ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    col = db["usage_logs"]
    try:
        await col.create_index([("provider", 1), ("model", 1), ("ts", -1)], name="provider_model_ts_v1")
        await col.create_index([("ts", -1)], name="ts_desc_v1")
        logger.info("MongoDB indexes ensured for usage_logs")
    except Exception as e:  # pragma: no cover - index creation failures should not crash
        logger.warning("Failed to ensure MongoDB indexes: %s", e)

