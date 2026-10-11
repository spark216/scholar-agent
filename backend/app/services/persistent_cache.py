"""Small versioned SQLite cache for repeatable evaluation requests."""

from __future__ import annotations

from contextlib import closing

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Any

from backend.app.config import ROOT_DIR, get_settings


_SCHEMA_VERSION = "scholaragent-search-cache-v1"
_LOCK = RLock()


class FrozenProviderCacheMiss(RuntimeError):
    """Raised when a frozen benchmark would otherwise contact the network."""


def cache_payload_sha256(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cache_path() -> Path:
    configured = get_settings().academic_provider_cache_path
    if configured:
        return Path(configured).expanduser()
    return ROOT_DIR / "backend" / "paper-test" / "cache" / "search_cache.sqlite3"


def _cache_key(namespace: str, payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(
        f"{_SCHEMA_VERSION}\x1f{namespace}\x1f{serialized}".encode("utf-8")
    ).hexdigest()


def _connect() -> sqlite3.Connection:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS search_cache ("
        "namespace TEXT NOT NULL, cache_key TEXT NOT NULL, payload TEXT NOT NULL, "
        "created_at REAL NOT NULL, PRIMARY KEY(namespace, cache_key))"
    )
    return connection


def load_cached_json(
    namespace: str,
    key_payload: Any,
    *,
    ttl_seconds: int,
    allow_expired: bool = False,
) -> Any | None:
    if ttl_seconds <= 0:
        return None
    key = _cache_key(namespace, key_payload)
    with _LOCK, closing(_connect()) as connection, connection:
        row = connection.execute(
            "SELECT payload, created_at FROM search_cache "
            "WHERE namespace = ? AND cache_key = ?",
            (namespace, key),
        ).fetchone()
        if row is None:
            return None
        if not allow_expired and time.time() - float(row[1]) > ttl_seconds:
            connection.execute(
                "DELETE FROM search_cache WHERE namespace = ? AND cache_key = ?",
                (namespace, key),
            )
            return None
        try:
            return json.loads(str(row[0]))
        except json.JSONDecodeError:
            connection.execute(
                "DELETE FROM search_cache WHERE namespace = ? AND cache_key = ?",
                (namespace, key),
            )
            return None


def save_cached_json(namespace: str, key_payload: Any, value: Any) -> None:
    key = _cache_key(namespace, key_payload)
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    with _LOCK, closing(_connect()) as connection, connection:
        connection.execute(
            "INSERT INTO search_cache(namespace, cache_key, payload, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(namespace, cache_key) DO UPDATE SET "
            "payload = excluded.payload, created_at = excluded.created_at",
            (namespace, key, serialized, time.time()),
        )
