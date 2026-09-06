from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Optional

from .paths import get_app_base_dir


class TranslationCache:
    """Thread-safe persistent cache for translated text chunks using SQLite."""

    def __init__(self, db_path: Path | None = None) -> None:
        if db_path is None:
            self.db_path = get_app_base_dir() / ".translation_cache.db"
        else:
            self.db_path = Path(db_path)

        self._lock = Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=15.0,
            )
            # Enable WAL mode for high concurrency read/write performance
            try:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
            except sqlite3.Error:
                pass
        return self._conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT,
                    source_text TEXT,
                    translated_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_cache_model ON chunk_cache(model);"
            )
            conn.commit()

    @staticmethod
    def make_cache_key(model: str, prefix: str, text: str) -> str:
        """Computes a deterministic SHA-256 hash for a given model, namespace prefix, and text."""
        raw = f"{model.strip()}:::{prefix}:::{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, cache_key: str) -> str | None:
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT translated_text FROM chunk_cache WHERE cache_key = ? LIMIT 1;",
                    (cache_key,),
                )
                row = cursor.fetchone()
                return row[0] if row is not None else None
            except sqlite3.Error:
                return None

    def set(
        self, cache_key: str, model: str, source_text: str, translated_text: str
    ) -> None:
        with self._lock:
            try:
                conn = self._get_connection()
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunk_cache
                    (cache_key, model, source_text, translated_text, created_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP);
                    """,
                    (cache_key, model, source_text, translated_text),
                )
                conn.commit()
            except sqlite3.Error:
                pass

    def get_stats(self) -> tuple[int, int]:
        """Returns (item_count, file_size_bytes)."""
        with self._lock:
            count = 0
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunk_cache;")
                row = cursor.fetchone()
                count = row[0] if row else 0
            except sqlite3.Error:
                count = 0

            size = 0
            if self.db_path.exists():
                try:
                    size = self.db_path.stat().st_size
                except OSError:
                    size = 0
            return count, size

    def clear(self) -> int:
        """Clears all cached translations and returns the count of deleted items."""
        with self._lock:
            deleted = 0
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chunk_cache;")
                row = cursor.fetchone()
                deleted = row[0] if row else 0
                conn.execute("DELETE FROM chunk_cache;")
                conn.commit()
                conn.execute("VACUUM;")
            except sqlite3.Error:
                pass
            return deleted

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None


_global_cache: TranslationCache | None = None
_global_cache_lock = Lock()


def get_translation_cache() -> TranslationCache:
    global _global_cache
    with _global_cache_lock:
        if _global_cache is None:
            _global_cache = TranslationCache()
        return _global_cache
