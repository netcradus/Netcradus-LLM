"""
Netcradus User Panel — SQLite Database.

Provides persistent user storage with schema migration support.
"""

import os
import json
import time
import sqlite3
import threading
from typing import Optional, Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(PROJECT_ROOT, "user_data")
DB_FILE = os.path.join(DB_DIR, "users.db")

SCHEMA_VERSION = 1

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    display_name TEXT DEFAULT '',
    bio TEXT DEFAULT '',
    avatar_color TEXT DEFAULT '#8b5cf6',
    created_at REAL NOT NULL,
    last_login REAL DEFAULT 0
);
"""

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
"""

CREATE_CHAT_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    messages TEXT NOT NULL,
    persona TEXT DEFAULT 'general',
    created_at REAL NOT NULL
);
"""

CREATE_SETTINGS_TABLE = """
CREATE TABLE IF NOT EXISTS user_settings (
    username TEXT PRIMARY KEY,
    settings TEXT NOT NULL DEFAULT '{}'
);
"""


class UserDatabase:
    """SQLite-backed persistent user store."""

    def __init__(self, db_path: str = DB_FILE):
        self._db_path = db_path
        # Reentrant lock: create_user()/update_user() call self.get_user() while
        # already holding the lock. A plain threading.Lock would deadlock the
        # thread on the inner acquisition (same bug class as BufferLogHandler),
        # so use an RLock which is safe to acquire recursively from one thread.
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_USERS_TABLE)
            conn.execute(CREATE_SESSIONS_TABLE)
            conn.execute(CREATE_CHAT_HISTORY_TABLE)
            conn.execute(CREATE_SETTINGS_TABLE)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {k: row[k] for k in row.keys()}

    # ------------------------------------------------------------------ users
    def create_user(
        self,
        username: str,
        password_hash: str,
        role: str = "user",
        display_name: str = "",
    ) -> Dict[str, Any]:
        with self._lock:
            with self._connect() as conn:
                now = time.time()
                conn.execute(
                    "INSERT INTO users (username, password_hash, role, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
                    (username, password_hash, role, display_name, now),
                )
                conn.commit()
                return self.get_user(username)

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM users WHERE username = ?", (username,)
                ).fetchone()
                if row is None:
                    return None
                return self._row_to_dict(row)

    def authenticate(self, username: str, password_hash: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM users WHERE username = ? AND password_hash = ?",
                    (username, password_hash),
                ).fetchone()
                if row is None:
                    return None
                user = self._row_to_dict(row)
                conn.execute(
                    "UPDATE users SET last_login = ? WHERE username = ?",
                    (time.time(), username),
                )
                conn.commit()
                return user

    def list_users(self) -> List[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT id, username, role, display_name, created_at, last_login FROM users ORDER BY created_at DESC"
                ).fetchall()
                return [self._row_to_dict(r) for r in rows]

    def update_user(
        self,
        username: str,
        password_hash: Optional[str] = None,
        display_name: Optional[str] = None,
        bio: Optional[str] = None,
        avatar_color: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            with self._connect() as conn:
                user = self.get_user(username)
                if user is None:
                    raise ValueError(f"User '{username}' not found")
                fields = []
                params = []
                if password_hash is not None:
                    fields.append("password_hash = ?")
                    params.append(password_hash)
                if display_name is not None:
                    fields.append("display_name = ?")
                    params.append(display_name)
                if bio is not None:
                    fields.append("bio = ?")
                    params.append(bio)
                if avatar_color is not None:
                    fields.append("avatar_color = ?")
                    params.append(avatar_color)
                if fields:
                    params.append(username)
                    conn.execute(
                        f"UPDATE users SET {', '.join(fields)} WHERE username = ?",
                        params,
                    )
                    conn.commit()
            return self.get_user(username)

    def delete_user(self, username: str) -> None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT username FROM users WHERE username = ?", (username,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"User '{username}' not found")
                conn.execute("DELETE FROM users WHERE username = ?", (username,))
                conn.execute("DELETE FROM sessions WHERE username = ?", (username,))
                conn.execute("DELETE FROM chat_history WHERE username = ?", (username,))
                conn.execute("DELETE FROM user_settings WHERE username = ?", (username,))
                conn.commit()

    # ---------------------------------------------------------------- sessions
    def create_session(self, username: str, role: str, token: str, ttl: float) -> None:
        with self._lock:
            with self._connect() as conn:
                now = time.time()
                conn.execute(
                    "INSERT INTO sessions (token, username, role, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
                    (token, username, role, now, now + ttl),
                )
                conn.commit()

    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE token = ? AND expires_at > ?",
                    (token, time.time()),
                ).fetchone()
                if row is None:
                    return None
                return self._row_to_dict(row)

    def revoke_session(self, token: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
                conn.commit()

    def revoke_all_sessions(self, username: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM sessions WHERE username = ?", (username,))
                conn.commit()

    # ---------------------------------------------------------------- chat
    def save_chat_history(
        self, username: str, messages: list, persona: str = "general"
    ) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO chat_history (username, messages, persona, created_at) VALUES (?, ?, ?, ?)",
                    (username, json.dumps(messages), persona, time.time()),
                )
                conn.commit()

    def get_chat_history(
        self, username: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM chat_history WHERE username = ? ORDER BY created_at DESC LIMIT ?",
                    (username, limit),
                ).fetchall()
                return [
                    {**self._row_to_dict(r), "messages": json.loads(r["messages"])}
                    for r in rows
                ]

    def delete_chat_history(self, username: str, chat_id: int) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM chat_history WHERE id = ? AND username = ?",
                    (chat_id, username),
                )
                conn.commit()

    # ---------------------------------------------------------------- settings
    def get_settings(self, username: str) -> Dict[str, Any]:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT settings FROM user_settings WHERE username = ?", (username,)
                ).fetchone()
                if row is None:
                    return {}
                return json.loads(row["settings"])

    def save_settings(self, username: str, settings: Dict[str, Any]) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO user_settings (username, settings) VALUES (?, ?)",
                    (username, json.dumps(settings)),
                )
                conn.commit()