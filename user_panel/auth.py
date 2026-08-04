"""
Netcradus User Panel — Authentication & Session Management.

Provides:
- PBKDF2-HMAC-SHA256 salted password hashing.
- JWT-style bearer token management with expiry.
- Persistent session store backed by SQLite.
"""

import os
import hashlib
import hmac
import secrets
import time
from typing import Optional, Dict, Any

USER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "user_data")
USERS_DB = os.path.join(USER_DIR, "users.db")
SESSION_TTL = 8 * 60 * 60


def hash_password(password: str, salt: str = None) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000)
    return f"{salt}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    if not stored or "$" not in stored:
        return False
    salt, _ = stored.split("$", 1)
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored)


class UserAuth:
    """Password hashing and verification."""

    @staticmethod
    def hash_password(password: str) -> str:
        return hash_password(password)

    @staticmethod
    def verify_password(password: str, stored: str) -> bool:
        return verify_password(password, stored)


class SessionManager:
    """SQLite-backed session manager with token expiry."""

    def __init__(self, db_path: str = USERS_DB, ttl: int = SESSION_TTL):
        self._db_path = db_path
        self._ttl = ttl
        self._lock = __import__("threading").Lock()

    def create(self, username: str, role: str) -> str:
        token = secrets.token_urlsafe(32)
        from user_panel.database import UserDatabase
        db = UserDatabase(self._db_path)
        db.create_session(username, role, token, self._ttl)
        return token

    def validate(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        from user_panel.database import UserDatabase
        db = UserDatabase(self._db_path)
        session = db.validate_session(token)
        if session is None:
            return None
        return {"username": session["username"], "role": session["role"]}

    def revoke(self, token: Optional[str]) -> None:
        if not token:
            return
        from user_panel.database import UserDatabase
        db = UserDatabase(self._db_path)
        db.revoke_session(token)

    def revoke_all(self, username: str) -> None:
        from user_panel.database import UserDatabase
        db = UserDatabase(self._db_path)
        db.revoke_all_sessions(username)