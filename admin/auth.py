"""
Netcradus Admin - Authentication & User Store.

===============================================================================
PURPOSE:
===============================================================================
Provides secure authentication for the admin panel:
- PBKDF2-HMAC-SHA256 salted password hashing (no external dependencies).
- Session token management (in-memory, expiring, bearer-token based).
- Persistent JSON user store with role assignment (admin / user).

===============================================================================
SECURITY NOTES:
===============================================================================
- Passwords are never stored in plaintext (salted + hashed only).
- Constant-time comparison via hmac.compare_digest.
- Tokens are cryptographically random (secrets.token_urlsafe) and expire.
- Default admin account is seeded on first run; change it after deployment.
===============================================================================
"""

import os
import json
import time
import hmac
import hashlib
import secrets
import threading
from typing import List, Dict, Optional, Any

ADMIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "admin_data")
USERS_FILE = os.path.join(ADMIN_DIR, "users.json")
SESSION_TTL = 8 * 60 * 60

DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASSWORD = "admin@netcradus2026"

_ROLES = ("admin", "user")


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


class UserStore:

    def __init__(self, users_file: str = USERS_FILE):
        self._file = users_file
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self._file), exist_ok=True)
        if not os.path.exists(self._file):
            self._seed()

    def _seed(self) -> None:
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump({
                "users": [{
                    "username": DEFAULT_ADMIN_USER,
                    "password_hash": hash_password(DEFAULT_ADMIN_PASSWORD),
                    "role": "admin",
                    "created_at": time.time(),
                }]
            }, f, indent=2)

    def _load(self) -> List[Dict[str, Any]]:
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("users", [])
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            return []

    def _save(self, users: List[Dict[str, Any]]) -> None:
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump({"users": users}, f, indent=2)

    def list_users(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "username": u["username"],
                    "role": u["role"],
                    "created_at": u.get("created_at"),
                }
                for u in self._load()
            ]

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for u in self._load():
                if u["username"] == username:
                    return u
        return None

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        user = self.get_user(username)
        if user and verify_password(password, user.get("password_hash", "")):
            return {
                "username": user["username"],
                "role": user["role"],
                "created_at": user.get("created_at"),
            }
        return None

    def create_user(self, username: str, password: str, role: str = "user") -> Dict[str, Any]:
        if role not in _ROLES:
            raise ValueError(f"Invalid role '{role}'. Must be one of {_ROLES}.")
        if not username or not password:
            raise ValueError("Username and password are required.")
        with self._lock:
            users = self._load()
            if any(u["username"] == username for u in users):
                raise ValueError(f"User '{username}' already exists.")
            users.append({
                "username": username,
                "password_hash": hash_password(password),
                "role": role,
                "created_at": time.time(),
            })
            self._save(users)
        return {"username": username, "role": role}

    def update_user(self, username: str, password: Optional[str] = None, role: Optional[str] = None) -> Dict[str, Any]:
        if role is not None and role not in _ROLES:
            raise ValueError(f"Invalid role '{role}'.")
        with self._lock:
            users = self._load()
            target = next((u for u in users if u["username"] == username), None)
            if target is None:
                raise ValueError(f"User '{username}' not found.")
            if password:
                target["password_hash"] = hash_password(password)
            if role is not None:
                target["role"] = role
            self._save(users)
        return {"username": username, "role": target["role"]}

    def delete_user(self, username: str) -> None:
        with self._lock:
            users = self._load()
            remaining = [u for u in users if u["username"] != username]
            if len(remaining) == len(users):
                raise ValueError(f"User '{username}' not found.")
            if not any(u["role"] == "admin" for u in remaining):
                raise ValueError("Cannot delete the last admin account.")
            self._save(remaining)


class SessionManager:

    def __init__(self, ttl: int = SESSION_TTL):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl
        self._lock = threading.Lock()

    def create(self, username: str, role: str) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = {
                "username": username,
                "role": role,
                "expires": time.time() + self._ttl,
            }
        return token

    def validate(self, token: Optional[str]) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        with self._lock:
            sess = self._sessions.get(token)
            if sess is None:
                return None
            if time.time() > sess["expires"]:
                self._sessions.pop(token, None)
                return None
            return {"username": sess["username"], "role": sess["role"]}

    def revoke(self, token: Optional[str]) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)