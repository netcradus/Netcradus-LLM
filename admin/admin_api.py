"""
Netcradus Admin - API Router & Business Logic.

===============================================================================
PURPOSE:
===============================================================================
Implements all admin-panel API endpoints consumed by web/admin.js:
  /api/auth/login | logout | me
  /api/admin/dashboard | users | llm | checkpoints | settings | training | logs

User Management APIs:
  GET    /api/admin/users              - List all users
  GET    /api/admin/users/<username>   - Get user profile
  POST   /api/admin/users              - Create user
  PUT    /api/admin/users/<username>   - Update user (role/password)
  POST   /api/admin/users/<username>/password - Change user password
  DELETE /api/admin/users/<username>   - Delete user

Every /api/admin/* route requires a valid session token AND the "admin" role.
===============================================================================
"""

import os
import json
import time
import logging
import threading
from collections import deque
from typing import Optional, Dict, Any, Tuple

from admin.auth import UserStore, SessionManager
from admin.training import TrainingJobManager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE = os.path.join(PROJECT_ROOT, "admin_data", "model_settings.json")
LOG_FILE = os.path.join(PROJECT_ROOT, "admin_data", "server.log")

DEFAULT_MODEL_SETTINGS = {
    "default_temperature": 0.7,
    "default_max_tokens": 512,
    "default_top_k": 50,
    "default_top_p": 0.9,
    "default_persona": "general",
    "max_tokens_limit": 2048,
    "stream_enabled": True,
}


class BufferLogHandler(logging.Handler):

    def __init__(self, maxlen: int = 1000):
        super().__init__()
        self.records = deque(maxlen=maxlen)
        # NOTE: must NOT reuse `self.lock` — logging.Handler uses self.lock
        # (a reentrant RLock, acquired in Handler.handle() before emit()) as an
        # internal guard. Reusing it here with a non-reentrant threading.Lock
        # caused a deadlock: handle() acquires it, then emit()'s `with
        # self.lock:` tried to re-acquire the same plain Lock on one thread.
        self._records_lock = threading.Lock()
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }
        with self._records_lock:
            self.records.append(entry)
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{entry['time']}] [{entry['level']}] {entry['logger']}: {message}\n")
        except OSError:
            pass


class AdminAPI:

    def __init__(self):
        self.users = UserStore()
        self.sessions = SessionManager()
        self.training = TrainingJobManager()
        self.settings = self._load_settings()

        self.log_handler = BufferLogHandler(maxlen=1000)
        self.log_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(self.log_handler)

        self._loaded_checkpoint: Optional[str] = None
        self._requests_served = 0
        self._boot_time = time.time()
        self._startup_note = "Admin panel ready. Default login: admin / admin@netcradus2026"

    def _load_settings(self) -> Dict[str, Any]:
        defaults = dict(DEFAULT_MODEL_SETTINGS)
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                defaults.update({k: v for k, v in data.items() if k in defaults})
            except (json.JSONDecodeError, OSError):
                pass
        return defaults

    def _save_settings(self) -> None:
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.settings, f, indent=2)

    def get_setting(self, key: str, default=None):
        return self.settings.get(key, default)

    def handle(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]],
        headers: Dict[str, str],
    ) -> Tuple[int, Dict[str, Any]]:
        segs = [s for s in path.split("/") if s]
        if len(segs) < 2 or segs[0] != "api":
            return 404, {"error": "Not found"}

        token = self._bearer_token(headers)

        if segs[1] == "auth":
            if len(segs) == 3 and segs[2] == "login" and method == "POST":
                return self._login(body or {})
            user = self.sessions.validate(token)
            if user is None:
                return 401, {"error": "Not authenticated"}
            if len(segs) == 3 and segs[2] == "logout" and method == "POST":
                self.sessions.revoke(token)
                return 200, {"success": True}
            if len(segs) == 3 and segs[2] == "me" and method == "GET":
                return 200, {"user": user}
            return 404, {"error": "Not found"}

        if segs[1] != "admin":
            return 404, {"error": "Not found"}

        user = self.sessions.validate(token)
        if user is None:
            return 401, {"error": "Not authenticated"}
        if user.get("role") != "admin":
            return 403, {"error": "Admin privileges required"}

        route = segs[2] if len(segs) > 2 else ""
        sub = segs[3] if len(segs) > 3 else None

        if route == "dashboard" and method == "GET":
            return self._dashboard()

        # Users: check specific sub-routes before the generic list/create branches
        # so GET /users/<name> and POST /users/<name>/password are not shadowed.
        if route == "users" and sub and len(segs) > 4 and segs[4] == "password" and method == "POST":
            return self._change_password(sub, body or {})
        if route == "users" and sub and method == "GET":
            return self._get_user(sub)
        if route == "users" and sub and method == "PUT":
            return self._update_user(sub, body or {})
        if route == "users" and sub and method == "DELETE":
            return self._delete_user(sub)
        if route == "users" and method == "POST":
            return self._create_user(body or {})
        if route == "users" and method == "GET":
            return 200, {"users": self.users.list_users()}

        if route == "llm" and method == "GET":
            return self._llm_info()
        if route == "llm" and sub == "load" and method == "POST":
            return self._load_checkpoint((body or {}).get("checkpoint"))
        if route == "llm" and sub == "unload" and method == "POST":
            return self._unload_model()
        if route == "checkpoints" and method == "GET":
            return 200, {"checkpoints": self._list_checkpoints()}
        if route == "checkpoints" and sub and method == "DELETE":
            return self._delete_checkpoint(sub)

        if route == "settings" and method == "GET":
            return 200, {"settings": self.settings}
        if route == "settings" and method == "POST":
            return self._update_settings(body or {})

        if route == "training" and sub == "status" and method == "GET":
            return 200, self.training.status()
        if route == "training" and sub == "start" and method == "POST":
            return self._training_start(body or {})
        if route == "training" and sub == "stop" and method == "POST":
            return self._training_stop()

        if route == "logs" and method == "GET":
            return 200, {"logs": list(self.log_handler.records)}
        if route == "logs" and method == "DELETE":
            self.log_handler.records.clear()
            return 200, {"success": True}

        return 404, {"error": "Not found"}

    @staticmethod
    def _bearer_token(headers: Dict[str, str]) -> Optional[str]:
        auth = headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth[len("Bearer "):].strip()
        return None

    def _login(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        username = str(body.get("username", "")).strip()
        password = str(body.get("password", ""))
        user = self.users.authenticate(username, password)
        if user is None:
            return 401, {"error": "Invalid username or password"}
        token = self.sessions.create(user["username"], user["role"])
        return 200, {"token": token, "user": user}

    def _dashboard(self) -> Tuple[int, Dict[str, Any]]:
        pipeline = self._pipeline()
        model = pipeline.model if pipeline else None
        return 200, {
            "uptime": round(time.time() - self._boot_time, 1),
            "requests": self._requests_served,
            "users": self.users.list_users(),
            "device": self._device(),
            "pipeline_loaded": pipeline is not None,
            "model_name": "Netcradus LLM",
            "model_config": (
                {k: v for k, v in model.config.to_dict().items() if not isinstance(v, dict)}
                if model is not None else None
            ),
            "params": sum(p.numel() for p in model.parameters()) if model is not None else 0,
            "loaded_checkpoint": self._loaded_checkpoint,
            "training_state": self.training.status().get("state"),
            "vocab_size": pipeline.tokenizer.vocab_size if pipeline else 32000,
            "checkpoints": len(self._list_checkpoints()),
            "startup_note": self._startup_note,
        }

    def _create_user(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        try:
            user = self.users.create_user(
                str(body.get("username", "")).strip(),
                str(body.get("password", "")),
                str(body.get("role", "user")),
            )
            return 200, {"success": True, "user": user}
        except ValueError as exc:
            return 400, {"error": str(exc)}

    def _update_user(self, username: str, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        try:
            user = self.users.update_user(
                username,
                password=body.get("password") or None,
                role=body.get("role"),
            )
            return 200, {"success": True, "user": user}
        except ValueError as exc:
            return 400, {"error": str(exc)}

    def _delete_user(self, username: str) -> Tuple[int, Dict[str, Any]]:
        try:
            self.users.delete_user(username)
            return 200, {"success": True}
        except ValueError as exc:
            return 400, {"error": str(exc)}

    def _get_user(self, username: str) -> Tuple[int, Dict[str, Any]]:
        try:
            user = self.users.get_user(username)
            if user is None:
                return 404, {"error": f"User '{username}' not found"}
            return 200, {"user": {
                "username": user["username"],
                "role": user["role"],
                "created_at": user.get("created_at"),
            }}
        except ValueError as exc:
            return 400, {"error": str(exc)}

    def _change_password(self, username: str, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        try:
            new_password = str(body.get("password", ""))
            if not new_password:
                return 400, {"error": "New password is required"}
            self.users.update_user(username, password=new_password)
            return 200, {"success": True, "message": f"Password updated for '{username}'"}
        except ValueError as exc:
            return 400, {"error": str(exc)}

    @staticmethod
    def _pipeline():
        try:
            import web_server
            return web_server.PIPELINE
        except Exception:
            return None

    @staticmethod
    def _device() -> str:
        try:
            import web_server
            return web_server.DEVICE
        except Exception:
            return "cpu"

    def _set_pipeline(self, pipeline) -> None:
        import web_server
        web_server.PIPELINE = pipeline

    def _llm_info(self) -> Tuple[int, Dict[str, Any]]:
        pipeline = self._pipeline()
        info = {
            "pipeline_loaded": pipeline is not None,
            "device": self._device(),
            "loaded_checkpoint": self._loaded_checkpoint,
            "status": "online" if pipeline is not None else "fallback",
        }
        if pipeline is not None:
            model = pipeline.model
            config = model.config.to_dict()
            info["model_name"] = "Netcradus LLM"
            info["vocab_size"] = pipeline.tokenizer.vocab_size
            info["architecture"] = "SwiGLU + GQA + RoPE"
            info["config"] = config
            info["params"] = sum(p.numel() for p in model.parameters())
        return 200, info

    def _checkpoint_dir(self) -> str:
        return os.path.join(PROJECT_ROOT, "checkpoints_demo")

    def _list_checkpoints(self) -> list:
        directory = self._checkpoint_dir()
        if not os.path.isdir(directory):
            return []
        results = []
        for name in sorted(os.listdir(directory)):
            if name.endswith(".pt"):
                path = os.path.join(directory, name)
                results.append({
                    "name": name,
                    "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
                    "modified": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))
                    ),
                })
        return results

    def _load_checkpoint(self, name: Optional[str]) -> Tuple[int, Dict[str, Any]]:
        if not name:
            return 400, {"error": "checkpoint name is required"}
        path = os.path.join(self._checkpoint_dir(), os.path.basename(name))
        if not os.path.exists(path):
            return 404, {"error": f"Checkpoint '{name}' not found"}
        try:
            import torch

            from netcradus_llm.config import NetcradusConfig, RUNTIME_CONFIG
            from netcradus_llm.model import NetcradusForCausalLM
            from netcradus_llm.tokenizer import NetcradusTokenizer
            from netcradus_llm.inference import NetcradusPipeline

            checkpoint = torch.load(path, map_location=self._device())
            config_dict = checkpoint.get("config") if isinstance(checkpoint, dict) else None
            if config_dict:
                config = NetcradusConfig.from_dict(config_dict)
            else:
                # No config in checkpoint: fall back to the small runtime config
                # (NOT the multi-GB architectural defaults, which would OOM).
                config = NetcradusConfig(**RUNTIME_CONFIG.to_dict())
            model = NetcradusForCausalLM(config)
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
            tokenizer = NetcradusTokenizer(vocab_size=config.vocab_size)
            self._set_pipeline(NetcradusPipeline(model=model, tokenizer=tokenizer, device=self._device()))
            self._loaded_checkpoint = name
            return 200, {
                "success": True,
                "message": f"Loaded '{name}' into inference pipeline.",
                "vocab_size": config.vocab_size,
            }
        except Exception as exc:
            return 500, {"error": f"Failed to load checkpoint: {exc}"}

    def _unload_model(self) -> Tuple[int, Dict[str, Any]]:
        self._set_pipeline(None)
        self._loaded_checkpoint = None
        return 200, {"success": True, "message": "Model unloaded. Server running in fallback mode."}

    def _delete_checkpoint(self, name: str) -> Tuple[int, Dict[str, Any]]:
        path = os.path.join(self._checkpoint_dir(), os.path.basename(name))
        if not os.path.exists(path):
            return 404, {"error": f"Checkpoint '{name}' not found"}
        os.remove(path)
        if self._loaded_checkpoint == name:
            self._loaded_checkpoint = None
        return 200, {"success": True, "message": f"Deleted checkpoint '{name}'."}

    def _update_settings(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        allowed = set(DEFAULT_MODEL_SETTINGS.keys())
        try:
            for key, value in body.items():
                if key not in allowed:
                    continue
                if key == "default_persona":
                    self.settings[key] = str(value)
                elif key == "stream_enabled":
                    self.settings[key] = bool(value)
                else:
                    self.settings[key] = max(0.0, float(value))
            self._save_settings()
            return 200, {"success": True, "settings": self.settings}
        except (TypeError, ValueError) as exc:
            return 400, {"error": f"Invalid setting value: {exc}"}

    def _training_start(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        try:
            self.training.start(body)
            return 200, self.training.status()
        except RuntimeError as exc:
            return 400, {"error": str(exc)}

    def _training_stop(self) -> Tuple[int, Dict[str, Any]]:
        try:
            return 200, self.training.stop()
        except RuntimeError as exc:
            return 400, {"error": str(exc)}

    def count_request(self) -> None:
        self._requests_served += 1