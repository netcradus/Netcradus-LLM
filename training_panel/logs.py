"""
Netcradus Training Panel — Log Management.

Provides live log streaming, training history, and error log filtering.
"""

import os
import time
import threading
import logging
from collections import deque
from typing import Optional, Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(PROJECT_ROOT, "admin_data", "server.log")
TRAINING_LOG_FILE = os.path.join(PROJECT_ROOT, "admin_data", "training.log")
MAX_LOG_ENTRIES = 10000


class LogManager:
    """Manages server logs with filtering and streaming support."""

    def __init__(self):
        self._lock = threading.Lock()
        self._buffer = deque(maxlen=MAX_LOG_ENTRIES)
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    def add_entry(self, level: str, message: str, logger_name: str = "training") -> None:
        """Add a log entry to the buffer and file."""
        entry = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "level": level,
            "logger": logger_name,
            "message": message,
        }
        with self._lock:
            self._buffer.append(entry)
        try:
            log_file = TRAINING_LOG_FILE if logger_name == "training" else LOG_FILE
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{entry['time']}] [{entry['level']}] {entry['logger']}: {message}\n")
        except OSError:
            pass

    def get_logs(
        self,
        level: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get filtered logs with pagination."""
        with self._lock:
            logs = list(self._buffer)

        if level:
            logs = [l for l in logs if l["level"] == level.upper()]
        if search:
            q = search.lower()
            logs = [l for l in logs if q in l["message"].lower() or q in l["logger"].lower()]

        total = len(logs)
        logs = logs[offset:offset + limit]

        return {
            "logs": logs,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
        }

    def get_errors(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get only error and critical log entries."""
        return self.get_logs(level="ERROR", limit=limit)["logs"]

    def get_training_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get training-related log entries."""
        with self._lock:
            logs = list(self._buffer)
        training_logs = [
            l for l in logs
            if l["logger"] in ("training", "netcradus_trainer")
        ]
        return training_logs[-limit:]

    def clear_logs(self) -> None:
        """Clear all buffered logs."""
        with self._lock:
            self._buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get log statistics."""
        with self._lock:
            logs = list(self._buffer)
        total = len(logs)
        by_level = {}
        for l in logs:
            level = l["level"]
            by_level[level] = by_level.get(level, 0) + 1
        return {
            "total_entries": total,
            "by_level": by_level,
            "oldest": logs[0]["time"] if logs else None,
            "newest": logs[-1]["time"] if logs else None,
        }

    def stream_logs(
        self,
        since_time: Optional[str] = None,
        level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get logs since a given timestamp for live streaming."""
        with self._lock:
            logs = list(self._buffer)
        if since_time:
            logs = [l for l in logs if l["time"] > since_time]
        if level:
            logs = [l for l in logs if l["level"] == level.upper()]
        return logs