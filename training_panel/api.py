"""
Netcradus Training Panel — API Router & Handlers.

Implements all training panel API endpoints:
  /api/training/datasets  - Dataset CRUD
  /api/training/train     - Training control (start/stop/pause/resume/status)
  /api/training/checkpoints - Checkpoint CRUD
  /api/training/logs      - Log viewing and management
"""

import os
import json
import time
import threading
from typing import Optional, Dict, Any, Tuple

from training_panel.dataset import DatasetManager
from training_panel.trainer import TrainingJobManager
from training_panel.models import CheckpointManager
from training_panel.logs import LogManager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TrainingAPI:
    """Central router + handlers for the training panel."""

    def __init__(self):
        self.datasets = DatasetManager()
        self.trainer = TrainingJobManager()
        self.checkpoints = CheckpointManager()
        self.logs = LogManager()

    def handle(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]],
        headers: Dict[str, str],
    ) -> Tuple[int, Dict[str, Any]]:
        segs = [s for s in path.split("/") if s]
        if len(segs) < 2 or segs[0] != "api" or segs[1] != "training":
            return 404, {"error": "Not found"}

        route = segs[2] if len(segs) > 2 else ""
        sub = segs[3] if len(segs) > 3 else None
        sub2 = segs[4] if len(segs) > 4 else None

        # Dataset management (specific sub-routes before generic list/upload)
        if route == "datasets":
            if sub and sub2 == "validate" and method == "POST":
                return self._validate_dataset(sub)
            if sub and method == "DELETE":
                return self._delete_dataset(sub)
            if sub and method == "GET":
                return self._get_dataset(sub)
            if method == "POST":
                return self._upload_dataset(body or {})
            if method == "GET":
                return 200, {"datasets": self.datasets.list_datasets()}

        # Training control
        if route == "train":
            if sub == "status" and method == "GET":
                self.trainer.update_resource_usage()
                return 200, self.trainer.status()
            if sub == "start" and method == "POST":
                return self._start_training(body or {})
            if sub == "stop" and method == "POST":
                return self._stop_training()
            if sub == "pause" and method == "POST":
                return self._pause_training()
            if sub == "resume" and method == "POST":
                return self._resume_training()
            if sub == "history" and method == "GET":
                return 200, {"history": self.trainer.get_history()}
            if sub == "loss-graph" and method == "GET":
                return 200, {"loss_data": self.trainer.get_loss_data()}

        # Checkpoint management (specific sub-routes before generic list/save)
        if route == "checkpoints":
            if sub and sub2 == "load" and method == "POST":
                return self._load_checkpoint(sub)
            if sub and sub2 == "export" and method == "POST":
                fmt = body.get("format", "safetensors") if body else "safetensors"
                return self._export_checkpoint(sub, fmt)
            if sub and method == "DELETE":
                return self._delete_checkpoint(sub)
            if sub and method == "GET":
                return self._get_checkpoint_info(sub)
            if method == "POST":
                return self._save_checkpoint(body or {})
            if method == "GET":
                return 200, {"checkpoints": self.checkpoints.list_checkpoints()}

        # Logs (specific sub-routes before generic get/clear)
        if route == "logs":
            if sub == "errors" and method == "GET":
                return 200, {"errors": self.logs.get_errors()}
            if sub == "stats" and method == "GET":
                return 200, {"stats": self.logs.get_stats()}
            if method == "DELETE":
                self.logs.clear_logs()
                return 200, {"success": True}
            if method == "POST":
                # Filtered log view (training.js POSTs {level, search, limit, offset}).
                return self._get_logs(body or {})
            if method == "GET":
                return self._get_logs(body or {})

        return 404, {"error": "Not found"}

    # ---------------------------------------------------------- datasets
    def _upload_dataset(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        try:
            filename = body.get("filename", "")
            content = body.get("content", "")
            if not filename:
                return 400, {"error": "Filename is required"}
            if isinstance(content, str):
                content = content.encode("utf-8")
            result = self.datasets.upload_dataset(filename, content)
            return 201, result
        except ValueError as exc:
            return 400, {"error": str(exc)}

    def _delete_dataset(self, filename: str) -> Tuple[int, Dict[str, Any]]:
        try:
            self.datasets.delete_dataset(filename)
            return 200, {"success": True, "message": f"Deleted dataset '{filename}'"}
        except ValueError as exc:
            return 400, {"error": str(exc)}

    def _get_dataset(self, filename: str) -> Tuple[int, Dict[str, Any]]:
        result = self.datasets.get_dataset(filename)
        if result is None:
            return 404, {"error": f"Dataset '{filename}' not found"}
        return 200, result

    def _validate_dataset(self, filename: str) -> Tuple[int, Dict[str, Any]]:
        try:
            result = self.datasets.validate_dataset(filename)
            return 200, result
        except ValueError as exc:
            return 400, {"error": str(exc)}

    # ---------------------------------------------------------- training
    def _start_training(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        try:
            result = self.trainer.start(body)
            return 200, result
        except RuntimeError as exc:
            return 400, {"error": str(exc)}

    def _stop_training(self) -> Tuple[int, Dict[str, Any]]:
        try:
            return 200, self.trainer.stop()
        except RuntimeError as exc:
            return 400, {"error": str(exc)}

    def _pause_training(self) -> Tuple[int, Dict[str, Any]]:
        try:
            return 200, self.trainer.pause()
        except RuntimeError as exc:
            return 400, {"error": str(exc)}

    def _resume_training(self) -> Tuple[int, Dict[str, Any]]:
        try:
            return 200, self.trainer.resume()
        except RuntimeError as exc:
            return 400, {"error": str(exc)}

    # ---------------------------------------------------------- checkpoints
    def _save_checkpoint(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        name = body.get("name", f"checkpoint_{int(time.time())}")
        metadata = body.get("metadata")
        result = self.checkpoints.save_checkpoint(name, metadata)
        if result.get("success"):
            return 200, result
        return 500, result

    def _get_checkpoint_info(self, name: str) -> Tuple[int, Dict[str, Any]]:
        result = self.checkpoints.get_checkpoint_info(name)
        if result is None:
            return 404, {"error": f"Checkpoint '{name}' not found"}
        return 200, result

    def _delete_checkpoint(self, name: str) -> Tuple[int, Dict[str, Any]]:
        result = self.checkpoints.delete_checkpoint(name)
        if result.get("success"):
            return 200, result
        return 404, result

    def _load_checkpoint(self, name: str) -> Tuple[int, Dict[str, Any]]:
        result = self.checkpoints.load_checkpoint(name)
        if result.get("success"):
            return 200, result
        return 500, result

    def _export_checkpoint(self, name: str, fmt: str) -> Tuple[int, Dict[str, Any]]:
        result = self.checkpoints.export_checkpoint(name, fmt)
        if result.get("success"):
            return 200, result
        return 500, result

    # ------------------------------------------------------------- logs
    def _get_logs(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        level = body.get("level")
        search = body.get("search")
        limit = body.get("limit", 500)
        offset = body.get("offset", 0)
        return 200, self.logs.get_logs(
            level=level, search=search, limit=limit, offset=offset
        )