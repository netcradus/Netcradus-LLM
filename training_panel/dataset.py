"""
Netcradus Training Panel — Dataset Management.

Provides upload, delete, list, and validation for training datasets.
Datasets are stored as plain text files in the data/processed directory.
"""

import os
import json
import time
import threading
from typing import Optional, Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DATASETS_DB = os.path.join(PROJECT_ROOT, "user_data", "datasets.json")

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB max per dataset file
ALLOWED_EXTENSIONS = (".txt", ".json", ".csv", ".tsv", ".parquet")


class DatasetManager:
    """Manages training datasets with file-based storage and metadata tracking."""

    def __init__(self):
        self._lock = threading.Lock()
        os.makedirs(DATASETS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(DATASETS_DB), exist_ok=True)
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        if os.path.exists(DATASETS_DB):
            try:
                with open(DATASETS_DB, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_metadata(self) -> None:
        with open(DATASETS_DB, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2)

    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets with metadata."""
        with self._lock:
            results = []
            if not os.path.isdir(DATASETS_DIR):
                return results
            for name in sorted(os.listdir(DATASETS_DIR)):
                path = os.path.join(DATASETS_DIR, name)
                if os.path.isfile(path):
                    ext = os.path.splitext(name)[1].lower()
                    if ext in ALLOWED_EXTENSIONS:
                        size = os.path.getsize(path)
                        results.append({
                            "name": name,
                            "size_bytes": size,
                            "size_mb": round(size / (1024 * 1024), 2),
                            "extension": ext,
                            "modified": time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))
                            ),
                            "lines": self._count_lines(path),
                        })
            return results

    def _count_lines(self, filepath: str) -> int:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def upload_dataset(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Upload a dataset file."""
        if not filename:
            raise ValueError("Filename is required")
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Invalid file extension '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        if len(content) > MAX_FILE_SIZE:
            raise ValueError(f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB")

        safe_name = os.path.basename(filename)
        path = os.path.join(DATASETS_DIR, safe_name)

        with self._lock:
            if os.path.exists(path):
                raise ValueError(f"Dataset '{safe_name}' already exists")
            with open(path, "wb") as f:
                f.write(content)
            self._metadata[safe_name] = {
                "uploaded_at": time.time(),
                "size_bytes": len(content),
            }
            self._save_metadata()

        return {
            "name": safe_name,
            "size_bytes": len(content),
            "size_mb": round(len(content) / (1024 * 1024), 2),
            "lines": self._count_lines(path),
        }

    def delete_dataset(self, filename: str) -> bool:
        """Delete a dataset file."""
        safe_name = os.path.basename(filename)
        path = os.path.join(DATASETS_DIR, safe_name)
        with self._lock:
            if not os.path.exists(path):
                raise ValueError(f"Dataset '{safe_name}' not found")
            os.remove(path)
            self._metadata.pop(safe_name, None)
            self._save_metadata()
        return True

    def get_dataset(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get dataset metadata."""
        safe_name = os.path.basename(filename)
        path = os.path.join(DATASETS_DIR, safe_name)
        if not os.path.exists(path):
            return None
        return {
            "name": safe_name,
            "size_bytes": os.path.getsize(path),
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            "lines": self._count_lines(path),
        }

    def validate_dataset(self, filename: str) -> Dict[str, Any]:
        """Validate a dataset file and return statistics."""
        safe_name = os.path.basename(filename)
        path = os.path.join(DATASETS_DIR, safe_name)
        if not os.path.exists(path):
            raise ValueError(f"Dataset '{safe_name}' not found")

        line_count = 0
        empty_lines = 0
        total_chars = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_count += 1
                stripped = line.strip()
                if not stripped:
                    empty_lines += 1
                total_chars += len(stripped)

        return {
            "name": safe_name,
            "total_lines": line_count,
            "empty_lines": empty_lines,
            "valid_lines": line_count - empty_lines,
            "total_chars": total_chars,
            "avg_line_length": round(total_chars / max(line_count, 1), 1),
        }