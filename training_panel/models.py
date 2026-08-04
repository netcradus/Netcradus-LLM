"""
Netcradus Training Panel — Checkpoint Management.

Provides save, load, delete, and export functionality for model checkpoints.
"""

import os
import json
import time
import shutil
import threading
from typing import Optional, Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_demo")
EXPORT_DIR = os.path.join(PROJECT_ROOT, "exports")


class CheckpointManager:
    """Manages model checkpoints with save, load, delete, and export."""

    def __init__(self):
        self._lock = threading.Lock()
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(EXPORT_DIR, exist_ok=True)

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        with self._lock:
            results = []
            if not os.path.isdir(CHECKPOINT_DIR):
                return results
            for name in sorted(os.listdir(CHECKPOINT_DIR)):
                if name.endswith(".pt"):
                    path = os.path.join(CHECKPOINT_DIR, name)
                    size = os.path.getsize(path)
                    results.append({
                        "name": name,
                        "size_bytes": size,
                        "size_mb": round(size / (1024 * 1024), 2),
                        "modified": time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))
                        ),
                    })
            return results

    def save_checkpoint(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Save a checkpoint with optional metadata."""
        safe_name = os.path.basename(name)
        if not safe_name.endswith(".pt"):
            safe_name += ".pt"
        path = os.path.join(CHECKPOINT_DIR, safe_name)

        with self._lock:
            checkpoint_data = {
                "config": {},
                "model_state_dict": {},
                "metadata": metadata or {},
                "saved_at": time.time(),
            }
            try:
                import torch
                from web_server import PIPELINE
                if PIPELINE is not None:
                    checkpoint_data["config"] = PIPELINE.model.config.to_dict()
                    checkpoint_data["model_state_dict"] = PIPELINE.model.state_dict()
                    checkpoint_data["vocab_size"] = PIPELINE.tokenizer.vocab_size
                else:
                    # No live model to snapshot: embed the runtime config so the
                    # saved checkpoint is loadable (state_dict stays empty).
                    from netcradus_llm.config import RUNTIME_CONFIG
                    checkpoint_data["config"] = RUNTIME_CONFIG.to_dict()
            except Exception:
                pass

            try:
                torch.save(checkpoint_data, path)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        return {
            "success": True,
            "name": safe_name,
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            "message": f"Checkpoint '{safe_name}' saved",
        }

    def load_checkpoint(self, name: str) -> Dict[str, Any]:
        """Load a checkpoint into the inference pipeline."""
        safe_name = os.path.basename(name)
        if not safe_name.endswith(".pt"):
            safe_name += ".pt"
        path = os.path.join(CHECKPOINT_DIR, safe_name)
        if not os.path.exists(path):
            return {"success": False, "error": f"Checkpoint '{safe_name}' not found"}

        try:
            import torch
            from netcradus_llm.config import NetcradusConfig, RUNTIME_CONFIG
            from netcradus_llm.model import NetcradusForCausalLM
            from netcradus_llm.tokenizer import NetcradusTokenizer
            from netcradus_llm.inference import NetcradusPipeline
            from web_server import PIPELINE, DEVICE

            checkpoint = torch.load(path, map_location=DEVICE)
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
            import web_server
            web_server.PIPELINE = NetcradusPipeline(
                model=model, tokenizer=tokenizer, device=DEVICE
            )
            return {
                "success": True,
                "message": f"Loaded '{safe_name}' into inference pipeline",
                "vocab_size": config.vocab_size,
            }
        except Exception as exc:
            return {"success": False, "error": f"Failed to load checkpoint: {exc}"}

    def delete_checkpoint(self, name: str) -> Dict[str, Any]:
        """Delete a checkpoint file."""
        safe_name = os.path.basename(name)
        if not safe_name.endswith(".pt"):
            safe_name += ".pt"
        path = os.path.join(CHECKPOINT_DIR, safe_name)
        with self._lock:
            if not os.path.exists(path):
                return {"success": False, "error": f"Checkpoint '{safe_name}' not found"}
            os.remove(path)
        return {"success": True, "message": f"Deleted checkpoint '{safe_name}'"}

    def export_checkpoint(self, name: str, format: str = "safetensors") -> Dict[str, Any]:
        """Export a checkpoint in the specified format."""
        safe_name = os.path.basename(name)
        if not safe_name.endswith(".pt"):
            safe_name += ".pt"
        path = os.path.join(CHECKPOINT_DIR, safe_name)
        if not os.path.exists(path):
            return {"success": False, "error": f"Checkpoint '{safe_name}' not found"}

        export_path = os.path.join(EXPORT_DIR, safe_name.replace(".pt", f".{format}"))
        with self._lock:
            try:
                import torch
                checkpoint = torch.load(path, map_location="cpu")
                if format == "safetensors":
                    try:
                        from safetensors.torch import save_file
                        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                            save_file(checkpoint["model_state_dict"], export_path)
                        else:
                            save_file(checkpoint, export_path)
                    except ImportError:
                        return {"success": False, "error": "safetensors not installed"}
                else:
                    torch.save(checkpoint, export_path)

                return {
                    "success": True,
                    "name": os.path.basename(export_path),
                    "size_mb": round(os.path.getsize(export_path) / (1024 * 1024), 2),
                    "format": format,
                    "message": f"Exported to {export_path}",
                }
            except Exception as exc:
                return {"success": False, "error": str(exc)}

    def get_checkpoint_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a checkpoint."""
        safe_name = os.path.basename(name)
        if not safe_name.endswith(".pt"):
            safe_name += ".pt"
        path = os.path.join(CHECKPOINT_DIR, safe_name)
        if not os.path.exists(path):
            return None
        try:
            import torch
            checkpoint = torch.load(path, map_location="cpu")
            info = {
                "name": safe_name,
                "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
                "keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
            }
            if isinstance(checkpoint, dict):
                config = checkpoint.get("config")
                if config:
                    info["config"] = config
                metadata = checkpoint.get("metadata")
                if metadata:
                    info["metadata"] = metadata
            return info
        except Exception:
            return {"name": safe_name, "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2)}