"""
Netcradus Admin - Background Training Job Manager.

===============================================================================
PURPOSE:
===============================================================================
Runs NetcradusTrainer in a background thread so the admin panel can start,
monitor, and stop training jobs without blocking the HTTP server.

State machine: idle -> running -> (finished | stopped | error) -> idle
===============================================================================
"""

import os
import time
import threading
import torch
from typing import Optional, Dict, Any

from netcradus_llm.config import NetcradusConfig
from netcradus_llm.model import NetcradusForCausalLM
from netcradus_llm.tokenizer import NetcradusTokenizer
from netcradus_llm.dataset import PretrainingDataset
from netcradus_llm.train import NetcradusTrainer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CORPUS = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_corpus.txt")

DEFAULT_TRAINING_CONFIG = {
    "vocab_size": 32000,
    "hidden_size": 256,
    "intermediate_size": 704,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "max_position_embeddings": 2048,
    "rope_theta": 10000.0,
    "max_seq_len": 256,
    "batch_size": 2,
    "learning_rate": 3e-4,
    "warmup_steps": 10,
    "max_steps": 100,
    "output_dir": os.path.join(PROJECT_ROOT, "checkpoints_demo"),
}


class TrainingJobManager:

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._trainer: Optional[NetcradusTrainer] = None
        self._state = "idle"
        self._message = ""
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None
        self._config: Dict[str, Any] = {}

    def status(self) -> Dict[str, Any]:
        trainer = self._trainer
        with self._lock:
            state = self._state
            message = self._message
            started_at = self._started_at
            config = dict(self._config)

        result = {
            "state": state,
            "message": message,
            "started_at": started_at,
            "elapsed": round(time.time() - started_at, 1) if started_at else 0.0,
            "config": config,
        }

        if trainer is not None:
            result["step"] = trainer.current_step
            result["loss"] = round(trainer.last_loss, 4)
            result["tokens_per_sec"] = round(trainer.tokens_per_sec, 1)
        else:
            result["step"] = 0
            result["loss"] = 0.0
            result["tokens_per_sec"] = 0.0

        if state in ("finished", "stopped", "error"):
            result["finished_at"] = self._finished_at
        return result

    def start(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            if self._state == "running":
                raise RuntimeError("A training job is already running.")
            self._config = {**DEFAULT_TRAINING_CONFIG, **(config or {})}
            cfg = self._config
            self._state = "running"
            self._message = "Initializing training job..."
            self._started_at = time.time()
            self._finished_at = None
            self._stop_event = threading.Event()
            self._trainer = None

        self._thread = threading.Thread(
            target=self._run_job,
            args=(dict(cfg),),
            daemon=True,
            name="netcradus-training-job",
        )
        self._thread.start()
        return self.status()

    def _run_job(self, cfg: Dict[str, Any]) -> None:
        stop_event = self._stop_event
        try:
            model_cfg = NetcradusConfig(
                vocab_size=int(cfg["vocab_size"]),
                hidden_size=int(cfg["hidden_size"]),
                intermediate_size=int(cfg["intermediate_size"]),
                num_hidden_layers=int(cfg["num_hidden_layers"]),
                num_attention_heads=int(cfg["num_attention_heads"]),
                num_key_value_heads=int(cfg["num_key_value_heads"]),
                max_position_embeddings=int(cfg["max_position_embeddings"]),
                rope_theta=float(cfg["rope_theta"]),
            )
            tokenizer = NetcradusTokenizer(vocab_size=model_cfg.vocab_size)
            model = NetcradusForCausalLM(model_cfg)

            corpus_path = cfg.get("corpus_path") or DEFAULT_CORPUS
            if not os.path.exists(corpus_path):
                fallback = os.path.join(os.path.dirname(corpus_path), "training_data.txt")
                if os.path.exists(fallback):
                    corpus_path = fallback
                else:
                    raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

            with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
                texts = [ln.strip() for ln in f if ln.strip()]

            dataset = PretrainingDataset(texts, tokenizer, max_seq_len=int(cfg["max_seq_len"]))
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=int(cfg["batch_size"]), shuffle=True
            )

            output_dir = cfg["output_dir"]
            os.makedirs(output_dir, exist_ok=True)

            trainer = NetcradusTrainer(
                model=model,
                train_dataloader=dataloader,
                learning_rate=float(cfg["learning_rate"]),
                warmup_steps=int(cfg["warmup_steps"]),
                max_steps=int(cfg["max_steps"]),
                output_dir=output_dir,
                stop_event=stop_event,
            )
            with self._lock:
                self._trainer = trainer
                self._message = "Training in progress..."
            trainer.train()

            stopped = stop_event.is_set()
            with self._lock:
                self._state = "stopped" if stopped else "finished"
                self._message = (
                    "Training stopped by admin." if stopped
                    else f"Training finished. Checkpoint saved to {output_dir}/netcradus_final.pt"
                )
                self._finished_at = time.time()
                # Release the heavy model + optimizer now that training is done so
                # memory is reclaimed before the next job (prevents OOM across jobs).
                try:
                    trainer.model = None
                    trainer.optimizer = None
                    trainer.train_dataloader = None
                except Exception:
                    pass
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._message = f"Training failed: {exc}"
                self._finished_at = time.time()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if self._state != "running":
                raise RuntimeError("No running training job to stop.")
            self._state = "stopping"
            self._message = "Stopping training job..."
        if self._stop_event is not None:
            self._stop_event.set()
        return self.status()