"""
Tokenizer Configuration Module for Netcradus LLM.

===============================================================================
PURPOSE:
===============================================================================
This module defines the configuration schema (`TokenizerConfig`) for training,
saving, loading, and running the Netcradus BPE Tokenizer. It ensures all tokenizer
hyperparameters, special token mappings, file paths, and normalization settings
are centralized and validated.

===============================================================================
WORKFLOW:
===============================================================================
1. Instantiate `TokenizerConfig` with default or custom parameters (e.g., vocab_size, paths).
2. Validate configuration options using `validate()`.
3. Pass `TokenizerConfig` to `BPEContainerTrainer` for training or `NetcradusTokenizerWrapper` for inference.
4. Export/load configuration to/from JSON via `to_dict()`, `from_dict()`, `save_json()`, `load_json()`.

===============================================================================
ALGORITHMS & DESIGN:
===============================================================================
- Dataclass-based configuration with strict type validation.
- Standardized special token ordering:
    Index 0: <PAD>  (Padding)
    Index 1: <BOS>  (Beginning of Sequence)
    Index 2: <EOS>  (End of Sequence)
    Index 3: <UNK>  (Unknown token fallback)
    Index 4: <MASK> (Mask token for MLM/FIM tasks)

===============================================================================
COMPLEXITY:
===============================================================================
- Time Complexity: O(1) for initialization and serialization.
- Space Complexity: O(1) memory overhead.

===============================================================================
EXAMPLE USAGE:
===============================================================================
>>> from tokenizer.tokenizer_config import TokenizerConfig
>>> config = TokenizerConfig(vocab_size=32000)
>>> config.validate()
>>> print(config.special_tokens_dict)
{'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<MASK>': 4}
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """
    Configuration parameters for the Netcradus LLM BPE Tokenizer.
    """

    # Vocabulary parameters
    vocab_size: int = 32000
    min_frequency: int = 2
    
    # Special Tokens
    pad_token: str = "<PAD>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    unk_token: str = "<UNK>"
    mask_token: str = "<MASK>"

    # Pre-tokenization & Normalization
    add_prefix_space: bool = False
    lowercase: bool = False
    unicode_normalizer: str = "NFKC"

    # Paths
    corpus_path: str = "data/processed/cleaned_corpus.txt"
    output_dir: str = "tokenizer_model"

    # Additional metadata
    algorithm: str = "Byte-Pair Encoding (BPE)"
    model_type: str = "BPE"
    version: str = "1.0.0"

    @property
    def special_tokens_list(self) -> List[str]:
        """Return the ordered list of special tokens."""
        return [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.mask_token,
        ]

    @property
    def special_tokens_dict(self) -> Dict[str, int]:
        """Return dictionary mapping special token strings to their fixed IDs."""
        return {token: idx for idx, token in enumerate(self.special_tokens_list)}

    def validate(self) -> None:
        """
        Validate configuration options and raise ValueError if invalid.
        """
        if self.vocab_size <= len(self.special_tokens_list):
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must be greater than "
                f"the number of special tokens ({len(self.special_tokens_list)})."
            )
        if self.min_frequency < 1:
            raise ValueError(f"min_frequency must be >= 1, got {self.min_frequency}")
        
        # Check for duplicates in special tokens
        unique_specials = set(self.special_tokens_list)
        if len(unique_specials) != len(self.special_tokens_list):
            raise ValueError("Duplicate special tokens detected in configuration!")

        logger.info("TokenizerConfig successfully validated.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration instance to dictionary."""
        d = asdict(self)
        d["special_tokens"] = self.special_tokens_list
        d["special_tokens_dict"] = self.special_tokens_dict
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenizerConfig":
        """Create a TokenizerConfig instance from a dictionary."""
        # Filter keys that match dataclass fields
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def save_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load_json(cls, filepath: str) -> "TokenizerConfig":
        """Load configuration from JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found at: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
