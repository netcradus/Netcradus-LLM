"""
Tokenizer Utilities Module for Netcradus LLM.

===============================================================================
PURPOSE:
===============================================================================
Provides the high-level operational interface (`NetcradusTokenizerWrapper` / `NetcradusTokenizer`)
for encoding raw text into numerical token IDs and decoding token IDs back into human-readable text.
Supports single sequence processing, batch operations, special token injection, and model persistence.

===============================================================================
WORKFLOW:
===============================================================================
1. Initialize wrapper by loading trained model files from a directory (`load()` or `__init__()`).
2. Encode strings using `encode(text)` or `batch_encode(texts)`.
3. Process token IDs in Transformer models / Datasets.
4. Decode predicted token IDs using `decode(tokens)` or `batch_decode(batch_tokens)`.
5. Persist or export modified tokenizer models using `save(directory)`.

===============================================================================
ALGORITHMS USED:
===============================================================================
- High-Performance Rust BPE Tokenizer Backend (via `tokenizers.Tokenizer`).
- Byte-level mapping for 100% loss-less Unicode string reconstruction.
- Batch vectorization for parallel encoding/decoding.

===============================================================================
COMPLEXITY:
===============================================================================
- Encode Time Complexity: O(L) where L is string length in characters.
- Decode Time Complexity: O(T) where T is length of token sequence.
- Space Complexity: O(T) memory for token lists.

===============================================================================
EXAMPLE USAGE:
===============================================================================
>>> from tokenizer.tokenizer_utils import NetcradusTokenizerWrapper
>>> tok = NetcradusTokenizerWrapper.load("tokenizer_model")
>>> ids = tok.encode("Hello Netcradus LLM!", add_special_tokens=True)
>>> print(ids)
[1, 1548, 8932, 2341, 2]
>>> decoded = tok.decode(ids, skip_special_tokens=True)
>>> print(decoded)
'Hello Netcradus LLM!'
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tokenizers import Tokenizer
from tokenizer.tokenizer_config import TokenizerConfig

logger = logging.getLogger("tokenizer_utils")


class NetcradusTokenizerWrapper:
    """
    Production wrapper class for Netcradus LLM BPE Tokenizer.
    """

    def __init__(self, tokenizer: Tokenizer, config: Optional[TokenizerConfig] = None):
        """
        Initialize wrapper with a trained Tokenizer instance and optional TokenizerConfig.
        """
        if tokenizer is None:
            raise ValueError("Tokenizer instance cannot be None.")
        self._tokenizer = tokenizer
        self.config = config or TokenizerConfig()

        # Cache special token IDs
        self._pad_id = self._tokenizer.token_to_id(self.config.pad_token)
        self._bos_id = self._tokenizer.token_to_id(self.config.bos_token)
        self._eos_id = self._tokenizer.token_to_id(self.config.eos_token)
        self._unk_id = self._tokenizer.token_to_id(self.config.unk_token)
        self._mask_id = self._tokenizer.token_to_id(self.config.mask_token)

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size including special tokens."""
        return self._tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> Optional[int]:
        """Return padding token ID (<PAD>)."""
        return self._pad_id

    @property
    def bos_id(self) -> Optional[int]:
        """Return beginning-of-sequence token ID (<BOS>)."""
        return self._bos_id

    @property
    def eos_id(self) -> Optional[int]:
        """Return end-of-sequence token ID (<EOS>)."""
        return self._eos_id

    @property
    def unk_id(self) -> Optional[int]:
        """Return unknown token ID (<UNK>)."""
        return self._unk_id

    @property
    def mask_id(self) -> Optional[int]:
        """Return mask token ID (<MASK>)."""
        return self._mask_id

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode a single string into a list of token IDs.

        Args:
            text: Raw input string to encode.
            add_special_tokens: If True, prepends <BOS> and appends <EOS>.

        Returns:
            List of integer token IDs.
        """
        if text is None:
            raise ValueError("Cannot encode None. Expected a string.")
        if not isinstance(text, str):
            text = str(text)

        if len(text) == 0:
            if add_special_tokens:
                tokens = []
                if self._bos_id is not None:
                    tokens.append(self._bos_id)
                if self._eos_id is not None:
                    tokens.append(self._eos_id)
                return tokens
            return []

        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids = encoding.ids

        if add_special_tokens:
            prefix = [self._bos_id] if self._bos_id is not None else []
            suffix = [self._eos_id] if self._eos_id is not None else []
            ids = prefix + ids + suffix

        return ids

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode a sequence of integer token IDs back into a text string.

        Args:
            tokens: List of token IDs.
            skip_special_tokens: If True, filters out special tokens (<PAD>, <BOS>, <EOS>, etc.).

        Returns:
            Decoded string.
        """
        if tokens is None:
            raise ValueError("Cannot decode None. Expected a list of integers.")
        if not isinstance(tokens, list):
            try:
                tokens = list(tokens)
            except Exception as e:
                raise TypeError(f"Input tokens must be a list or iterable of integers. Got {type(tokens)}") from e

        if len(tokens) == 0:
            return ""

        # Validate that all items are integers
        clean_tokens = []
        for t in tokens:
            if not isinstance(t, int):
                try:
                    clean_tokens.append(int(t))
                except (ValueError, TypeError):
                    logger.warning(f"Skipping non-integer token in decode: {t}")
            else:
                clean_tokens.append(t)

        return self._tokenizer.decode(clean_tokens, skip_special_tokens=skip_special_tokens)

    def batch_encode(self, texts: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        """
        Encode a list of text strings in batch.

        Args:
            texts: List of raw strings.
            add_special_tokens: If True, adds <BOS> and <EOS> to each encoded sequence.

        Returns:
            List of token ID lists.
        """
        if not isinstance(texts, list):
            raise TypeError(f"batch_encode expects a list of strings, got {type(texts)}")
        
        return [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]

    def batch_decode(self, batch_tokens: List[List[int]], skip_special_tokens: bool = False) -> List[str]:
        """
        Decode a batch of token ID lists back into strings.

        Args:
            batch_tokens: List of token ID lists.
            skip_special_tokens: If True, filters out special tokens.

        Returns:
            List of decoded text strings.
        """
        if not isinstance(batch_tokens, list):
            raise TypeError(f"batch_decode expects a list of token lists, got {type(batch_tokens)}")

        return [self.decode(tokens, skip_special_tokens=skip_special_tokens) for tokens in batch_tokens]

    def save(self, directory: str) -> None:
        """
        Save tokenizer and configuration files to specified directory.

        Args:
            directory: Absolute or relative target directory path.
        """
        os.makedirs(directory, exist_ok=True)
        tokenizer_json_path = os.path.join(directory, "tokenizer.json")
        self._tokenizer.save(tokenizer_json_path)

        # Save config JSON
        config_path = os.path.join(directory, "tokenizer_config.json")
        self.config.save_json(config_path)

        # Save vocab.json
        vocab = self._tokenizer.get_vocab()
        sorted_vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
        vocab_path = os.path.join(directory, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(sorted_vocab, f, indent=2, ensure_ascii=False)

        # Save special_tokens.json
        special_tokens_path = os.path.join(directory, "special_tokens.json")
        special_tokens_info = {
            "special_tokens": self.config.special_tokens_list,
            "special_tokens_dict": self.config.special_tokens_dict,
        }
        with open(special_tokens_path, "w", encoding="utf-8") as f:
            json.dump(special_tokens_info, f, indent=2, ensure_ascii=False)

        logger.info(f"Tokenizer successfully saved to '{directory}'")

    @classmethod
    def load(cls, directory: str) -> "NetcradusTokenizerWrapper":
        """
        Load tokenizer model and configuration from target directory.

        Args:
            directory: Path to directory containing saved tokenizer files (`tokenizer.json`).

        Returns:
            Loaded `NetcradusTokenizerWrapper` instance.
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Tokenizer model directory does not exist: '{directory}'")

        tokenizer_json_path = os.path.join(directory, "tokenizer.json")
        if not os.path.exists(tokenizer_json_path):
            raise FileNotFoundError(f"Missing required 'tokenizer.json' file in '{directory}'")

        tokenizer = Tokenizer.from_file(tokenizer_json_path)

        config_path = os.path.join(directory, "tokenizer_config.json")
        if os.path.exists(config_path):
            config = TokenizerConfig.load_json(config_path)
        else:
            config = TokenizerConfig()

        logger.info(f"Successfully loaded tokenizer from '{directory}' (vocab size: {tokenizer.get_vocab_size()})")
        return cls(tokenizer, config)


# Alias for seamless backward compatibility across Netcradus LLM codebase
NetcradusTokenizer = NetcradusTokenizerWrapper
