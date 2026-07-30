"""
Tokenizer Training Pipeline for Netcradus LLM.

===============================================================================
PURPOSE:
===============================================================================
Trains a production-grade Byte Pair Encoding (BPE) tokenizer from scratch using a cleaned text corpus.
Generates all required serialization artifacts (tokenizer.json, vocab.json, merges.txt, 
special_tokens.json, and tokenizer_config.json) under `tokenizer_model/`.

===============================================================================
WORKFLOW:
===============================================================================
1. Load configuration (`TokenizerConfig`) and validate parameters.
2. Read text corpus in chunks to support scalable memory usage on large files.
3. Apply Unicode NFKC normalization and byte-level pre-tokenization.
4. Train BPE model to construct a vocabulary of target size (32,000 tokens).
5. Ensure special tokens (<PAD>, <BOS>, <EOS>, <UNK>, <MASK>) are registered at fixed IDs (0..4).
6. Export serialized tokenizer files:
   - tokenizer.json (HuggingFace tokenizers format)
   - vocab.json (Token -> ID dictionary)
   - merges.txt (BPE merge rules list)
   - special_tokens.json (Special token ID lookup)
   - tokenizer_config.json (Configuration metadata)

===============================================================================
ALGORITHMS USED:
===============================================================================
1. Byte Pair Encoding (BPE):
   - Starts with base byte vocabulary (256 bytes + special tokens).
   - Iteratively counts character/token pairs across corpus and merges the most frequent pair.
   - Repeats until target vocabulary size (32,000) is reached.
2. Byte-Level Pre-Tokenization:
   - Map bytes to printable Unicode characters, allowing handling of arbitrary UTF-8 text without OOV errors.

===============================================================================
COMPLEXITY:
===============================================================================
- Time Complexity: O(N * log(V)) where N is corpus size (number of tokens) and V is vocab size.
- Space Complexity: O(N + V) memory overhead during pair count aggregation.

===============================================================================
EXAMPLE USAGE:
===============================================================================
$ python tokenizer/train_tokenizer.py
Or programmatically:
>>> from tokenizer.train_tokenizer import train_tokenizer
>>> config = TokenizerConfig(vocab_size=32000)
>>> train_tokenizer(config)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Iterator, List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
from tokenizer.tokenizer_config import TokenizerConfig

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_tokenizer")


def line_iterator(file_path: str, chunk_size: int = 1000) -> Iterator[str]:
    """
    Stream text from file in line chunks to handle large files efficiently without OOM errors.
    
    Args:
        file_path: Path to text file.
        chunk_size: Number of lines per yielded chunk.
        
    Yields:
        Concatenated text string for each chunk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Corpus file not found: '{file_path}'")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError(f"Corpus file is empty (0 bytes): '{file_path}'")

    logger.info(f"Streaming text corpus from '{file_path}' ({file_size / (1024 * 1024):.2f} MB)")
    
    lines_read = 0
    chunk = []
    
    # Open file with fallback for invalid UTF-8 bytes
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                chunk.append(stripped)
                lines_read += 1
            if len(chunk) >= chunk_size:
                yield "\n".join(chunk)
                chunk = []
        if chunk:
            yield "\n".join(chunk)

    if lines_read == 0:
        raise ValueError(f"Corpus file '{file_path}' contains no valid text lines!")


class BPEContainerTrainer:
    """
    Modular BPE Tokenizer Trainer encapsulating build, training, validation, and export routines.
    """

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.config.validate()

    def build_raw_tokenizer(self) -> Tokenizer:
        """
        Construct and configure un-trained HuggingFace Tokenizer with BPE model,
        NFKC normalizer, ByteLevel pre-tokenizer, and ByteLevel decoder.
        """
        # Initialize raw BPE model with UNK fallback
        bpe_model = models.BPE(unk_token=self.config.unk_token)
        tokenizer = Tokenizer(bpe_model)

        # Set Normalizer: Unicode NFKC
        tokenizer.normalizer = normalizers.NFKC()

        # Set Pre-Tokenizer: ByteLevel pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=self.config.add_prefix_space
        )

        # Set Decoder: ByteLevel decoder
        tokenizer.decoder = decoders.ByteLevel()

        return tokenizer

    def train(self) -> Tokenizer:
        """
        Execute training on the configured text corpus and return the trained tokenizer.
        """
        logger.info("Initializing BPE Tokenizer training...")
        tokenizer = self.build_raw_tokenizer()

        # Configure Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            special_tokens=self.config.special_tokens_list,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        # Obtain iterator for dataset
        text_stream = line_iterator(self.config.corpus_path)

        # Execute training
        logger.info(f"Training BPE Tokenizer with target vocab size = {self.config.vocab_size}...")
        tokenizer.train_from_iterator(text_stream, trainer=trainer)

        actual_vocab_size = tokenizer.get_vocab_size()
        logger.info(f"Training completed successfully. Final vocabulary size: {actual_vocab_size}")

        # Validate duplicate tokens or vocabulary anomaly
        vocab = tokenizer.get_vocab()
        if len(vocab) != actual_vocab_size:
            logger.warning("Mismatch detected between vocab mapping length and get_vocab_size()")

        # Verify special tokens placement
        for token, expected_id in self.config.special_tokens_dict.items():
            token_id = tokenizer.token_to_id(token)
            logger.info(f"Special token '{token}' mapped to ID: {token_id} (expected: {expected_id})")

        return tokenizer

    def save_artifacts(self, tokenizer: Tokenizer, output_dir: str) -> Dict[str, str]:
        """
        Export all required model files to output_dir:
        1. tokenizer.json
        2. vocab.json
        3. merges.txt
        4. special_tokens.json
        5. tokenizer_config.json
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}

        # 1. Save tokenizer.json
        tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
        tokenizer.save(tokenizer_json_path)
        saved_files["tokenizer.json"] = tokenizer_json_path

        # 2. Save vocab.json
        vocab = tokenizer.get_vocab()
        # Sort vocab by token ID for clear ordering
        sorted_vocab = dict(sorted(vocab.items(), key=lambda item: item[1]))
        vocab_path = os.path.join(output_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(sorted_vocab, f, indent=2, ensure_ascii=False)
        saved_files["vocab.json"] = vocab_path

        # 3. Save merges.txt
        merges_path = os.path.join(output_dir, "merges.txt")
        # Extract BPE merges directly from BPE model representation if available,
        # or parse from tokenizer json export
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tok_json_data = json.load(f)
        
        merges = tok_json_data.get("model", {}).get("merges", [])
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.2 - BPE Merges for Netcradus LLM\n")
            for merge in merges:
                if isinstance(merge, list):
                    f.write(" ".join(merge) + "\n")
                elif isinstance(merge, str):
                    f.write(merge + "\n")
        saved_files["merges.txt"] = merges_path

        # 4. Save special_tokens.json
        special_tokens_path = os.path.join(output_dir, "special_tokens.json")
        special_tokens_info = {
            "special_tokens": self.config.special_tokens_list,
            "special_tokens_dict": self.config.special_tokens_dict,
            "pad_token": self.config.pad_token,
            "bos_token": self.config.bos_token,
            "eos_token": self.config.eos_token,
            "unk_token": self.config.unk_token,
            "mask_token": self.config.mask_token,
        }
        with open(special_tokens_path, "w", encoding="utf-8") as f:
            json.dump(special_tokens_info, f, indent=2, ensure_ascii=False)
        saved_files["special_tokens.json"] = special_tokens_path

        # 5. Save tokenizer_config.json
        config_path = os.path.join(output_dir, "tokenizer_config.json")
        self.config.save_json(config_path)
        saved_files["tokenizer_config.json"] = config_path

        logger.info(f"Successfully saved all 5 tokenizer artifacts to '{output_dir}':")
        for filename, path in saved_files.items():
            file_size = os.path.getsize(path)
            logger.info(f"  - {filename}: {path} ({file_size} bytes)")

        return saved_files


def train_tokenizer(config: TokenizerConfig = None) -> Tokenizer:
    """
    Main helper function to initialize config, train BPE tokenizer, and save output artifacts.
    """
    if config is None:
        config = TokenizerConfig()
    
    trainer = BPEContainerTrainer(config)
    tokenizer = trainer.train()
    trainer.save_artifacts(tokenizer, config.output_dir)
    return tokenizer


if __name__ == "__main__":
    logger.info("Starting Netcradus LLM Tokenizer Training CLI...")
    default_config = TokenizerConfig(
        vocab_size=32000,
        corpus_path="data/processed/cleaned_corpus.txt",
        output_dir="tokenizer_model"
    )
    trained_tok = train_tokenizer(default_config)
    logger.info("Tokenizer training pipeline completed successfully!")
