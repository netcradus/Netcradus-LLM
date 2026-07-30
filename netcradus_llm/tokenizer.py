import os
from typing import List, Union, Dict, Optional

class NetcradusTokenizer:
    """
    Byte-Level Tokenizer wrapper for Netcradus LLM.
    Supports chat formatting (<|im_start|>, <|im_end|>) and Fill-In-The-Middle (FIM) tokens.
    Uses fallback character/byte encoding when tiktoken is not available.
    """
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<|pad|>": 0,
            "<|bos|>": 1,
            "<|eos|>": 2,
            "<|fim_prefix|>": 3,
            "<|fim_middle|>": 4,
            "<|fim_suffix|>": 5,
            "<|im_start|>": 6,
            "<|im_end|>": 7
        }
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}
        import re
        special_pattern = "|".join([re.escape(k) for k in self.special_tokens.keys()])
        self.special_regex = re.compile(f"({special_pattern})")

        self.has_tiktoken = False
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
            self.has_tiktoken = True
        except ImportError:
            self.enc = None

    def _map_token_id(self, raw_id: int) -> int:
        shifted = raw_id + 8
        if shifted < self.vocab_size:
            return shifted
        num_usable = max(1, self.vocab_size - 8)
        return (raw_id % num_usable) + 8

    def _unmap_token_id(self, token_id: int) -> int:
        return max(0, token_id - 8)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode string text to list of token IDs."""
        tokens = []
        # Split text by special token patterns if present
        parts = self.special_regex.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
            else:
                if self.has_tiktoken and self.enc:
                    raw_tokens = self.enc.encode(part, allowed_special="all")
                    mapped = [self._map_token_id(t) for t in raw_tokens]
                else:
                    raw_bytes = part.encode("utf-8")
                    mapped = [self._map_token_id(b) for b in raw_bytes]
                tokens.extend(mapped)

        if add_special_tokens:
            tokens = [self.special_tokens["<|bos|>"]] + tokens + [self.special_tokens["<|eos|>"]]
            
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ID list back to text string."""
        res_parts = []
        curr_raw_ids = []

        def flush_raw_ids():
            if not curr_raw_ids:
                return
            if self.has_tiktoken and self.enc:
                try:
                    res_parts.append(self.enc.decode(curr_raw_ids))
                except Exception:
                    b_vals = [max(0, min(255, tid)) for tid in curr_raw_ids]
                    res_parts.append(bytes(b_vals).decode("utf-8", errors="replace"))
            else:
                b_vals = [max(0, min(255, tid)) for tid in curr_raw_ids]
                res_parts.append(bytes(b_vals).decode("utf-8", errors="replace"))
            curr_raw_ids.clear()

        for t in tokens:
            if t in self.inv_special_tokens:
                flush_raw_ids()
                if not skip_special_tokens:
                    res_parts.append(self.inv_special_tokens[t])
            else:
                curr_raw_ids.append(self._unmap_token_id(t))

        flush_raw_ids()
        return "".join(res_parts)

    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format multi-turn conversation into Netcradus chat prompt schema:
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        Hello!<|im_end|>
        <|im_start|>assistant
        """
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def format_fim_prompt(self, prefix: str, suffix: str) -> str:
        """
        Format Fill-In-The-Middle (FIM) prompt:
        <|fim_prefix|>prefix_code<|fim_suffix|>suffix_code<|fim_middle|>
        """
        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
