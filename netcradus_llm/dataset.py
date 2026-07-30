import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple

from netcradus_llm.tokenizer import NetcradusTokenizer


class PretrainingDataset(Dataset):
    """
    Dataset for Large Language Model pretraining.
    Packs tokenized text chunks into fixed sequence lengths (max_seq_len)
    and applies random Fill-In-The-Middle (FIM) transformations on code snippets.
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer: NetcradusTokenizer,
        max_seq_len: int = 512,
        fim_rate: float = 0.2
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.fim_rate = fim_rate
        
        # Tokenize and pack all text chunks into uniform sequence blocks
        all_token_ids = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_token_ids.extend(tokens)
            
        self.samples = []
        for i in range(0, len(all_token_ids), max_seq_len):
            chunk = all_token_ids[i : i + max_seq_len]
            if len(chunk) == max_seq_len:
                self.samples.append(chunk)
            elif len(chunk) > 0:
                padded = chunk + [0] * (max_seq_len - len(chunk))
                self.samples.append(padded)

        if len(self.samples) == 0:
            self.samples.append([0] * max_seq_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.samples[idx]
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()  # Next-token prediction labels

        return {
            "input_ids": input_ids,
            "labels": labels
        }


class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning (SFT) Dataset for instruction following.
    Masks out user prompt tokens (label = 0 / pad_token_id) so the model only
    computes loss on target assistant output tokens.
    """
    def __init__(
        self,
        conversations: List[List[Dict[str, str]]],
        tokenizer: NetcradusTokenizer,
        max_seq_len: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        for conv in conversations:
            formatted_prompt = self.tokenizer.format_chat_prompt(conv)
            tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=True)
            
            # Truncate or pad to max_seq_len
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            else:
                tokens = tokens + [0] * (max_seq_len - len(tokens))

            input_ids = torch.tensor(tokens, dtype=torch.long)
            labels = input_ids.clone()

            # Mask out non-assistant prompt tokens (label = -100) so loss is only computed on target outputs
            # Find assistant indicator token sequence (<|im_start|> assistant \n)
            assistant_token = self.tokenizer.special_tokens.get("<|im_start|>", 6)
            for i in range(len(tokens)):
                if tokens[i] == assistant_token:
                    # Keep labels from assistant start onwards
                    break
                labels[i] = -100

            self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]
