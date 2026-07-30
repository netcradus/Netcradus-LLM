# Netcradus LLM - Production-Grade Language Model Architecture & Pipeline

Netcradus LLM is a modular, high-performance Transformer architecture built for modern pretraining, fine-tuning, and inference pipelines.

---

## 🚀 Phase 4 – Tokenizer Design & Training

Phase 4 delivers a production-quality, byte-level Byte Pair Encoding (BPE) tokenizer training pipeline built completely from scratch using the HuggingFace `tokenizers` engine and custom modular utilities.

### 🌟 Key Features

- **Byte Pair Encoding (BPE)**: High-performance ByteLevel pre-tokenization and decoding for 100% loss-less Unicode string reconstruction.
- **Vocabulary Size**: 32,000 tokens.
- **Special Token Placement**:
  - `<PAD>`: ID 0
  - `<BOS>`: ID 1
  - `<EOS>`: ID 2
  - `<UNK>`: ID 3
  - `<MASK>`: ID 4
- **Robust Error Handling**: Handles missing files, invalid UTF-8 bytes gracefully, empty inputs, duplicate tokens, and streams large corpus files to prevent high RAM consumption.

---

## 📁 Directory Structure

```
Netcradus-LLM/
│
├── tokenizer/
│   ├── train_tokenizer.py    # Training pipeline with BPEContainerTrainer & streaming iterator
│   ├── tokenizer_utils.py    # NetcradusTokenizerWrapper with encode/decode & batch APIs
│   ├── tokenizer_config.py   # TokenizerConfig dataclass with validation & JSON export
│   ├── tokenizer_test.py     # 7-stage automated unit and integration test suite
│   └── __init__.py           # Package exports
│
├── tokenizer_model/          # Model serialization artifacts
│   ├── tokenizer.json        # Standard HuggingFace tokenizer JSON definition
│   ├── vocab.json            # Token -> ID vocabulary mapping
│   ├── merges.txt            # BPE merge rules list
│   ├── special_tokens.json   # Special token metadata dictionary
│   └── tokenizer_config.json # Hyperparameter configuration export
│
├── data/
│   └── processed/
│       └── cleaned_corpus.txt # Cleaned training dataset corpus
│
├── requirements.txt          # Package dependencies (torch, tokenizers, tiktoken, etc.)
└── README.md                 # Project documentation
```

---

## 🛠️ Usage Guide

### 1. Training the Tokenizer
To train the tokenizer on your custom text corpus (`data/processed/cleaned_corpus.txt`):

```bash
python tokenizer/train_tokenizer.py
```

### 2. Encoding and Decoding in Python

```python
from tokenizer import NetcradusTokenizerWrapper

# Load trained tokenizer model
tokenizer = NetcradusTokenizerWrapper.load("tokenizer_model")

# Encode single text sequence
text = "Artificial intelligence and deep learning."
tokens = tokenizer.encode(text, add_special_tokens=True)
print("Token IDs:", tokens)
# Output: [1, 1768, 648, 276, 2724, 767, 2144, 18, 2]

# Decode token IDs back to text
decoded = tokenizer.decode(tokens, skip_special_tokens=True)
print("Decoded Text:", decoded)
# Output: 'Artificial intelligence and deep learning.'

# Batch Operations
texts = ["Hello Netcradus!", "Transformer architecture."]
batch_ids = tokenizer.batch_encode(texts, add_special_tokens=True)
batch_texts = tokenizer.batch_decode(batch_ids, skip_special_tokens=True)
```

---

## 🧪 Testing & Verification

Run the comprehensive 7-stage test suite:

```bash
python tokenizer/tokenizer_test.py
```

### Test Coverage:
1. **Basic Encoding & Decoding**: Exact string roundtrip verification.
2. **Special Tokens**: `<BOS>`, `<EOS>`, `<PAD>`, and `<MASK>` injection and recognition.
3. **Unknown Words & OOV**: Byte-level fallback subword resolution.
4. **Multilingual & Unicode**: Preservation of Emojis, CJK, Devanagari, and mathematical symbols.
5. **Long Documents**: Large document paragraph compression stress testing.
6. **Edge Cases**: Empty strings `""` and empty sequence handling.
7. **Serialization**: Save and load roundtrip consistency across filesystem locations.
