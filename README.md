# Netcradus LLM - Production-Grade Language Model Architecture & Pipeline

Netcradus LLM is a modular, high-performance Transformer architecture built for modern pretraining, fine-tuning, and inference pipelines.

---

## 🌐 ChatGPT Web Interface

Launch a production-grade, dark-mode ChatGPT-style web UI to interact with Netcradus LLM in real-time.

```bash
python web_server.py --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your web browser.

### 🌟 Web UI Features
- **ChatGPT Layout**: Collapsible sidebar, prompt suggestions hero section, conversation history management, and clear chat.
- **Rich Aesthetics**: Dark glassmorphism interface (`#090d16`, `#101726`, `#8b5cf6`, `#06b6d4`), Inter & JetBrains Mono typography, pulse typing animations.
- **Model Parameters Control**: Dynamic sliders for Temperature (0.1–1.5) and Max Tokens (16–512).
- **Code Block Formatting**: Markdown syntax highlighting with one-click "Copy Code" button.
- **Zero External Dependencies**: Built with Vanilla HTML/CSS/JS and native Python HTTP server.

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
├── web/                      # ChatGPT Web Interface
│   ├── index.html            # ChatGPT-style web layout & HTML markup
│   ├── styles.css            # Dark glassmorphism CSS design system
│   └── app.js                # Frontend JS app logic & Markdown renderer
│
├── web_server.py             # Multi-threaded HTTP backend API server
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

### 1. Launching the ChatGPT Web Interface
```bash
python web_server.py --port 8000
```

### 2. Training the Tokenizer
To train the tokenizer on your custom text corpus (`data/processed/cleaned_corpus.txt`):

```bash
python tokenizer/train_tokenizer.py
```

### 3. Encoding and Decoding in Python

```python
from tokenizer import NetcradusTokenizerWrapper

# Load trained tokenizer model
tokenizer = NetcradusTokenizerWrapper.load("tokenizer_model")

# Encode single text sequence
text = "Artificial intelligence and deep learning."
tokens = tokenizer.encode(text, add_special_tokens=True)
print("Token IDs:", tokens)

# Decode token IDs back to text
decoded = tokenizer.decode(tokens, skip_special_tokens=True)
print("Decoded Text:", decoded)
```

---

## 🧪 Testing & Verification

Run the comprehensive test suites:

```bash
# Tokenizer Test Suite
python tokenizer/tokenizer_test.py

# End-to-End LLM Architecture Test Suite
python test_suite.py
```
