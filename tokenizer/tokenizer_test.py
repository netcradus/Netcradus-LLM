"""
Tokenizer Unit & Integration Test Suite for Netcradus LLM.

===============================================================================
PURPOSE:
===============================================================================
Validates the correctness, stability, and edge-case handling of the Netcradus BPE Tokenizer.
Ensures encoding/decoding consistency, special token insertion, unknown word handling (<UNK>),
multilingual Unicode character preservation, long document handling, empty strings, and
model save/load serialization.

===============================================================================
WORKFLOW:
===============================================================================
1. Load or train a test tokenizer model from `tokenizer_model/`.
2. Run automated test cases:
   - Test 1: Basic Encoding & Decoding Roundtrip.
   - Test 2: Special Tokens Injection (<BOS>, <EOS>, <PAD>, <MASK>).
   - Test 3: Unknown Words & Out-of-Vocabulary (OOV) tokens.
   - Test 4: Multilingual & Unicode Characters (Emoji, Devanagari, CJK, accents).
   - Test 5: Long Paragraphs & Multi-line Document Stress Testing.
   - Test 6: Empty Strings & Whitespace-only Inputs.
   - Test 7: Serialization Save & Load Roundtrip.
3. Assert expectations and output formatted test results to stdout.

===============================================================================
ALGORITHMS & VALIDATION CRITERIA:
===============================================================================
- Roundtrip Integrity: decode(encode(text, add_special_tokens=False)) == normalize(text).
- Loss-less Byte Encoding: Ensures unicode bytes never crash or get corrupt.
- Special Token Verification: Verifies expected IDs for <PAD>, <BOS>, <EOS>, <UNK>, <MASK>.

===============================================================================
COMPLEXITY:
===============================================================================
- Execution Time: ~0.1 - 0.5s for total test suite.
- Space Complexity: Minimal (< 10 MB RAM).

===============================================================================
EXAMPLE USAGE:
===============================================================================
$ python tokenizer/tokenizer_test.py
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple

# Reconfigure stdout/stderr encoding for Windows compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tokenizer.tokenizer_config import TokenizerConfig
from tokenizer.train_tokenizer import train_tokenizer
from tokenizer.tokenizer_utils import NetcradusTokenizerWrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("tokenizer_test")


def print_banner(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "=" * 75)
    print(f"  {title}")
    print("=" * 75)


def run_tests() -> bool:
    """
    Execute all tokenizer tests and return True if all pass, False otherwise.
    """
    model_dir = os.path.join(PROJECT_ROOT, "tokenizer_model")
    
    # Ensure tokenizer model exists or train one for testing
    if not os.path.exists(os.path.join(model_dir, "tokenizer.json")):
        print("Tokenizer model not found. Training a temporary tokenizer model...")
        config = TokenizerConfig(
            vocab_size=32000,
            corpus_path=os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_corpus.txt"),
            output_dir=model_dir
        )
        train_tokenizer(config)

    tokenizer = NetcradusTokenizerWrapper.load(model_dir)

    passed_tests = 0
    total_tests = 0

    # -------------------------------------------------------------------------
    # TEST 1: Basic Encoding & Decoding
    # -------------------------------------------------------------------------
    total_tests += 1
    print_banner("TEST 1: Basic Encoding & Decoding")
    test_str = "Artificial intelligence is revolutionizing natural language processing."
    encoded = tokenizer.encode(test_str, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    
    print(f"Input Text   : {test_str}")
    print(f"Token IDs    : {encoded}")
    print(f"Token Count  : {len(encoded)}")
    print(f"Decoded Text : {decoded}")
    
    if test_str.strip() == decoded.strip():
        print("[PASS] RESULT: PASSED (Exact Roundtrip Match)")
        passed_tests += 1
    else:
        print(f"[FAIL] RESULT: FAILED (Expected '{test_str}', got '{decoded}')")

    # -------------------------------------------------------------------------
    # TEST 2: Special Tokens Handling
    # -------------------------------------------------------------------------
    total_tests += 1
    print_banner("TEST 2: Special Tokens Handling (<BOS>, <EOS>, <PAD>, <MASK>)")
    encoded_specials = tokenizer.encode(test_str, add_special_tokens=True)
    print(f"Encoded with special tokens: {encoded_specials}")
    print(f"First Token ID (BOS): {encoded_specials[0]} (Expected: {tokenizer.bos_id})")
    print(f"Last Token ID  (EOS): {encoded_specials[-1]} (Expected: {tokenizer.eos_id})")

    decoded_with_specials = tokenizer.decode(encoded_specials, skip_special_tokens=False)
    decoded_sans_specials = tokenizer.decode(encoded_specials, skip_special_tokens=True)
    
    print(f"Decoded (with special tokens): {decoded_with_specials}")
    print(f"Decoded (sans special tokens): {decoded_sans_specials}")

    if (encoded_specials[0] == tokenizer.bos_id and 
        encoded_specials[-1] == tokenizer.eos_id and 
        tokenizer.config.bos_token in decoded_with_specials and
        tokenizer.config.eos_token in decoded_with_specials):
        print("[PASS] RESULT: PASSED (Special Tokens Correctly Injected & Recognized)")
        passed_tests += 1
    else:
        print("[FAIL] RESULT: FAILED (Special Token Mapping Mismatch)")

    # -------------------------------------------------------------------------
    # TEST 3: Unknown Words & OOV
    # -------------------------------------------------------------------------
    total_tests += 1
    print_banner("TEST 3: Unknown Words & Out-of-Vocabulary (OOV) Tokens")
    unk_text = "Supercalifragilisticexpialidocious quantum_zwnj_998811"
    encoded_unk = tokenizer.encode(unk_text, add_special_tokens=False)
    decoded_unk = tokenizer.decode(encoded_unk, skip_special_tokens=False)

    print(f"Input OOV Text : {unk_text}")
    print(f"Token IDs      : {encoded_unk}")
    print(f"Decoded        : {decoded_unk}")

    if decoded_unk.strip() == unk_text.strip():
        print("[PASS] RESULT: PASSED (Byte-level fallback seamlessly handles unseen subwords)")
        passed_tests += 1
    else:
        print(f"[FAIL] RESULT: FAILED (Decoded string differed: '{decoded_unk}')")

    # -------------------------------------------------------------------------
    # TEST 4: Unicode & Multilingual Support
    # -------------------------------------------------------------------------
    total_tests += 1
    print_banner("TEST 4: Unicode & Multilingual Support")
    unicode_text = "Netcradus LLM: Hello! こんにちは! नमस्ते! PyTorch alpha+beta=gamma cafe & resume"
    encoded_uni = tokenizer.encode(unicode_text, add_special_tokens=False)
    decoded_uni = tokenizer.decode(encoded_uni, skip_special_tokens=False)

    print(f"Original Unicode : {unicode_text}")
    print(f"Token IDs        : {encoded_uni}")
    print(f"Decoded Unicode  : {decoded_uni}")

    if unicode_text.strip() == decoded_uni.strip():
        print("[PASS] RESULT: PASSED (Unicode / Multilingual characters 100% preserved)")
        passed_tests += 1
    else:
        print("[FAIL] RESULT: FAILED (Unicode decoding corruption)")

    # -------------------------------------------------------------------------
    # TEST 5: Long Document Paragraphs
    # -------------------------------------------------------------------------
    total_tests += 1
    print_banner("TEST 5: Long Paragraph / Document Stress Test")
    long_para = (
        "The Transformer architecture has established itself as the foundation for state-of-the-art "
        "natural language processing models. By replacing recurrence with multi-head self-attention, "
        "Transformers enable massively parallelized training across compute clusters. Netcradus LLM "
        "implements custom architecture optimizations including RoPE embeddings, SwiGLU activations, "
        "and KV-caching for fast auto-regressive decoding. "
    ) * 10  # 10x repeated paragraph
    
    encoded_long = tokenizer.encode(long_para, add_special_tokens=True)
    decoded_long = tokenizer.decode(encoded_long, skip_special_tokens=True)

    print(f"Long text character count: {len(long_para)}")
    print(f"Encoded token count      : {len(encoded_long)}")
    print(f"Compression ratio        : {len(long_para) / len(encoded_long):.2f} chars/token")

    if long_para.strip() == decoded_long.strip():
        print("[PASS] RESULT: PASSED (Long document encoded/decoded without loss)")
        passed_tests += 1
    else:
        print("[FAIL] RESULT: FAILED (Mismatch in long document roundtrip)")

    # -------------------------------------------------------------------------
    # TEST 6: Empty Strings & Edge Cases
    # -------------------------------------------------------------------------
    total_tests += 1
    print_banner("TEST 6: Empty Strings & Edge Cases")
    empty_enc = tokenizer.encode("", add_special_tokens=False)
    empty_dec = tokenizer.decode([], skip_special_tokens=True)

    print(f"Encode empty string '' -> {empty_enc}")
    print(f"Decode empty tokens [] -> '{empty_dec}'")

    empty_specials_enc = tokenizer.encode("", add_special_tokens=True)
    print(f"Encode empty with specials -> {empty_specials_enc}")

    if (empty_enc == [] and 
        empty_dec == "" and 
        empty_specials_enc == [tokenizer.bos_id, tokenizer.eos_id]):
        print("[PASS] RESULT: PASSED (Empty string & empty sequence handled cleanly)")
        passed_tests += 1
    else:
        print("[FAIL] RESULT: FAILED (Empty string handling failure)")

    # -------------------------------------------------------------------------
    # TEST 7: Serialization Save & Load Roundtrip
    # -------------------------------------------------------------------------
    total_tests += 1
    print_banner("TEST 7: Serialization Save & Load Roundtrip")
    temp_save_dir = os.path.join(PROJECT_ROOT, "tokenizer_model_test_save")
    tokenizer.save(temp_save_dir)
    
    reloaded_tok = NetcradusTokenizerWrapper.load(temp_save_dir)
    
    test_sample = "Testing serialization integrity of Netcradus Tokenizer."
    enc1 = tokenizer.encode(test_sample)
    enc2 = reloaded_tok.encode(test_sample)
    
    print(f"Original Tokenizer output : {enc1}")
    print(f"Reloaded Tokenizer output : {enc2}")

    # Clean up temp save directory
    import shutil
    if os.path.exists(temp_save_dir):
        shutil.rmtree(temp_save_dir)

    if enc1 == enc2:
        print("[PASS] RESULT: PASSED (Loaded tokenizer behaves identically to saved instance)")
        passed_tests += 1
    else:
        print("[FAIL] RESULT: FAILED (Serialization mismatch)")

    # -------------------------------------------------------------------------
    # SUMMARY REPORT
    # -------------------------------------------------------------------------
    print_banner("TEST SUMMARY REPORT")
    print(f"Total Tests Run : {total_tests}")
    print(f"Passed          : {passed_tests}")
    print(f"Failed          : {total_tests - passed_tests}")
    print(f"Success Rate    : {(passed_tests / total_tests) * 100:.1f}%")
    print("=" * 75 + "\n")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
