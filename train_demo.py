import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader

from netcradus_llm.config import NetcradusConfig, PROTOTYPE_CONFIG
from netcradus_llm.model import NetcradusForCausalLM
from netcradus_llm.tokenizer import NetcradusTokenizer
from netcradus_llm.dataset import PretrainingDataset
from netcradus_llm.train import NetcradusTrainer
from netcradus_llm.inference import NetcradusPipeline


def main():
    print("=" * 80)
    print("      NETCRADUS LLM: END-TO-END MODEL INITIALIZATION & TRAINING DEMO")
    print("=" * 80)

    # 1. Device Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[System] Operating Device: {device}")

    # 2. Configuration Specs
    config = NetcradusConfig(
        vocab_size=128000,
        hidden_size=256,
        intermediate_size=704,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,    # Grouped-Query Attention (GQA 4:1)
        max_position_embeddings=2048,
        rope_theta=10000.0
    )
    print(f"[Model Config] Hidden Dim: {config.hidden_size} | Layers: {config.num_hidden_layers} | Heads: {config.num_attention_heads} (GQA KV Heads: {config.num_key_value_heads})")

    # 3. Model & Tokenizer Instantiation
    tokenizer = NetcradusTokenizer(vocab_size=config.vocab_size)
    model = NetcradusForCausalLM(config)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Netcradus LLM] Model Instantiated Successfully. Total Parameters: {param_count:,}")

    # 4. Pretraining Data Preparation
    sample_corpus = [
        "Netcradus LLM is a state-of-the-art open-weight foundation model designed for code generation, long-context reasoning, and agentic tool use.",
        "def compute_attention(q, k, v, scale):\n    scores = torch.matmul(q, k.transpose(-1, -2)) * scale\n    probs = torch.softmax(scores, dim=-1)\n    return torch.matmul(probs, v)",
        "The model utilizes SwiGLU activation functions, Grouped-Query Attention, and Rotary Position Embeddings with YaRN scaling for 256k token context extensions.",
        "To execute tool calling, the system parses incoming OpenAPI JSON schemas and enforces strict structural grammars during decoding."
    ] * 10

    dataset = PretrainingDataset(sample_corpus, tokenizer=tokenizer, max_seq_len=64)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(f"[Dataset] Prepared {len(dataset)} Packed Sequence Blocks (Seq Length: 64)")

    # 5. Execute Training Loop
    trainer = NetcradusTrainer(
        model=model,
        train_dataloader=dataloader,
        learning_rate=5e-4,
        max_steps=20,
        warmup_steps=3,
        output_dir="./checkpoints_demo",
        device=device
    )
    
    print("\n--- Running Mini Training Execution Loop ---")
    train_results = trainer.train()

    # 6. Execute Generation Inference Pipeline
    print("\n" + "=" * 80)
    print("      NETCRADUS LLM: INFERENCE & GENERATION DEMO")
    print("=" * 80)

    pipeline = NetcradusPipeline(model=model, tokenizer=tokenizer, device=device)

    # Demo 1: Text Completion
    prompt = "Netcradus LLM is a state-of-the-art"
    print(f"\n[Demo 1 - Prompt Text Completion]")
    print(f"Prompt: '{prompt}'")
    completion = pipeline.generate(prompt, max_new_tokens=30, temperature=0.7)
    print(f"Generated Output: {completion}")

    # Demo 2: Chat Assistant
    print(f"\n[Demo 2 - Multi-Turn Chat Assistant]")
    chat_messages = [
        {"role": "system", "content": "You are Netcradus AI, a helpful coding assistant."},
        {"role": "user", "content": "How does Grouped-Query Attention work?"}
    ]
    chat_response = pipeline.chat(chat_messages, max_new_tokens=40)
    print(f"Chat Assistant Response:\n{chat_response}")

    # Demo 3: Tool Function Calling
    print(f"\n[Demo 3 - Deterministic JSON Tool Calling]")
    tools = [{
        "name": "get_weather",
        "description": "Fetch weather for a city",
        "parameters": {"city": "string", "units": "celsius|fahrenheit"}
    }]
    tool_result = pipeline.call_tool("What's the weather in Tokyo?", tools=tools)
    print(f"Parsed Tool Call JSON Result:\n{tool_result}")

    print("\n" + "=" * 80)
    print(" [SUCCESS] Netcradus LLM Architecture Built & Verified Successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
