import json
# pyrefly: ignore [missing-import]
import torch
from typing import List, Dict, Any, Optional

from netcradus_llm.config import NetcradusConfig
from netcradus_llm.model import NetcradusForCausalLM
from netcradus_llm.tokenizer import NetcradusTokenizer


class NetcradusPipeline:
    """
    High-level Inference Pipeline for Netcradus LLM.
    Supports chat dialogue, code completion (FIM), tool function execution, and JSON parsing.
    """
    def __init__(self, model: NetcradusForCausalLM, tokenizer: NetcradusTokenizer, device: str = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> str:
        """Generate raw text response given prompt string."""
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        gen_tokens = output_ids[0][len(tokens):].tolist()
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7
    ) -> str:
        """Generate assistant response for multi-turn chat messages."""
        formatted_prompt = self.tokenizer.format_chat_prompt(messages)
        return self.generate(formatted_prompt, max_new_tokens=max_new_tokens, temperature=temperature)

    def fill_middle(
        self,
        prefix: str,
        suffix: str,
        max_new_tokens: int = 64
    ) -> str:
        """Fill in the middle (FIM) code completion."""
        fim_prompt = self.tokenizer.format_fim_prompt(prefix, suffix)
        return self.generate(fim_prompt, max_new_tokens=max_new_tokens, temperature=0.2)

    def call_tool(
        self,
        user_query: str,
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute deterministic Tool Calling by injecting tool definitions and JSON schema instructions into the prompt.
        """
        system_prompt = (
            "You are a tool-calling assistant. Answer queries by selecting a tool and providing a JSON object matching its parameters.\n"
            f"Available Tools: {json.dumps(tools, indent=2)}\n"
            'Output format: {"tool": "tool_name", "parameters": {...}}'
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        response = self.chat(messages, max_new_tokens=100, temperature=0.1)

        # Parse JSON output from model text
        try:
            # Extract JSON block if wrapped in markdown
            if "{" in response and "}" in response:
                start = response.index("{")
                end = response.rindex("}") + 1
                json_str = response[start:end]
                return json.loads(json_str)
        except Exception:
            pass

        return {"raw_response": response}
