import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union, Dict, Any

from netcradus_llm.config import NetcradusConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) with YaRN context extension support."""
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 500000.0,
        scaling_config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_config = scaling_config
        
        # Apply YaRN frequency scaling if configured
        if scaling_config and scaling_config.get("type") == "yarn":
            scaling_factor = scaling_config.get("factor", 1.0)
            base = base * (scaling_factor ** (dim / (dim - 2)))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # q, k shape: [batch, heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU) Feed-Forward Network."""
    def __init__(self, config: NetcradusConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) implementation with KV Cache support for fast inference.
    Supports 4:1 (or arbitrary) grouping ratio between Query Heads and KV Heads.
    """
    def __init__(self, config: NetcradusConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_config=config.rope_scaling
        )

    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads for GQA broadcasting."""
        batch_size, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch_size, num_kv_heads * n_rep, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = seq_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        
        # Apply RoPE to query and key
        if past_key_value is not None:
            # Shift rotary embeddings for new tokens
            cos_curr = cos[kv_seq_len - seq_len : kv_seq_len]
            sin_curr = sin[kv_seq_len - seq_len : kv_seq_len]
            q, k = apply_rotary_pos_emb(q, k, cos_curr, sin_curr)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos[:seq_len], sin[:seq_len])

        # KV Cache concatenation for autoregressive decoding
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_key_value = (k, v) if use_cache else None

        # Repeat KV heads for Grouped-Query Attention
        k = self.repeat_kv(k, self.num_key_value_groups)
        v = self.repeat_kv(v, self.num_key_value_groups)

        # Compute Scaled Dot-Product Attention
        if hasattr(F, 'scaled_dot_product_attention') and attention_mask is None:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=(seq_len > 1 and past_key_value is None))
        else:
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            elif seq_len > 1:
                # Causal mask for pretraining/prompt encoding matching (seq_len, kv_seq_len)
                causal_mask = torch.full((seq_len, kv_seq_len), float("-inf"), device=hidden_states.device)
                causal_mask = torch.triu(causal_mask, diagonal=1 + (kv_seq_len - seq_len)).unsqueeze(0).unsqueeze(0)
                attn_weights = attn_weights + causal_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        return output, new_key_value


class NetcradusDecoderLayer(nn.Module):
    """Netcradus Transformer Decoder Block."""
    def __init__(self, config: NetcradusConfig, layer_idx: int = 0):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-LN Self Attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = residual + hidden_states

        # Pre-LN SwiGLU MLP with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value


class NetcradusModel(nn.Module):
    """Core Decoder Transformer Backbone."""
    def __init__(self, config: NetcradusConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([NetcradusDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        hidden_states = self.embed_tokens(input_ids)
        next_decoder_cache = [] if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present_kv = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_kv,
                use_cache=use_cache
            )
            if use_cache and present_kv is not None:
                next_decoder_cache.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_decoder_cache


class NetcradusForCausalLM(nn.Module):
    """
    Netcradus Causal Language Model with Language Model Head,
    Autoregressive Generation, Nucleus/Top-K Sampling, and KV Caching.
    """
    def __init__(self, config: NetcradusConfig):
        super().__init__()
        self.config = config
        self.model = NetcradusModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Initialize weights using initializer_range
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Dict[str, Any]:
        hidden_states, next_cache = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits and labels for autoregressive next-token loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ignore_id = -100 if (shift_labels == -100).any() else self.config.pad_token_id
            loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_id)
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_cache
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Fast autoregressive text generation with KV caching, Temperature, Top-K, and Top-P sampling.
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        self.eval()
        generated = input_ids.clone()
        past_key_values = None
        curr_input_ids = input_ids

        for _ in range(max_new_tokens):
            outputs = self.forward(curr_input_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]
            past_key_values = outputs["past_key_values"]

            if temperature > 0:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=-1)
            curr_input_ids = next_token

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated
