import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

@dataclass
class NetcradusConfig:
    """
    Configuration class for Netcradus LLM architecture.
    Default settings align with the 7.2B / prototype baseline model spec.
    """
    vocab_size: int = 128000
    hidden_size: int = 4096
    intermediate_size: int = 11008  # SwiGLU FFN hidden dimension
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8    # Grouped-Query Attention (GQA 4:1 ratio)
    head_dim: Optional[int] = None  # Inferred as hidden_size // num_attention_heads if None
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072  # 128K Base Context
    rope_theta: float = 500000.0           # Base RoPE Frequency
    rope_scaling: Optional[Dict[str, Any]] = field(default_factory=lambda: {
        "type": "yarn",
        "factor": 4.0,                      # Extends context to 256K via YaRN
        "original_max_position_embeddings": 32768
    })
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Special Token IDs for FIM and Chat
    fim_prefix_id: int = 3
    fim_middle_id: int = 4
    fim_suffix_id: int = 5
    im_start_id: int = 6
    im_end_id: int = 7

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json_file(self, json_file_path: str):
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NetcradusConfig":
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, json_file_path: str) -> "NetcradusConfig":
        with open(json_file_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Prototype / Micro Configuration for quick local testing and training
PROTOTYPE_CONFIG = NetcradusConfig(
    vocab_size=128000,
    hidden_size=512,
    intermediate_size=1376,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=4096,
    rope_theta=10000.0,
    rope_scaling=None
)
