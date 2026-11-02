"""Construct a model from a ModelConfig, and load one from a checkpoint."""
import torch

from .config import ModelConfig, RunConfig, from_dict
from .models import LoopedTransformer, Transformer


def build_model(m: ModelConfig, device=None, dtype=None):
    common = dict(vocab_size=m.vocab_size, context_length=m.context_length, d_model=m.d_model,
                  num_heads=m.num_heads, d_ff=m.d_ff, rope_theta=m.rope_theta,
                  base_d_model=m.base_d_model, base_d_ff=m.base_d_ff, moe=m.moe,
                  num_experts=m.num_experts if m.moe else None,
                  num_active=m.num_active if m.moe else None, device=device, dtype=dtype)
    if m.looped:
        return LoopedTransformer(num_unique_layers=m.num_unique_layers, num_loops=m.num_loops, **common)
    return Transformer(num_layers=m.num_layers, **common)


def load_checkpoint_model(path: str, device="cpu"):
    """Load a model saved by scripts/train.py. Returns (model, RunConfig)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg: RunConfig = from_dict(ckpt["config"])
    model = build_model(cfg.model)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg
