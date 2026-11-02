"""Shared helpers for the early-exit, convergence, and routing analyses: model loading and
reading out every effective layer through the final norm and unembedding."""
import torch

from .build import load_checkpoint_model
from .config import ModelConfig


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(spec: str, device: str):
    """Load a model from a scripts/train.py checkpoint (*.pt) or a Hugging Face repo/folder
    exported with scripts/upload_hf.py. Returns (plain PyTorch model, ModelConfig)."""
    if spec.endswith(".pt"):
        model, cfg = load_checkpoint_model(spec, device)
        return model, cfg.model
    from transformers import AutoModelForCausalLM
    wrapper = AutoModelForCausalLM.from_pretrained(spec, trust_remote_code=True)
    return wrapper.model.to(device).eval(), ModelConfig(**wrapper.config.model)


def unique_layers(model):
    """(list of unique blocks, number of loops) for any of the four architectures."""
    if hasattr(model, "stack"):
        return list(model.stack.layers), model.num_stacks
    return list(model.layers), 1


def project(model, hidden: torch.Tensor) -> torch.Tensor:
    """Read out a hidden state through the final norm and unembedding (logit lens)."""
    return model.lm_head(model.ln_final(hidden))


class LayerRecorder:
    """Records the output of every effective layer during a forward pass, in execution
    order: entry i is the hidden state after effective layer i + 1. Optionally records
    each MoE router's top-k expert ids the same way."""

    def __init__(self, model, routers: bool = False):
        self.hidden, self.experts = [], []
        self.handles = []
        layers, self.num_loops = unique_layers(model)
        self.num_unique = len(layers)
        for layer in layers:
            self.handles.append(layer.register_forward_hook(self._save_hidden))
            if routers:
                self.handles.append(layer.router.register_forward_hook(self._save_router))

    def _save_hidden(self, _m, _inp, out):
        self.hidden.append((out[0] if isinstance(out, tuple) else out).detach())

    def _save_router(self, _m, _inp, out):
        self.experts.append(out[3].detach())  # (batch, seq, k)

    def clear(self):
        self.hidden.clear()
        self.experts.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


def forward_logits(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out
