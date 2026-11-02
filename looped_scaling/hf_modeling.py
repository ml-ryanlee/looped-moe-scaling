"""Hugging Face wrapper so trained models can be shared on the Hub and evaluated with
standard tooling (e.g. OLMES) via `trust_remote_code=True`.

scripts/upload_hf.py copies this file together with models.py, config.py, and build.py
into the model repo; it imports them relatively, so the looped_scaling package does not
need to be installed to load a model from the Hub.
"""
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .build import build_model
from .config import ModelConfig
from .models import Transformer  # noqa: F401  (direct import so the Hub fetches models.py)


class LoopMoEConfig(PretrainedConfig):
    model_type = "loop-moe"

    def __init__(self, model: Optional[dict] = None, **kwargs):
        self.model = model or {"arch": "base"}
        m = ModelConfig(**self.model)
        kwargs.setdefault("vocab_size", m.vocab_size)
        kwargs.setdefault("max_length", m.context_length)  # read by eval harnesses
        super().__init__(**kwargs)

    @classmethod
    def from_model_config(cls, m: ModelConfig, **kwargs):
        return cls(model=asdict(m), **kwargs)


class LoopMoEForCausalLM(PreTrainedModel, GenerationMixin):
    """Causal LM over the four architectures. `self.model` is the plain PyTorch model
    from looped_scaling.models. Generation works without a KV cache (correct but slow)."""

    config_class = LoopMoEConfig
    base_model_prefix = "model"

    def __init__(self, config: LoopMoEConfig):
        super().__init__(config)
        self.model = build_model(ModelConfig(**config.model))
        self.post_init()

    def _init_weights(self, module):
        pass  # weights are initialized by the model itself (muP), then loaded

    def get_input_embeddings(self):
        return self.model.token_embeddings

    def forward(self, input_ids: torch.LongTensor, attention_mask=None,
                labels: Optional[torch.LongTensor] = None, **kwargs) -> CausalLMOutputWithPast:
        """attention_mask is ignored (causal attention is built in). If labels are given,
        returns the next-token cross-entropy with the usual one-position shift."""
        logits, _, _ = self.model(input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten())
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
