"""Base, Looped, MoE, and Looped-MoE transformers in the Maximal Update Parameterization.

All four architectures share one decoder block (pre-RMSNorm, RoPE multi-head attention,
SwiGLU FFN) and differ only in whether the FFN is dense or a top-k MoE, and whether a
stack of unique layers is repeated ("looped") through depth.

muP: every hidden weight (attention, FFN, experts, router, unembedding) is initialized
with std = std_base / sqrt(width_ratio), where std_base is the Glorot std of the same
weight at the base width (base_d_model, base_d_ff) and width_ratio = d_model /
base_d_model. The optimizer scales the learning rate of the same weights by
1 / width_ratio (see optim.py). Attention logits are scaled by 1/d_k.

This module depends only on torch and einops so it can be shipped to the Hugging Face
Hub alongside hf_modeling.py for `trust_remote_code` loading.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce, repeat
from torch import Tensor
from torch.nn.functional import grouped_mm, silu
from torch.nn.functional import scaled_dot_product_attention as sdpa


def glorot_std(fan_in: int, fan_out: int) -> float:
    return math.sqrt(2 / (fan_in + fan_out))


def stable_softmax(logits: Tensor, dim: int) -> Tensor:
    logits = logits - logits.max(dim=dim, keepdim=True).values
    exps = torch.exp(logits)
    return exps / exps.sum(dim=dim, keepdim=True)


class Linear(nn.Module):
    """y = W x (no bias), muP-scaled truncated-normal init."""

    def __init__(self, in_features, out_features, width_ratio, std_base, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = std_base / math.sqrt(width_ratio)
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """RMSNorm without a learnable gain (muP)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms).to(in_dtype)


class SwiGLU(nn.Module):
    """FFN(x) = W2(SiLU(W1 x) * W3 x)."""

    def __init__(self, d_model, d_ff, width_ratio, base_d_model, base_d_ff, device=None, dtype=None):
        super().__init__()
        std_base = glorot_std(base_d_model, base_d_ff)
        self.w1 = Linear(d_model, d_ff, width_ratio, std_base, device, dtype)
        self.w2 = Linear(d_ff, d_model, width_ratio, std_base, device, dtype)
        self.w3 = Linear(d_model, d_ff, width_ratio, std_base, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        pos = torch.arange(max_seq_len, dtype=torch.float64)
        inv_freq = theta ** (-2 * torch.arange(d_k // 2, dtype=torch.float64) / d_k)
        angles = torch.outer(pos, inv_freq)                       # (seq, d_k/2)
        cos, sin = torch.cos(angles), torch.sin(angles)
        rotations = torch.stack([torch.stack([cos, -sin], -1),
                                 torch.stack([sin, cos], -1)], -2)  # (seq, d_k/2, 2, 2)
        self.register_buffer("rotations", rotations.to(device=device, dtype=dtype or torch.float32),
                             persistent=True)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        rot = self.rotations[token_positions]
        pairs = rearrange(x, "... seq (f i) -> ... seq f i", i=2)
        out = einsum(rot.to(x.dtype), pairs, "... seq f i j, ... seq f j -> ... seq f i")
        return rearrange(out, "... seq f i -> ... seq (f i)")


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta, width_ratio, base_d_model,
                 device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        std_base = glorot_std(base_d_model, base_d_model)
        self.q_proj = Linear(d_model, d_model, width_ratio, std_base, device, dtype)
        self.k_proj = Linear(d_model, d_model, width_ratio, std_base, device, dtype)
        self.v_proj = Linear(d_model, d_model, width_ratio, std_base, device, dtype)
        self.output_proj = Linear(d_model, d_model, width_ratio, std_base, device, dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device)

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        seq = x.shape[-2]
        d_k = self.d_model // self.num_heads
        q = rearrange(self.q_proj(x), "... seq (h d) -> ... h seq d", d=d_k)
        k = rearrange(self.k_proj(x), "... seq (h d) -> ... h seq d", d=d_k)
        v = rearrange(self.v_proj(x), "... seq (h d) -> ... h seq d", d=d_k)
        if token_positions is None:
            token_positions = torch.arange(seq, device=x.device).unsqueeze(0)
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        # muP: scale attention logits by 1/d_k rather than 1/sqrt(d_k)
        out = sdpa(q, k, v, is_causal=True, scale=1.0 / d_k)
        return self.output_proj(rearrange(out, "... h seq d -> ... seq (h d)"))


class PrenormBlock(nn.Module):
    """Dense block: x + Attn(Norm(x)), then + SwiGLU(Norm(.))."""

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, width_ratio,
                 base_d_model, base_d_ff, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, width_ratio,
                                           base_d_model, device, dtype)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, width_ratio, base_d_model, base_d_ff, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class Router(nn.Module):
    """Top-k token-choice router; selected probabilities are renormalized to sum to 1."""

    def __init__(self, d_model, num_experts, num_active, width_ratio, base_d_model, device=None, dtype=None):
        super().__init__()
        self.gate = Linear(d_model, num_experts, width_ratio, glorot_std(base_d_model, num_experts),
                           device, dtype)
        self.num_active = num_active

    def forward(self, x: Tensor):
        logits = self.gate(x)
        probs = stable_softmax(logits, dim=-1)
        top_scores, top_experts = torch.topk(probs, k=self.num_active, dim=-1)
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)
        return logits, probs, top_scores, top_experts


class MoEPrenormBlock(nn.Module):
    """MoE block: the dense FFN is replaced by E SwiGLU experts of width d_ff / k.

    Experts are stored stacked, (E, d_in, d_out), and run with torch grouped_mm. Their
    init std is the Glorot std of a single expert at the base width,
    sqrt(2 / (base_d_model + base_d_ff / k)).

    Returns (output, load_balancing_loss, router_z_loss).
    """

    def __init__(self, d_model, num_heads, d_ff, num_experts, num_active, max_seq_len, theta,
                 width_ratio, base_d_model, base_d_ff, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, width_ratio,
                                           base_d_model, device, dtype)
        self.ln2 = RMSNorm(d_model)
        self.router = Router(d_model, num_experts, num_active, width_ratio, base_d_model, device, dtype)
        self.num_experts = num_experts
        self.num_active = num_active

        d_ff_expert = d_ff // num_active
        std = glorot_std(base_d_model, base_d_ff // num_active) / math.sqrt(width_ratio)

        def expert_weights(d_in, d_out):
            w = torch.empty(num_experts, d_in, d_out, device=device, dtype=dtype)
            nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3 * std, b=3 * std)
            return nn.Parameter(w)

        self.experts_w1 = expert_weights(d_model, d_ff_expert)
        self.experts_w2 = expert_weights(d_ff_expert, d_model)
        self.experts_w3 = expert_weights(d_model, d_ff_expert)

    def forward(self, x: Tensor):
        batch, seq, dim = x.shape
        n_tokens = batch * seq
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)

        logits, probs, top_scores, top_experts = self.router(h)

        # Sort (token, slot) pairs by expert so each expert sees a contiguous group.
        h_flat = rearrange(h, "b s d -> (b s) d")
        expert_ids = rearrange(top_experts, "b s k -> (b s k)")
        scores = rearrange(top_scores, "b s k -> (b s k)")
        token_ids = repeat(torch.arange(n_tokens, device=x.device), "n -> (n k)", k=self.num_active)
        order = expert_ids.argsort(stable=True)
        expert_ids, token_ids, scores = expert_ids[order], token_ids[order], scores[order]
        counts = torch.bincount(expert_ids, minlength=self.num_experts)
        offs = counts.cumsum(0).to(torch.int32)

        # grouped_mm is not covered by autocast: cast explicitly so expert matmuls run in
        # the autocast dtype (bf16) when mixed precision is on, and are a no-op otherwise.
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else h_flat.dtype
        xs = h_flat[token_ids].to(dt)
        h1 = grouped_mm(xs, self.experts_w1.to(dt), offs=offs)
        h3 = grouped_mm(xs, self.experts_w3.to(dt), offs=offs)
        out = grouped_mm((silu(h1) * h3).to(dt), self.experts_w2.to(dt), offs=offs)
        out = out * scores.unsqueeze(-1).to(out.dtype)
        combined = torch.zeros(n_tokens, dim, device=x.device, dtype=out.dtype)
        combined.index_add_(0, token_ids, out)
        x = x + rearrange(combined, "(b s) d -> b s d", b=batch, s=seq).to(x.dtype)

        # Load balancing (Switch) and router z-loss (ST-MoE)
        f = counts.float() / (n_tokens * self.num_active)
        p = reduce(probs, "b s e -> e", "mean")
        lb = self.num_experts * (f * p).sum()
        lz = (torch.logsumexp(logits.float(), dim=-1) ** 2).mean()
        return x, lb, lz


class _LMBase(nn.Module):
    """Embedding, final norm, and unembedding shared by all architectures."""

    def __init__(self, vocab_size, d_model, width_ratio, base_d_model, device=None, dtype=None):
        super().__init__()
        self.width_ratio = width_ratio
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, width_ratio, glorot_std(base_d_model, vocab_size),
                              device, dtype)

    def project(self, hidden: Tensor) -> Tensor:
        """Final norm + unembedding; also used to read out intermediate layers."""
        return self.lm_head(self.ln_final(hidden))


def _run_layers(layers, x):
    lb = lz = x.new_zeros(())
    for layer in layers:
        out = layer(x)
        if isinstance(out, tuple):
            x, l_b, l_z = out
            lb, lz = lb + l_b, lz + l_z
        else:
            x = out
    return x, lb, lz


class LoopedStack(nn.Module):
    """The stack of unique layers that a looped model repeats."""

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor):
        return _run_layers(self.layers, x)


class Transformer(_LMBase):
    """Base (moe=False) or MoE (moe=True) transformer with num_layers unique layers.

    forward returns (logits, lb, lz); lb and lz are averaged over layers (zero if dense).
    """

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                 base_d_model, base_d_ff, moe=False, num_experts=None, num_active=None,
                 device=None, dtype=None):
        width_ratio = d_model / base_d_model
        super().__init__(vocab_size, d_model, width_ratio, base_d_model, device, dtype)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            _make_block(moe, d_model, num_heads, d_ff, num_experts, num_active, context_length,
                        rope_theta, width_ratio, base_d_model, base_d_ff, device, dtype)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids: Tensor):
        x, lb, lz = _run_layers(self.layers, self.token_embeddings(input_ids))
        return self.project(x), lb / self.num_layers, lz / self.num_layers


class LoopedTransformer(_LMBase):
    """Looped (moe=False) or Looped-MoE (moe=True) transformer: a stack of
    num_unique_layers layers applied num_loops times (effective depth = product).

    forward returns (logits, lb, lz); lb and lz are averaged over effective layers.
    """

    def __init__(self, vocab_size, context_length, d_model, num_unique_layers, num_loops, num_heads,
                 d_ff, rope_theta, base_d_model, base_d_ff, moe=False, num_experts=None,
                 num_active=None, device=None, dtype=None):
        width_ratio = d_model / base_d_model
        super().__init__(vocab_size, d_model, width_ratio, base_d_model, device, dtype)
        self.num_stacks = num_loops
        self.num_effective_layers = num_unique_layers * num_loops
        self.stack = LoopedStack(nn.ModuleList([
            _make_block(moe, d_model, num_heads, d_ff, num_experts, num_active, context_length,
                        rope_theta, width_ratio, base_d_model, base_d_ff, device, dtype)
            for _ in range(num_unique_layers)
        ]))

    def forward(self, input_ids: Tensor):
        x = self.token_embeddings(input_ids)
        lb = lz = x.new_zeros(())
        for _ in range(self.num_stacks):
            x, l_b, l_z = self.stack(x)
            lb, lz = lb + l_b, lz + l_z
        return self.project(x), lb / self.num_effective_layers, lz / self.num_effective_layers


def _make_block(moe, d_model, num_heads, d_ff, num_experts, num_active, context_length, rope_theta,
                width_ratio, base_d_model, base_d_ff, device, dtype):
    if moe:
        return MoEPrenormBlock(d_model, num_heads, d_ff, num_experts, num_active, context_length,
                               rope_theta, width_ratio, base_d_model, base_d_ff, device, dtype)
    return PrenormBlock(d_model, num_heads, d_ff, context_length, rope_theta, width_ratio,
                        base_d_model, base_d_ff, device, dtype)
