"""AdamW with muP per-layer learning rates and independent weight decay; WSD schedule."""
import math
from typing import Callable, Optional

import torch

# Parameters whose learning rate is scaled by 1 / width_ratio under muP: attention,
# dense FFN, router, experts, and the unembedding. Embeddings keep the base rate.
MUP_SCALED = ("attn", "ffn", "gate", "experts", "lm_head")


class AdamW(torch.optim.Optimizer):
    """AdamW where each parameter's lr is multiplied by `lr_scale` (1/width_ratio for
    muP hidden weights), and weight decay is independent of the learning rate:
    p <- p - weight_decay * p every step."""

    def __init__(self, named_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 width_ratio=1.0):
        groups = []
        for name, p in named_params:
            scale = 1.0 / width_ratio if any(k in name for k in MUP_SCALED) else 1.0
            groups.append({"params": [p], "name": name, "lr_scale": scale})
        super().__init__(groups, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] * group["lr_scale"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                t, m, v = state["t"], state["m"], state["v"]
                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.sub_(lr_t * m / (v + group["eps"]).sqrt())
                p.sub_(group["weight_decay"] * p)
                state["t"] = t + 1
        return loss


def wsd_schedule(step: int, peak_lr: float, warmup_steps: int, total_steps: int,
                 min_lr_ratio: float = 0.05, decay_frac: float = 0.1) -> float:
    """Warmup-Stable-Decay: linear warmup, constant, then sqrt cooldown to
    min_lr_ratio * peak over the final decay_frac of training."""
    decay_start = total_steps - total_steps * decay_frac
    min_lr = min_lr_ratio * peak_lr
    if step < warmup_steps:
        return step / warmup_steps * peak_lr
    if step < decay_start:
        return peak_lr
    if step <= total_steps:
        progress = (step - decay_start) / (total_steps - decay_start)
        return min_lr + (peak_lr - min_lr) * (1 - math.sqrt(progress))
    return min_lr
