"""Run configuration: YAML loading, width-derived sizes, parameter counts, and FLOP budgets.

A config file describes one architecture and its training recipe. Width (d_model) and the
compute budget (or a fixed step count) are chosen per run, so one file serves a whole
isoFLOP sweep. Sizes that follow from the width are derived here:

    d_ff      = 8/3 * d_model, rounded up to a multiple of 64
    num_heads = d_model / 64
    steps     = round(C / (6 * N_active * batch_size * context_length))

N_active counts parameters used in one forward pass: looped layers once per loop, only the
k active experts of each MoE layer, and the embedding and unembedding matrices. The small
router weights are not counted.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields

import yaml

ARCHS = ("base", "looped", "moe", "looped_moe")


@dataclass
class ModelConfig:
    arch: str                      # base | looped | moe | looped_moe
    d_model: int = 128
    num_layers: int = 16           # unique layers (non-looped models)
    num_unique_layers: int = 8     # layers in the repeated stack (looped models)
    num_loops: int = 2             # passes through the stack (looped models)
    num_experts: int = 8
    num_active: int = 2
    vocab_size: int = 50257
    context_length: int = 1024
    rope_theta: float = 10000.0
    base_d_model: int = 128        # muP base width
    d_ff: int | None = None        # default: derived from d_model
    num_heads: int | None = None   # default: derived from d_model
    base_d_ff: int | None = None   # default: derived from base_d_model

    def __post_init__(self):
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {self.arch!r}")
        if self.d_ff is None:
            self.d_ff = ffn_width(self.d_model)
        if self.num_heads is None:
            self.num_heads = self.d_model // 64
        if self.base_d_ff is None:
            self.base_d_ff = ffn_width(self.base_d_model)

    @property
    def looped(self) -> bool:
        return self.arch in ("looped", "looped_moe")

    @property
    def moe(self) -> bool:
        return self.arch in ("moe", "looped_moe")

    @property
    def unique_layers(self) -> int:
        return self.num_unique_layers if self.looped else self.num_layers

    @property
    def effective_layers(self) -> int:
        return self.num_unique_layers * self.num_loops if self.looped else self.num_layers

    @property
    def shape(self) -> str:
        """e.g. '16', '8x2', '8x4'."""
        return f"{self.num_unique_layers}x{self.num_loops}" if self.looped else str(self.num_layers)


@dataclass
class TrainConfig:
    lr: float = 0.01
    batch_size: int = 16
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4     # independent (not multiplied by the learning rate)
    warmup_frac: float = 0.1
    decay_frac: float = 0.1        # WSD: sqrt cooldown over the final 10% of steps
    min_lr_ratio: float = 0.05     # cooldown to 5% of peak
    grad_clip: float = 1.0
    lb_loss_weight: float = 0.01   # load balancing (follows OLMoE)
    z_loss_weight: float = 0.001   # router z-loss (follows OLMoE)
    precision: str = "fp32"        # fp32 | bf16 (autocast)
    eval_batches: int = 50
    eval_interval: int = 500
    init_seed: int = 42            # model init; fixed across data seeds
    data_seed: int = 42            # training data order
    train_path: str = "data/fw_train_tokens.npy"
    val_path: str = "data/fw_valid_tokens.npy"
    wandb_project: str | None = None


@dataclass
class RunConfig:
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)
    flop_budget: float | None = None
    steps: int | None = None       # set directly, or derived from flop_budget

    def __post_init__(self):
        if self.steps is None:
            if self.flop_budget is None:
                raise ValueError("set either flop_budget or steps")
            self.steps = steps_for_budget(self.model, self.flop_budget, self.train.batch_size)

    @property
    def tokens(self) -> int:
        return self.steps * self.train.batch_size * self.model.context_length

    @property
    def name(self) -> str:
        m = self.model
        parts = [m.arch.replace("_", "-"), m.shape]
        if m.moe:
            parts.append(f"e{m.num_experts}k{m.num_active}")
        parts.append(f"d{m.d_model}")
        parts.append(format_budget(self.flop_budget) if self.flop_budget else f"{self.steps}steps")
        parts.append(f"lr{self.train.lr:g}")
        parts.append(f"seed{self.train.data_seed}")
        return "-".join(parts)

    def to_dict(self) -> dict:
        return {"model": asdict(self.model), "train": asdict(self.train),
                "flop_budget": self.flop_budget, "steps": self.steps}


def ffn_width(d_model: int) -> int:
    return int(math.ceil(8 * d_model / 3 / 64) * 64)


def count_params(m: ModelConfig) -> dict:
    """Active (used per forward pass) and stored (unique) parameter counts."""
    attn = 4 * m.d_model ** 2
    ffn_active = 3 * m.d_model * m.d_ff
    if m.moe:
        ffn_stored = m.num_experts * 3 * m.d_model * (m.d_ff // m.num_active)
        router = m.d_model * m.num_experts
    else:
        ffn_stored, router = ffn_active, 0
    embed = 2 * m.vocab_size * m.d_model
    return {
        "active": m.effective_layers * (attn + ffn_active) + embed,
        "stored": m.unique_layers * (attn + ffn_stored + router) + embed,
    }


def steps_for_budget(m: ModelConfig, flop_budget: float, batch_size: int) -> int:
    tokens = flop_budget / (6 * count_params(m)["active"])
    return round(tokens / (batch_size * m.context_length))


def format_budget(c: float) -> str:
    exp = int(math.floor(math.log10(c)))
    return f"{round(c / 10 ** exp)}e{exp}"


def load_config(path: str, **overrides) -> RunConfig:
    """Load a YAML config. Overrides use dotted keys, e.g. {"model.d_model": 704}."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = copy.deepcopy(raw)
    for key, value in overrides.items():
        if value is None:
            continue
        *path_keys, last = key.split(".")
        node = raw
        for k in path_keys:
            node = node.setdefault(k, {})
        node[last] = value
    return from_dict(raw)


def from_dict(raw: dict) -> RunConfig:
    model_raw = dict(raw["model"])
    # sizes derived from d_model are recomputed unless set explicitly for this run
    model = ModelConfig(**_known(ModelConfig, model_raw))
    train = TrainConfig(**_known(TrainConfig, raw.get("train", {})))
    budget = raw.get("flop_budget")
    return RunConfig(model=model, train=train,
                     flop_budget=float(budget) if budget is not None else None,
                     steps=raw.get("steps"))


def _known(cls, d: dict) -> dict:
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return {k: v for k, v in d.items() if k in names}
