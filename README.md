# Sparse Layers are Critical to Scaling Looped Language Models

Code for the paper by Ryan Lee, Jacob Biloki, Edward J. Hu, and Jonathan May.

We compare four decoder-only transformers that differ only in whether the FFN is dense or
a top-k Mixture-of-Experts, and whether a stack of layers is looped through depth:

| Architecture | FFN | Layers (16-layer study) |
|---|---|---|
| Base | dense | 16 |
| Looped | dense | 8 × 2 |
| MoE | top-2 of 8 experts | 16 |
| Looped-MoE | top-2 of 8 experts | 8 × 2 |

All models use μP, so the learning rate tuned at a small base width transfers across
widths, and are compared at equal training compute (C = 6 N_active D).

## Setup

```bash
uv sync --extra data --extra hf --extra wandb
# OLMES (for scripts/eval_olmes.py) has conflicting pins, so install it as a separate tool.
# This fork of allenai/olmes supports torch>=2.10 (needed for grouped_mm).
uv tool install --python 3.11 "ai2-olmes @ git+https://github.com/ml-ryanlee/olmes.git@3897a974d2881a4e4c4a6af6b2dc3550b6881be3"
uv run python scripts/prepare_data.py --split val
uv run python scripts/prepare_data.py --split train   # FineWeb-Edu 10BT sample, GPT-2 tokenizer (~20 GB)
```

## Layout

```
looped_scaling/
  models.py        Base / Looped / MoE / Looped-MoE transformers (μP)
  config.py        YAML configs; d_ff, heads, and step counts derived from width and budget
  build.py         build a model from a config; load a checkpoint
  optim.py         AdamW with μP learning-rate scaling and independent weight decay; WSD schedule
  data.py          token-chunk data loader
  analysis.py      shared helpers for the analysis scripts
  hf_modeling.py   Hugging Face wrapper (trust_remote_code)
scripts/
  prepare_data.py  tokenize FineWeb-Edu
  train.py         train one model
  sweep.py         expand a sweep file into train.py commands
  fit_isoflops.py  isoFLOP fits and scaling-law figures
  plot_transfer.py μP transfer figure
  early_exit.py    training-free early exit
  convergence.py   JSD of intermediate layers to the final output
  routing.py       expert routing overlap across loop passes
  eval_olmes.py    OLMES Core 9 / Core 8
  upload_hf.py     export a checkpoint to Hugging Face format and push it
configs/
  transfer/        μP transfer test (4 effective layers)
  16L/             main study: 4 architectures, loop-depth variants, sweeps
  32L/             32-layer study (Appendix B)
```

A config file describes one architecture and its recipe; the width and the compute budget
are chosen per run:

```bash
uv run python scripts/train.py --config configs/16L/looped_moe.yaml --d_model 704 --budget 1e18
```

`d_ff` (8/3 · d_model rounded up to a multiple of 64), the number of heads (d_model / 64),
and the number of steps (from C = 6 N_active D, batch 16 × 1024 tokens) are derived.
Each run appends its final validation loss and parameter counts to `results/runs.jsonl`
and writes checkpoints to `checkpoints/<run name>/`.

## Reproducing the paper

Training runs are defined by sweep files; `uv run python scripts/sweep.py <sweep> --run`
runs one (without `--run` it prints the `train.py` commands for a job scheduler).

| Sweep | Paper | Then |
|---|---|---|
| `configs/transfer/sweep.yaml` | Fig. 2 | `plot_transfer.py` |
| `configs/16L/isoflop_sweep.yaml` | Figs. 1, 3, 9, 10 | `fit_isoflops.py` |
| `configs/16L/seeds_sweep.yaml` | Tables 4, 6 | `upload_hf.py`, `eval_olmes.py` |
| `configs/16L/loop_depth_sweep.yaml` | Fig. 7 | `early_exit.py` |
| `configs/32L/lr_sweep.yaml`, `configs/32L/isoflop_sweep.yaml` | Appendix B | `fit_isoflops.py`, `early_exit.py`, `eval_olmes.py` |

The analyses (`early_exit.py`, Figs. 4 and 7, Table 7; `convergence.py`, Fig. 6;
`routing.py`, Fig. 5) run on the compute-optimal models at 1e18 FLOPs: 16 layers: Base
d896, Looped d896, MoE d832, Looped-MoE d704; 32 layers: Base d512, MoE d512, Looped 16×2
d640, Looped-MoE 16×2 d512, Looped-MoE 8×4 (16 experts) d640.

## Training recipe

AdamW (β = 0.9, 0.999, ε = 1e-8) with independent weight decay 1e-4; Warmup-Stable-Decay
schedule (10% warmup, sqrt cooldown to 5% of peak over the final 10%); gradient clipping
at 1.0; batch 16 × 1024 tokens. MoE layers use top-2 token-choice routing with load
balancing (weight 0.01) and router z-loss (weight 0.001). μP base width 128 (16 layers,
peak lr 0.01) and 256 (32 layers, peak lr 0.005). Model initialization is seeded with 42;
`--seed` changes only the training data order.

## Hugging Face

```bash
uv run python scripts/upload_hf.py checkpoints/<run>/final.pt --out exports/<run> [--repo <user>/<name>]
```

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("<user>/<name>", trust_remote_code=True)
```
