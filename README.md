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
uv sync --extra data --extra hf --extra wandb   # or: pip install -e ".[data,hf,wandb]"
uv sync --extra eval          # OLMES, for scripts/eval_olmes.py
python scripts/prepare_data.py --split val
python scripts/prepare_data.py --split train   # FineWeb-Edu 10BT sample, GPT-2 tokenizer (~20 GB)
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
python scripts/train.py --config configs/16L/looped_moe.yaml --d_model 704 --budget 1e18
```

`d_ff` (8/3 · d_model rounded up to a multiple of 64), the number of heads (d_model / 64),
and the number of steps (from C = 6 N_active D, batch 16 × 1024 tokens) are derived.
Each run appends its final validation loss and parameter counts to `results/runs.jsonl`
and writes checkpoints to `checkpoints/<run name>/`.

## Reproducing the paper

Each sweep file expands into `train.py` commands. `sweep.py` prints them, so you can hand
them to a job scheduler, or runs them in sequence with `--run`.

| Paper | Commands |
|---|---|
| Fig. 2 μP transfer | `python scripts/sweep.py configs/transfer/sweep.yaml --run`, then `python scripts/plot_transfer.py` |
| Fig. 1, 3, 9, 10 isoFLOP scaling | `python scripts/sweep.py configs/16L/isoflop_sweep.yaml --run`, then `python scripts/fit_isoflops.py` |
| Tables 4, 6 benchmarks (6 seeds) | `python scripts/sweep.py configs/16L/seeds_sweep.yaml --run`, export each model with `upload_hf.py`, then `python scripts/eval_olmes.py --models ...` |
| Fig. 4, Table 7 early exit | `python scripts/early_exit.py --models base=... looped=... moe=... looped_moe=...` (compute-optimal models at 1e18) |
| Fig. 5 routing | `python scripts/routing.py --model <looped_moe checkpoint>` |
| Fig. 6 convergence | `python scripts/convergence.py --models base=... looped=... moe=... looped_moe=...` |
| Fig. 7 loop depth | `python scripts/sweep.py configs/16L/loop_depth_sweep.yaml --run`, then `early_exit.py` on the 8×2, 4×4, and 2×8 models |
| Appendix B, 32 layers | `python scripts/sweep.py configs/32L/lr_sweep.yaml --run`, `python scripts/sweep.py configs/32L/isoflop_sweep.yaml --run`, then `fit_isoflops.py`, `early_exit.py`, and `eval_olmes.py` on the compute-optimal models |

Compute-optimal widths at 1e18 FLOPs: 16 layers: Base d896, Looped d896, MoE d832,
Looped-MoE d704. 32 layers: Base d512, MoE d512, Looped 16×2 d640, Looped-MoE 16×2 d512,
Looped-MoE 8×4 (16 experts) d640.

## Training recipe

AdamW (β = 0.9, 0.999, ε = 1e-8) with independent weight decay 1e-4; Warmup-Stable-Decay
schedule (10% warmup, sqrt cooldown to 5% of peak over the final 10%); gradient clipping
at 1.0; batch 16 × 1024 tokens. MoE layers use top-2 token-choice routing with load
balancing (weight 0.01) and router z-loss (weight 0.001). μP base width 128 (16 layers,
peak lr 0.01) and 256 (32 layers, peak lr 0.005). Model initialization is seeded with 42;
`--seed` changes only the training data order.

## Hugging Face

```bash
python scripts/upload_hf.py checkpoints/<run>/final.pt --out exports/<run> [--repo <user>/<name>]
```

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("<user>/<name>", trust_remote_code=True)
```
