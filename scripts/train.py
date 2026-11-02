"""Train one model from a config file.

The width and the compute budget (or a fixed step count) are usually set on the command
line so that one config serves a whole sweep:

    uv run python scripts/train.py --config configs/16L/looped_moe.yaml --d_model 704 --budget 1e18
    uv run python scripts/train.py --config configs/16L/base.yaml --d_model 896 --budget 1e18 --seed 43
    uv run python scripts/train.py --config configs/transfer/base.yaml --d_model 512 --steps 12500 --lr 0.01

Each finished run appends one JSON line (config, parameter counts, final validation loss)
to --results, which scripts/fit_isoflops.py and scripts/plot_transfer.py read.
Checkpoints are written to --out_dir/<run name>/ (pre-decay and final).
"""
import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from looped_scaling.build import build_model
from looped_scaling.config import count_params, load_config
from looped_scaling.data import chunk_order, get_batch
from looped_scaling.optim import AdamW, wsd_schedule

VAL_ORDER_SEED = 42  # identical validation batches for every run


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--d_model", type=int)
    p.add_argument("--budget", type=float, help="training FLOPs (C = 6 N_active D)")
    p.add_argument("--steps", type=int, help="fixed step count (instead of --budget)")
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int, help="data-order seed (model init seed stays fixed)")
    p.add_argument("--eval_batches", type=int, help="validation batches for the reported loss")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--results", default="results/runs.jsonl")
    p.add_argument("--resume", action="store_true", help="resume from <run>/latest.pt")
    p.add_argument("--save_every", type=int, default=0, help="also save latest.pt every N steps")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, valset, order, cfg, device, autocast):
    model.eval()
    losses = []
    for b in range(cfg.train.eval_batches):
        x, y = get_batch(valset, order, b, cfg.train.batch_size, cfg.model.context_length, device)
        with autocast():
            logits, _, _ = model(x)
        losses.append(F.cross_entropy(logits.float().flatten(0, 1), y.flatten()).item())
    model.train()
    return float(np.mean(losses))


def save(path, model, optimizer, step, cfg, params):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "config": cfg.to_dict(), "params": params}, path)


def main():
    args = parse_args()
    overrides = {"model.d_model": args.d_model, "train.lr": args.lr, "train.data_seed": args.seed,
                 "train.eval_batches": args.eval_batches}
    if args.budget is not None:
        overrides.update({"flop_budget": args.budget, "steps": None})
    if args.steps is not None:
        overrides.update({"steps": args.steps, "flop_budget": None})
    cfg = load_config(args.config, **overrides)
    m, t = cfg.model, cfg.train
    device = args.device
    params = count_params(m)
    run_dir = Path(args.out_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    warmup = int(t.warmup_frac * cfg.steps)
    decay_start = math.ceil(cfg.steps * (1 - t.decay_frac))
    print(f"run {cfg.name}\n  {m.arch} {m.shape}  d_model={m.d_model} d_ff={m.d_ff} heads={m.num_heads}"
          f"  muP base={m.base_d_model}/{m.base_d_ff}\n  N_active={params['active']:,}"
          f"  N_stored={params['stored']:,}  steps={cfg.steps:,}  tokens={cfg.tokens:,}")

    torch.manual_seed(t.init_seed)
    model = build_model(m, device=device)
    optimizer = AdamW(model.named_parameters(), lr=t.lr, betas=(t.beta1, t.beta2), eps=t.eps,
                      weight_decay=t.weight_decay, width_ratio=m.d_model / m.base_d_model)
    autocast = ((lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
                if t.precision == "bf16" else nullcontext)

    start = 0
    if args.resume and (run_dir / "latest.pt").exists():
        ckpt = torch.load(run_dir / "latest.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start = ckpt["step"]
        print(f"  resumed at step {start}")

    trainset = np.load(t.train_path, mmap_mode="r")
    valset = np.load(t.val_path, mmap_mode="r")
    train_order = chunk_order(len(trainset), m.context_length, seed=t.data_seed)
    val_order = chunk_order(len(valset), m.context_length, seed=VAL_ORDER_SEED)

    wandb = None
    if args.wandb:
        import wandb
        wandb.init(project=t.wandb_project or "loop-moe-scaling", name=cfg.name,
                   config={**cfg.to_dict(), "params": params}, resume="allow" if args.resume else None)

    model.train()
    val = None
    t0 = time.time()
    for step in range(start, cfg.steps):
        x, y = get_batch(trainset, train_order, step, t.batch_size, m.context_length, device)
        with autocast():
            logits, lb, lz = model(x)
            ce = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
            loss = ce + (t.lb_loss_weight * lb + t.z_loss_weight * lz if m.moe else 0.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        lr = wsd_schedule(step, t.lr, warmup, cfg.steps, t.min_lr_ratio, t.decay_frac)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()

        done = step + 1
        if wandb and done % 10 == 0:
            wandb.log({"train/loss": loss.item(), "train/ce": ce.item(), "lr": lr}, step=done)
        if done % t.eval_interval == 0 or done == cfg.steps:
            val = evaluate(model, valset, val_order, cfg, device, autocast)
            print(f"  step {done:>7}/{cfg.steps}  loss {ce.item():.4f}  val {val:.4f}  "
                  f"lr {lr:.2e}  {time.time() - t0:.0f}s")
            if wandb:
                wandb.log({"val/loss": val}, step=done)
        if done == decay_start:
            save(run_dir / "pre_decay.pt", model, optimizer, done, cfg, params)
        if args.save_every and done % args.save_every == 0:
            save(run_dir / "latest.pt", model, optimizer, done, cfg, params)

    if val is None:
        val = evaluate(model, valset, val_order, cfg, device, autocast)
    save(run_dir / "final.pt", model, optimizer, cfg.steps, cfg, params)
    record = {"name": cfg.name, "arch": m.arch, "shape": m.shape, "d_model": m.d_model,
              "num_experts": m.num_experts if m.moe else None, "lr": t.lr, "data_seed": t.data_seed,
              "flop_budget": cfg.flop_budget, "steps": cfg.steps, "tokens": cfg.tokens,
              "active_params": params["active"], "stored_params": params["stored"],
              "final_loss": val, "config": args.config}
    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(record) + "\n")
    if wandb:
        wandb.summary.update({"final_loss": val, **params})
        wandb.finish()
    print(f"done: final val loss {val:.4f} -> {args.results}")


if __name__ == "__main__":
    main()
