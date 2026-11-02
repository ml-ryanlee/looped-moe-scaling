"""Training-free early exit (Section 4.2, Figure 4, Table 7, Appendix B).

Each candidate exit point reads out the hidden state through the final norm and
unembedding. A token exits at the first exit point where its exit criterion falls below a
threshold tau; otherwise it runs the full forward pass. Sweeping tau traces compute saved
vs perplexity.

  exit points  non-looped models: after every layer 1 .. L-1
               looped models:     after every loop except the last (loop boundaries)
  criteria     entropy  H(p) of the read-out distribution (reported in the paper)
               margin   1 - (p_top1 - p_top2)
  compute      inference FLOPs per token, F(l) = 2 N(l) + 12 l H Q T (PaLM), where
               N(l) = l (4 d^2 + 3 d d_ff) + 2 V d for l layers run; compute saved is
               1 - E[F(exit layer)] / F(full depth)

Usage:
    uv run python scripts/early_exit.py --models base=checkpoints/<run>/final.pt \\
        looped_moe=checkpoints/<run>/final.pt --out_dir results/early_exit

Models are NAME=PATH, where PATH is a scripts/train.py checkpoint (*.pt) or a Hugging
Face repo/folder exported with scripts/upload_hf.py. Writes <out_dir>/<NAME>.npz with the
per-token criteria and NLLs and the swept curves, prints perplexity at 5/10/20/30% compute
saved, and saves <out_dir>/early_exit.png.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from looped_scaling.analysis import LayerRecorder, default_device, forward_logits, load_model, project
from looped_scaling.config import ModelConfig
from looped_scaling.data import chunk_order, get_batch

SAVED_LEVELS = [5, 10, 20, 30]
N_THRESHOLDS = 1600
CRITERIA = ("entropy", "margin")


def flops_per_token(m: ModelConfig, layers: int) -> float:
    n = layers * (4 * m.d_model ** 2 + 3 * m.d_model * m.d_ff) + 2 * m.vocab_size * m.d_model
    return 2 * n + 12 * layers * m.d_model * m.context_length


def exit_layers(m: ModelConfig) -> list[int]:
    if m.looped:
        return [m.num_unique_layers * r for r in range(1, m.num_loops)]
    return list(range(1, m.num_layers))


@torch.no_grad()
def collect(model, m, valset, order, args, device):
    """Per-token criteria and NLL at every exit point, and NLL of the full pass."""
    exits = exit_layers(m)
    rec = LayerRecorder(model)
    crit = {c: [] for c in CRITERIA}
    nll_exit, nll_full = [], []
    for b in range(args.num_batches):
        rec.clear()
        x, y = get_batch(valset, order, b, args.batch_size, m.context_length, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu"):
            logits = forward_logits(model, x)
        y = y.flatten()
        nll_full.append(F.cross_entropy(logits.float().flatten(0, 1), y, reduction="none").cpu())
        per_exit = {c: [] for c in CRITERIA}
        per_nll = []
        for layer in exits:
            lp = F.log_softmax(project(model, rec.hidden[layer - 1]).float().flatten(0, 1), dim=-1)
            p = lp.exp()
            per_nll.append(F.nll_loss(lp, y, reduction="none").cpu())
            per_exit["entropy"].append(-(p * lp).sum(-1).cpu())
            top2 = p.topk(2, dim=-1).values
            per_exit["margin"].append((1 - (top2[:, 0] - top2[:, 1])).cpu())
            del lp, p
        nll_exit.append(torch.stack(per_nll, 1))
        for c in CRITERIA:
            crit[c].append(torch.stack(per_exit[c], 1))
        if (b + 1) % 10 == 0:
            print(f"  batch {b + 1}/{args.num_batches}")
    rec.remove()
    return ({c: torch.cat(v).numpy() for c, v in crit.items()},
            torch.cat(nll_exit).numpy(), torch.cat(nll_full).numpy())


def sweep(criterion, nll_exit, nll_full, f_exit, f_full):
    """First-crossing exit for thresholds at evenly spaced exit-rate quantiles.
    Returns (compute saved %, perplexity) arrays.

    A token exits at the first exit point k with criterion[k] < tau. As tau rises, its
    exit moves earlier exactly when tau passes a new running minimum of its criterion.
    Each such crossing is an event that changes the token's NLL and FLOPs by a fixed
    amount, so sorting all events once gives every threshold's totals by cumulative sum.
    """
    n, num_exits = criterion.shape
    prev_min = np.concatenate([np.full((n, 1), np.inf),
                               np.minimum.accumulate(criterion, axis=1)[:, :-1]], axis=1)
    is_event = criterion < prev_min
    d_nll = np.zeros(nll_exit.shape)
    d_f = np.zeros(nll_exit.shape)
    next_nll, next_f = nll_full.astype(np.float64), np.full(n, f_full)
    for k in reversed(range(num_exits)):  # the later exit a crossing replaces
        ev = is_event[:, k]
        d_nll[ev, k] = nll_exit[ev, k] - next_nll[ev]
        d_f[ev, k] = f_exit[k] - next_f[ev]
        next_nll = np.where(ev, nll_exit[:, k], next_nll)
        next_f = np.where(ev, f_exit[k], next_f)
    order = np.argsort(criterion[is_event], kind="stable")
    thresholds = criterion[is_event][order]
    cum_nll = np.concatenate([[0.0], np.cumsum(d_nll[is_event][order])])
    cum_f = np.concatenate([[0.0], np.cumsum(d_f[is_event][order])])

    taus = np.quantile(criterion.min(axis=1), np.linspace(0, 1, N_THRESHOLDS + 1))
    idx = np.searchsorted(thresholds, taus, side="left")  # events with threshold < tau
    mean_nll = (nll_full.sum() + cum_nll[idx]) / n
    mean_f = (n * f_full + cum_f[idx]) / n
    return 100 * (1 - mean_f / f_full), np.exp(mean_nll)


def at_saved(saved, ppl, target):
    return ppl[np.argmin(np.abs(saved - target))]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, metavar="NAME=PATH")
    p.add_argument("--val_path", default="data/fw_valid_tokens.npy")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_batches", type=int, default=50)
    p.add_argument("--out_dir", default="results/early_exit")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    valset = np.load(args.val_path, mmap_mode="r")
    curves = {}
    for item in args.models:
        name, spec = item.split("=", 1)
        model, m = load_model(spec, args.device)
        order = chunk_order(len(valset), m.context_length, seed=42)
        print(f"{name}: {m.arch} {m.shape} d_model={m.d_model}, exits after layers {exit_layers(m)}")
        crit, nll_exit, nll_full = collect(model, m, valset, order, args, args.device)
        del model
        f_exit = np.array([flops_per_token(m, l) for l in exit_layers(m)], dtype=np.float64)
        f_full = flops_per_token(m, m.effective_layers)
        result = {"nll_exit": nll_exit, "nll_full": nll_full, "f_exit": f_exit, "f_full": f_full,
                  "exit_layers": np.array(exit_layers(m))}
        for c in CRITERIA:
            saved, ppl = sweep(crit[c], nll_exit, nll_full, f_exit, f_full)
            result.update({c: crit[c], f"{c}_saved": saved, f"{c}_ppl": ppl})
        np.savez(out / f"{name}.npz", **result)
        curves[name] = result

    header = "".join(f"{s:>9}%" for s in SAVED_LEVELS)
    for c in CRITERIA:
        print(f"\nPerplexity at compute saved ({c} criterion)\n{'model':<24}{'full':>9}{header}")
        for name, r in curves.items():
            vals = "".join(f"{at_saved(r[f'{c}_saved'], r[f'{c}_ppl'], s):>10.2f}" for s in SAVED_LEVELS)
            print(f"{name:<24}{np.exp(r['nll_full'].mean()):>9.2f}{vals}")

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for name, r in curves.items():
        keep = r["entropy_saved"] <= 40
        ax.plot(r["entropy_saved"][keep], r["entropy_ppl"][keep], label=name)
    ax.set_xlabel("Compute saved (%)")
    ax.set_ylabel("Perplexity")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.savefig(out / "early_exit.png", dpi=300, bbox_inches="tight")
    print(f"\nsaved {out}/early_exit.png")


if __name__ == "__main__":
    main()
