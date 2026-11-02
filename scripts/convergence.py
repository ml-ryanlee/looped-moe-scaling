"""Convergence of intermediate layers to the final output (Section 6.2, Figure 6).

Every effective layer's hidden state is read out through the final norm and unembedding,
and compared with the model's final output distribution by Jensen-Shannon divergence,
normalized to [0, 1] (divided by ln 2). A token counts as converged at a layer if its
normalized JSD is below --threshold (0.5).

    uv run python scripts/convergence.py --models base=checkpoints/<run>/final.pt \\
        looped=... moe=... looped_moe=... --out_dir results/convergence

Writes <out_dir>/<NAME>.npz (per-token JSD, shape tokens x effective layers) and
<out_dir>/convergence.png (left: mean JSD per layer, band = std across sequences;
right: fraction of tokens converged per layer).
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from looped_scaling.analysis import LayerRecorder, default_device, forward_logits, load_model, project
from looped_scaling.data import chunk_order, get_batch


def normalized_jsd(logits_a, logits_b):
    la, lb = F.log_softmax(logits_a, -1), F.log_softmax(logits_b, -1)
    pa, pb = la.exp(), lb.exp()
    lm = (0.5 * (pa + pb)).clamp_min(1e-40).log()
    jsd = 0.5 * ((pa * (la - lm)).sum(-1) + (pb * (lb - lm)).sum(-1))
    return jsd / math.log(2)


@torch.no_grad()
def collect(model, m, valset, args, device):
    order = chunk_order(len(valset), m.context_length, seed=42)
    rec = LayerRecorder(model)
    batches = []
    for b in range(args.num_batches):
        rec.clear()
        x, _ = get_batch(valset, order, b, args.batch_size, m.context_length, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu"):
            final = forward_logits(model, x).float()
        cols = [normalized_jsd(project(model, h).float(), final).cpu() for h in rec.hidden]
        batches.append(torch.stack(cols, -1))  # (batch, seq, layers)
        if (b + 1) % 10 == 0:
            print(f"  batch {b + 1}/{args.num_batches}")
    rec.remove()
    return torch.cat(batches).numpy()  # (sequences, seq, layers)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, metavar="NAME=PATH")
    p.add_argument("--val_path", default="data/fw_valid_tokens.npy")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_batches", type=int, default=50)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out_dir", default="results/convergence")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    valset = np.load(args.val_path, mmap_mode="r")
    fig, (ax_jsd, ax_frac) = plt.subplots(1, 2, figsize=(10, 3.6))
    for item in args.models:
        name, spec = item.split("=", 1)
        model, m = load_model(spec, args.device)
        print(f"{name}: {m.arch} {m.shape} d_model={m.d_model}")
        jsd = collect(model, m, valset, args, args.device)
        del model
        np.savez(out / f"{name}.npz", jsd=jsd.reshape(-1, jsd.shape[-1]))
        layers = np.arange(1, jsd.shape[-1] + 1)
        per_seq = jsd.mean(axis=1)                      # (sequences, layers)
        mean, std = per_seq.mean(0), per_seq.std(0)
        frac = (jsd < args.threshold).mean(axis=(0, 1))
        line, = ax_jsd.plot(layers, mean, marker="o", markersize=3, label=name)
        ax_jsd.fill_between(layers, mean - std, mean + std, color=line.get_color(), alpha=0.15)
        ax_frac.plot(layers, frac, marker="o", markersize=3, color=line.get_color(), label=name)
        if m.looped:
            for r in range(1, m.num_loops):
                ax_frac.axvline(m.num_unique_layers * r, color=line.get_color(), linestyle=":", alpha=0.5)
        print("  converged fraction by layer: " + " ".join(f"{f:.2f}" for f in frac))
    ax_jsd.set(xlabel="Effective layer", ylabel="Normalized JSD to final output")
    ax_frac.set(xlabel="Effective layer", ylabel=f"Fraction of tokens with JSD < {args.threshold}")
    ax_jsd.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=300)
    print(f"saved {out}/convergence.png")


if __name__ == "__main__":
    main()
