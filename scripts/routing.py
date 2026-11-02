"""Expert routing across loop passes in a Looped-MoE model (Section 6.1, Figure 5).

For every unique layer, compares each token's top-k expert set on loop pass 1 with its set
on each later pass: identical, partially shared, or no shared expert.

    python scripts/routing.py --model checkpoints/<looped-moe run>/final.pt --out_dir results/routing

Writes <out_dir>/routing.npz (expert ids, shape tokens x loops x layers x k) and
<out_dir>/routing_overlap.png, and prints the per-layer breakdown.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from looped_scaling.analysis import LayerRecorder, default_device, forward_logits, load_model
from looped_scaling.data import chunk_order, get_batch


@torch.no_grad()
def collect(model, m, valset, args, device):
    order = chunk_order(len(valset), m.context_length, seed=42)
    rec = LayerRecorder(model, routers=True)
    batches = []
    for b in range(args.num_batches):
        rec.clear()
        x, _ = get_batch(valset, order, b, args.batch_size, m.context_length, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device != "cpu"):
            forward_logits(model, x)
        # router calls run in execution order: loop 1 layers 1..U, loop 2 layers 1..U, ...
        e = torch.stack(rec.experts).cpu()                     # (loops*U, batch, seq, k)
        e = e.view(m.num_loops, m.num_unique_layers, *e.shape[1:])
        batches.append(e.flatten(2, 3).permute(2, 0, 1, 3))    # (tokens, loops, U, k)
    rec.remove()
    return torch.cat(batches).numpy()


def overlap(a, b):
    """Number of experts shared between two top-k sets, per token."""
    return (a[..., :, None] == b[..., None, :]).any(-1).sum(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--val_path", default="data/fw_valid_tokens.npy")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_batches", type=int, default=50)
    p.add_argument("--out_dir", default="results/routing")
    p.add_argument("--device", default=default_device())
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, m = load_model(args.model, args.device)
    assert m.arch == "looped_moe", "routing analysis needs a Looped-MoE model"
    valset = np.load(args.val_path, mmap_mode="r")
    experts = collect(model, m, valset, args, args.device)
    np.savez(out / "routing.npz", experts=experts)

    k = m.num_active
    layers = np.arange(1, m.num_unique_layers + 1)
    fig, axes = plt.subplots(1, m.num_loops - 1, figsize=(6 * (m.num_loops - 1), 3.6), squeeze=False)
    for r in range(1, m.num_loops):
        shared = overlap(experts[:, 0], experts[:, r])           # (tokens, U)
        exact, none = (shared == k).mean(0), (shared == 0).mean(0)
        partial = 1 - exact - none
        print(f"\npass 1 vs pass {r + 1}\n{'layer':>6}{'identical':>11}{'partial':>10}{'none':>8}")
        for i in range(m.num_unique_layers):
            print(f"{i + 1:>6}{exact[i]:>11.1%}{partial[i]:>10.1%}{none[i]:>8.1%}")
        ax, w = axes[0, r - 1], 0.27
        ax.bar(layers - w, exact, w, label="identical experts")
        ax.bar(layers, partial, w, label="partially shared")
        ax.bar(layers + w, none, w, label="no shared expert")
        ax.set(xlabel="Unique layer", ylabel="Fraction of tokens", title=f"Pass 1 vs pass {r + 1}",
               xticks=layers)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "routing_overlap.png", dpi=300)
    print(f"\nsaved {out}/routing_overlap.png")


if __name__ == "__main__":
    main()
