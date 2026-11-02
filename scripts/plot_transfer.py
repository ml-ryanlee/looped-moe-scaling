"""Plot the muP learning-rate transfer test (Figure 2): final validation loss vs learning
rate, one curve per width, one panel per architecture.

    python scripts/plot_transfer.py --results results/runs.jsonl

Reads the runs from configs/transfer/sweep.yaml (fixed-step runs of configs/transfer/*).
For each panel, prints the best learning rate per width and how far the loss at the
smallest width's best learning rate is from each width's own best.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PANELS = [("base", "Base (4 layers)"), ("looped", "Looped (2x2)"),
          ("moe", "MoE (4 layers)"), ("looped_moe", "Looped-MoE (2x2)")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/runs.jsonl")
    p.add_argument("--config_dir", default="configs/transfer")
    p.add_argument("--out", default="figures/transfer.png")
    args = p.parse_args()

    data = defaultdict(lambda: defaultdict(dict))  # arch -> d_model -> lr -> loss
    with open(args.results) as f:
        for line in f:
            r = json.loads(line)
            if Path(r["config"]).parent == Path(args.config_dir):
                data[r["arch"]][r["d_model"]][r["lr"]] = r["final_loss"]

    fig, axes = plt.subplots(1, len(PANELS), figsize=(4 * len(PANELS), 3.4), sharey=True)
    for ax, (arch, title) in zip(axes, PANELS):
        widths = sorted(data[arch])
        if not widths:
            ax.set_title(f"{title}\n(no runs)")
            continue
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(widths)))
        base_best = min(data[arch][widths[0]].items(), key=lambda kv: kv[1])[0]
        print(f"\n{title}: best lr at d{widths[0]} = {base_best:g}")
        for w, c in zip(widths, colors):
            lrs = sorted(data[arch][w])
            losses = [data[arch][w][lr] for lr in lrs]
            best_lr, best = min(zip(lrs, losses), key=lambda t: t[1])
            ax.plot(lrs, losses, "o-", color=c, markersize=3, label=f"d={w}")
            ax.plot(best_lr, best, "*", color=c, markersize=11, markeredgecolor="white")
            gap = data[arch][w].get(base_best)
            gap_str = f"{100 * (gap - best) / best:.2f}%" if gap is not None else "n/a"
            print(f"  d{w}: best lr {best_lr:g} (loss {best:.4f}); at lr {base_best:g}: +{gap_str}")
        ax.axvline(base_best, color="gray", linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Learning rate")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Validation loss")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
