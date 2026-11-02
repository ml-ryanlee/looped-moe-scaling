"""Fit isoFLOP curves and loss-vs-parameter scaling laws from training results.

For each architecture and compute budget, a quadratic in log(N_active) is fit to final
validation loss; its minimum is the compute-optimal (N*, L*) for that budget. A power law
L = a * N^(-alpha) is then fit through the compute-optimal points (linear fit in log-log
space).

    python scripts/fit_isoflops.py --results results/runs.jsonl
    python scripts/fit_isoflops.py --archs base looped_moe --shape 16 8x2

Outputs (in --out_dir):
    isoflops_<arch>.png    isoFLOP curves with compute-optimal points and the fitted law
    isoflops_combined.png  fitted scaling laws for all architectures together
and prints alpha and the compute-optimal points per architecture.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

KAPLAN_ALPHA = 0.076
MIN_POINTS = 3  # budgets with fewer widths cannot be fit
STYLE = {  # display name, color, line style
    "base": ("Base", "tab:blue", "-"),
    "looped": ("Looped", "tab:red", (0, (1, 1.5))),
    "moe": ("MoE", "tab:green", "--"),
    "looped_moe": ("Looped-MoE", "tab:orange", (0, (3, 1, 1, 1.5))),
}


def load_runs(path, archs, shapes):
    runs = defaultdict(lambda: defaultdict(list))  # arch -> budget -> [(N, loss)]
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["arch"] not in archs or not r.get("flop_budget"):
                continue
            if shapes and r["shape"] not in shapes:
                continue
            if r.get("num_experts") not in (None, 8):
                continue
            runs[r["arch"]][r["flop_budget"]].append((r["active_params"], r["final_loss"]))
    return runs


def fit_budget(points):
    """Quadratic fit of loss vs log N; returns (coeffs, log N*, L*)."""
    log_n = np.log([n for n, _ in points])
    loss = np.array([l for _, l in points])
    coeffs = np.polyfit(log_n, loss, 2)
    a, b, _ = coeffs
    log_n_star = -b / (2 * a) if a > 0 else log_n[np.argmin(loss)]
    return coeffs, log_n_star, float(np.polyval(coeffs, log_n_star))


def fit_power_law(n_opt, l_opt):
    slope, intercept = np.polyfit(np.log(n_opt), np.log(l_opt), 1)
    return np.exp(intercept), -slope  # L = a * N^-alpha


def fmt_params(x, _):
    return f"{x / 1e6:.0f}M" if x < 1e9 else f"{x / 1e9:.1f}B"


def style_axes(ax, xlabel="Active Parameters"):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=[1, 2, 5]))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_params))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test Loss")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/runs.jsonl")
    p.add_argument("--archs", nargs="+", default=list(STYLE))
    p.add_argument("--shape", nargs="+", default=["16", "8x2"],
                   help="layer shapes to include (default: the 16-layer study)")
    p.add_argument("--out_dir", default="figures")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results, set(args.archs), set(args.shape))
    laws = {}

    for arch in args.archs:
        budgets = {c: pts for c, pts in sorted(runs[arch].items()) if len(pts) >= MIN_POINTS}
        if not budgets:
            print(f"{arch}: no budgets with >= {MIN_POINTS} runs")
            continue
        name, color, _ = STYLE[arch]
        fig, ax = plt.subplots()
        markers = iter("osD^v<>p")
        n_opt, l_opt = [], []
        print(f"\n{name}")
        for c, pts in budgets.items():
            coeffs, log_n_star, l_star = fit_budget(pts)
            xs = np.array([n for n, _ in pts])
            ax.plot(xs, [l for _, l in pts], next(markers), color=color, label=f"{c:.0e}".replace("e+", "e"),
                    markeredgecolor="white", markeredgewidth=0.3)
            grid = np.linspace(np.log(xs.min()), np.log(xs.max()), 100)
            ax.plot(np.exp(grid), np.polyval(coeffs, grid), "-", color=color, alpha=0.4, linewidth=2.5)
            ax.plot(np.exp(log_n_star), l_star, "*", color=color, markersize=12,
                    markeredgecolor="white", markeredgewidth=0.5)
            n_opt.append(np.exp(log_n_star))
            l_opt.append(l_star)
            print(f"  C={c:.0e}  N*={np.exp(log_n_star) / 1e6:.1f}M  L*={l_star:.4f}  ({len(pts)} widths)")
        if len(n_opt) >= 2:
            a, alpha = fit_power_law(n_opt, l_opt)
            laws[arch] = (a, alpha, min(n_opt), max(n_opt))
            print(f"  alpha = {alpha:.3f}")
            n_grid = np.logspace(np.log10(min(n_opt) * 0.7), np.log10(max(n_opt) * 1.5), 100)
            ax.plot(n_grid, a * n_grid ** -alpha, "-", color=color, linewidth=2.5,
                    label=rf"Fitted $L\propto N^{{-{alpha:.3f}}}$")
            # Kaplan et al. exponent, anchored at the geometric mean of the optimal points
            n_mid, l_mid = np.exp(np.mean(np.log(n_opt))), np.exp(np.mean(np.log(l_opt)))
            ax.plot(n_grid, l_mid * (n_grid / n_mid) ** -KAPLAN_ALPHA, "--", color="black",
                    linewidth=1.2, alpha=0.5, label=rf"Kaplan $L\propto N^{{-{KAPLAN_ALPHA}}}$")
        style_axes(ax)
        ax.set_title(f"IsoFLOP Curves: {name}")
        ax.legend(fontsize=8)
        fig.savefig(out / f"isoflops_{arch}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if laws:
        fig, ax = plt.subplots()
        for arch, (a, alpha, lo, hi) in laws.items():
            name, color, ls = STYLE[arch]
            n_grid = np.logspace(np.log10(lo), np.log10(hi), 100)
            ax.plot(n_grid, a * n_grid ** -alpha, linestyle=ls, color=color, linewidth=2.0, label=name)
        style_axes(ax)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.2f}"))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_title("Compute-Optimal Scaling Laws (Active Parameters)")
        ax.legend(fontsize=8, title="Model", title_fontsize=8)
        fig.savefig(out / "isoflops_combined.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"\nfigures written to {out}/")


if __name__ == "__main__":
    main()
