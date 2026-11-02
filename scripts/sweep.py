"""Expand a sweep file into scripts/train.py commands, and print or run them.

A sweep file lists groups of runs. Each group names a config and any of: budgets,
widths per budget (or one list for all), fixed steps, learning rates, data seeds, and
extra train.py arguments (`args`); every combination becomes one run.

    # print the commands (e.g. to hand to a job scheduler)
    uv run python scripts/sweep.py configs/16L/isoflop_sweep.yaml
    # run them one after another on this machine
    uv run python scripts/sweep.py configs/16L/isoflop_sweep.yaml --run
    # run only part of a sweep
    uv run python scripts/sweep.py configs/16L/isoflop_sweep.yaml --only looped_moe --run

Extra arguments after `--` are passed through to train.py, e.g. `-- --wandb`.
"""
import argparse
import itertools
import shlex
import subprocess
import sys

import yaml


def expand(sweep: dict):
    for name, group in sweep["groups"].items():
        widths = group["widths"]
        budgets = group.get("budgets", [None])
        for budget in budgets:
            ws = widths[budget] if isinstance(widths, dict) else widths
            for w, lr, seed in itertools.product(ws, group.get("lrs", [None]), group.get("seeds", [None])):
                cmd = ["uv", "run", "python", "scripts/train.py", "--config", group["config"], "--d_model", str(w)]
                if budget is not None:
                    cmd += ["--budget", str(float(budget))]
                if "steps" in group:
                    cmd += ["--steps", str(group["steps"])]
                if lr is not None:
                    cmd += ["--lr", str(lr)]
                if seed is not None:
                    cmd += ["--seed", str(seed)]
                yield name, cmd + [str(a) for a in group.get("args", [])]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep")
    p.add_argument("--only", nargs="+", help="group names to include")
    p.add_argument("--run", action="store_true", help="run sequentially instead of printing")
    args, passthrough = p.parse_known_args()
    passthrough = [a for a in passthrough if a != "--"]

    with open(args.sweep) as f:
        sweep = yaml.safe_load(f)
    defaults = sweep.get("train_args", [])
    for name, cmd in expand(sweep):
        if args.only and name not in args.only:
            continue
        cmd = cmd + defaults + passthrough
        print(shlex.join(cmd), flush=True)
        if args.run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    sys.exit(main())
