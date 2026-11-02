"""Downstream evaluation with AI2 OLMES (Core 9 multiple-choice suite) and the Core 8
average (Core 9 without BoolQ) reported in the paper.

Models must be in Hugging Face format (export with scripts/upload_hf.py). Requires the
`olmes` command, installed as a separate tool (see README) from a fork of
https://github.com/allenai/olmes that supports torch>=2.10 (needed for grouped_mm).

    # evaluate (runs `olmes`), then summarize
    uv run python scripts/eval_olmes.py --models base=exports/<run> looped_moe=<user>/<repo>
    # summarize existing OLMES outputs only
    uv run python scripts/eval_olmes.py --models base=exports/<run> --summarize_only
"""
import argparse
import json
import subprocess
from pathlib import Path

TASKS = ["arc_easy", "arc_challenge", "boolq", "csqa", "hellaswag",
         "openbookqa", "piqa", "socialiqa", "winogrande"]
HEADERS = ["ARC-E", "ARC-C", "BoolQ", "CSQA", "HSwag", "OBQA", "PIQA", "SIQA", "WinoG"]
CORE8 = [t for t in TASKS if t != "boolq"]


def run_olmes(model, out_dir, batch_size):
    cmd = ["olmes", "--model", model,
           "--model-args", json.dumps({"trust_remote_code": True, "max_length": 1024}),
           "--task", "core_9mcqa::olmes", "--batch-size", str(batch_size), "--output-dir", str(out_dir)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def read_scores(out_dir):
    scores = {}
    with open(Path(out_dir) / "metrics-all.jsonl") as f:
        for line in f:
            r = json.loads(line)
            task = r.get("task_name", "").replace("::olmes", "")
            if task in TASKS:
                scores[task] = 100 * r["metrics"]["primary_score"]
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, metavar="NAME=MODEL")
    p.add_argument("--out_dir", default="results/olmes")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--summarize_only", action="store_true")
    args = p.parse_args()

    rows = []
    for item in args.models:
        name, model = item.split("=", 1)
        out = Path(args.out_dir) / name
        if not args.summarize_only:
            run_olmes(model, out, args.batch_size)
        rows.append((name, read_scores(out)))

    print(f"\n{'model':<20}" + "".join(f"{h:>7}" for h in HEADERS) + f"{'Core9':>8}{'Core8':>8}")
    for name, s in rows:
        missing = [t for t in TASKS if t not in s]
        if missing:
            print(f"{name:<20} missing tasks: {missing}")
            continue
        core9 = sum(s[t] for t in TASKS) / len(TASKS)
        core8 = sum(s[t] for t in CORE8) / len(CORE8)
        print(f"{name:<20}" + "".join(f"{s[t]:>7.1f}" for t in TASKS) + f"{core9:>8.2f}{core8:>8.2f}")


if __name__ == "__main__":
    main()
