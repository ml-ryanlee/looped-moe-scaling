"""Export a checkpoint from scripts/train.py as a Hugging Face model, and optionally push it.

The export folder holds model.safetensors, config.json (with auto_map), the GPT-2
tokenizer, and the modeling code (hf_modeling.py, build.py, config.py, models.py), so the
model loads anywhere with:

    AutoModelForCausalLM.from_pretrained(<repo or folder>, trust_remote_code=True)

Usage:
    python scripts/upload_hf.py checkpoints/<run>/final.pt --out exports/<run>
    python scripts/upload_hf.py checkpoints/<run>/final.pt --out exports/<run> --repo <user>/<name>

Pushing requires `huggingface-cli login` or HF_TOKEN.
"""
import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

import looped_scaling
from looped_scaling.config import from_dict
from looped_scaling.hf_modeling import LoopMoEConfig, LoopMoEForCausalLM

CODE_FILES = ["hf_modeling.py", "build.py", "config.py", "models.py"]
AUTO_MAP = {"AutoConfig": "hf_modeling.LoopMoEConfig",
            "AutoModelForCausalLM": "hf_modeling.LoopMoEForCausalLM"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--repo", help="Hub repo id to push to (e.g. user/looped-moe-8x2-d704-1e18)")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = from_dict(ckpt["config"])
    hf_config = LoopMoEConfig.from_model_config(cfg.model, auto_map=AUTO_MAP,
                                                architectures=["LoopMoEForCausalLM"])
    model = LoopMoEForCausalLM(hf_config)
    model.model.load_state_dict(ckpt["model"])

    out = Path(args.out)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(out)
    pkg = Path(looped_scaling.__file__).parent
    for name in CODE_FILES:
        shutil.copy(pkg / name, out / name)
    print(f"exported {cfg.name} -> {out}")

    if args.repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.repo)
        print(f"pushed to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
