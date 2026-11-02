"""Tokenize the FineWeb-Edu 10B-token sample with the GPT-2 tokenizer.

Documents are shuffled with seed 42; the first 97,135 form the validation split and the
rest the training split. Each document is followed by <|endoftext|>. Output is a flat
uint16 array of token ids per split (~19.7 GB train, ~0.2 GB validation).

    uv run python scripts/prepare_data.py --split val
    uv run python scripts/prepare_data.py --split train
"""
import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

NUM_VAL_DOCS = 97135
SHUFFLE_SEED = 42
OUTPUTS = {"train": "fw_train_tokens.npy", "val": "fw_valid_tokens.npy"}

tokenizer = Tokenizer.from_pretrained("gpt2")
EOS_ID = tokenizer.token_to_id("<|endoftext|>")


def tokenize(example):
    return {"input_ids": tokenizer.encode(example["text"]).ids + [EOS_ID]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--out_dir", default="data")
    p.add_argument("--num_proc", type=int, default=8)
    args = p.parse_args()

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    ds = ds.shuffle(seed=SHUFFLE_SEED)
    ds = ds.select(range(NUM_VAL_DOCS) if args.split == "val" else range(NUM_VAL_DOCS, len(ds)))
    ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=args.num_proc, desc="tokenizing")

    total = sum(len(ids) for ids in tqdm(ds["input_ids"], desc="counting"))
    out = Path(args.out_dir) / OUTPUTS[args.split]
    out.parent.mkdir(parents=True, exist_ok=True)
    tokens = np.lib.format.open_memmap(str(out), mode="w+", dtype=np.uint16, shape=(total,))
    offset = 0
    for ids in tqdm(ds["input_ids"], desc="writing"):
        tokens[offset:offset + len(ids)] = ids
        offset += len(ids)
    tokens.flush()
    print(f"wrote {total:,} tokens to {out}")


if __name__ == "__main__":
    main()
