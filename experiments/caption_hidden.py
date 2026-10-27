"""
experiments/caption_hidden.py
Hidden states of the object tokens of a finished caption run, for the
trained linear-probe baseline (caption_bench/evaluate.py).

Each captioned image is read once more with its stored caption in a single
forward pass (no generation). For every first mention of an object class it
saves the state at the object's first token for every layer, which is the
"at" state the latent statistics use.

Outputs, in --out/hidden:
  <image_id>.npz   token_index, token_id      position and id of the object token
                   at        (M, L+1, d) fp16 object-token states, one row per first mention
                   check_max (M, L+1)         max-patch cosine with the image at the same
                                              layer, recomputed from these states; the
                                              evaluation compares it with feats/grid_at
                                              to confirm the states line up
  hidden.json      model and library versions

Resumable, and --shard K/N works like in caption_run.py (one process per GPU).

Usage:
    python experiments/caption_hidden.py --model llava15_7b_4bit \
        --run results/caption_runs/llava15_7b_4bit_n1000_s42 \
        --images_dir <coco>/val2017 --out <dir>
    # then unzip <dir>/hidden into the run folder
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from caption_bench.capture import make_capturer, token_at_char

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_shard(text: str):
    k, n = (int(v) for v in text.split("/"))
    if not 0 <= k < n:
        raise ValueError(f"--shard must be K/N with 0 <= K < N, got {text}")
    return k, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/caption_bench.yaml")
    p.add_argument("--model", required=True, help="model key the run was made with")
    p.add_argument("--run", required=True, help="run folder with captions.jsonl and run.json")
    p.add_argument("--images_dir", default=None, help="override data.images_dir")
    p.add_argument("--out", default=None, help="output folder (default: the run folder)")
    p.add_argument("--shard", default="0/1")
    args = p.parse_args()
    shard, n_shards = parse_shard(args.shard)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_cfg = config["models"][args.model]
    with open(os.path.join(args.run, "run.json")) as f:
        run_info = json.load(f)
    if run_info["model"]["model_id"] != model_cfg["id"]:
        sys.exit(f"--model {args.model} is {model_cfg['id']}, but the run used "
                 f"{run_info['model']['model_id']}")
    images_dir = args.images_dir or config["data"]["images_dir"]
    out_dir = os.path.join(args.out or args.run, "hidden")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.run, "captions.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = [r for r in records if any(m["first"] for m in r["mentions"])]
    todo = [r for r in records[shard::n_shards]
            if not os.path.exists(os.path.join(out_dir, f"{r['image_id']}.npz"))]
    logger.info(f"{len(records)} images with objects, {len(todo)} to go in shard {shard}/{n_shards}")

    capturer = make_capturer(model_cfg)
    capturer.load()
    if shard == 0:
        with open(os.path.join(out_dir, "hidden.json"), "w") as f:
            json.dump({"run": run_info["model"], "model": capturer.describe()}, f, indent=2)

    for rec in tqdm(todo, desc=f"Hidden states [{args.model}] shard {shard}/{n_shards}"):
        path = os.path.join(images_dir, f"{rec['image_id']:012d}.jpg")
        image = Image.open(path).convert("RGB")
        states, image_states, ids, starts = capturer.teacher_force(
            image, run_info["prompt"], caption=rec["caption"])

        firsts = [m for m in rec["mentions"] if m["first"]]
        pos = [token_at_char(starts, m["start"]) for m in firsts]
        at = states[:, pos].transpose(0, 1)                                       # (M, L+1, d)
        t = F.normalize(at.float(), dim=-1)
        v = F.normalize(image_states.float(), dim=-1)
        check = torch.einsum("mld,lpd->mlp", t, v).max(-1).values                 # (M, L+1)

        np.savez(os.path.join(out_dir, f"{rec['image_id']}.npz"),
                 token_index=np.array(pos, dtype=np.int32),
                 token_id=np.array([ids[j] for j in pos], dtype=np.int64),
                 at=at.to(torch.float16).cpu().numpy(),
                 check_max=check.cpu().numpy().astype(np.float32))
        del states, image_states
        if capturer.device == "cuda":
            torch.cuda.empty_cache()

    logger.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
