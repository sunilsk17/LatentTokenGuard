"""
experiments/caption_run.py
Caption COCO images and record per-object features.

For each sampled image:
  1. one captured generation (caption + hidden states + logits)
  2. CHAIR object extraction from the caption, labelled real/hallucinated
  3. per-object features (see caption_bench/features.py)

Outputs, in --out:
  run.json            settings, model/library versions, git commit
  captions.jsonl      one line per image: caption, mentions with labels
  feats/<id>.npz      feature arrays for that image's mentions

The run is resumable: images already captioned in --out are skipped, so a
Kaggle/Colab session can be restarted with the same command.

With several GPUs, run one process per GPU with --shard K/N. Shard K captions
every N-th image into captions.shardKofN.jsonl; --merge then folds the shard
files into captions.jsonl.

Usage:
    python experiments/caption_run.py --model tiny_llava15 --n_images 3
    python experiments/caption_run.py --model llava15_7b --n_images 1000 \
        --out results/caption_runs/llava15_7b_n1000
    CUDA_VISIBLE_DEVICES=1 python experiments/caption_run.py ... --shard 1/2
    python experiments/caption_run.py --merge --out <run dir>
"""

import argparse
import glob
import json
import logging
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from caption_bench.capture import LlavaCapturer
from caption_bench.chair import ChairExtractor, CocoGroundTruth
from caption_bench.features import STAT_NAMES, mention_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sample_images(gt: CocoGroundTruth, n: int, seed: int):
    ids = gt.image_ids()
    rng = random.Random(seed)
    return ids if n >= len(ids) else sorted(rng.sample(ids, n))


def caption_files(out_dir: str):
    return sorted(glob.glob(os.path.join(out_dir, "captions*.jsonl")))


def done_ids(out_dir: str):
    """Images captioned in any captions file of the run (merged or per shard)."""
    ids = set()
    for path in caption_files(out_dir):
        with open(path) as f:
            ids |= {json.loads(line)["image_id"] for line in f if line.strip()}
    return ids


def parse_shard(text: str):
    k, n = (int(v) for v in text.split("/"))
    if not 0 <= k < n:
        raise ValueError(f"--shard must be K/N with 0 <= K < N, got {text}")
    return k, n


def merge_shards(out_dir: str):
    """Append shard captions to captions.jsonl (skipping duplicates) and delete the shard files."""
    merged = os.path.join(out_dir, "captions.jsonl")
    seen = set()
    if os.path.exists(merged):
        with open(merged) as f:
            seen = {json.loads(line)["image_id"] for line in f if line.strip()}
    shards = [p for p in caption_files(out_dir) if p != merged]
    added = 0
    with open(merged, "a") as out:
        for path in shards:
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    image_id = json.loads(line)["image_id"]
                    if image_id not in seen:
                        out.write(line if line.endswith("\n") else line + "\n")
                        seen.add(image_id)
                        added += 1
    for path in shards:
        os.remove(path)
    logger.info(f"Merged {len(shards)} shard files: {added} images added, {len(seen)} in captions.jsonl")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/caption_bench.yaml")
    p.add_argument("--model", default="tiny_llava15")
    p.add_argument("--n_images", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--images_dir", default=None, help="override data.images_dir")
    p.add_argument("--annotations_dir", default=None,
                   help="folder with instances_val2017.json and captions_val2017.json")
    p.add_argument("--limit", type=int, default=None,
                   help="caption at most this many new images, then stop (for timed sessions)")
    p.add_argument("--shard", default="0/1",
                   help="K/N: caption only every N-th image starting at K (one process per GPU)")
    p.add_argument("--merge", action="store_true",
                   help="merge shard caption files in --out into captions.jsonl and exit")
    args = p.parse_args()

    if args.merge:
        if not args.out:
            p.error("--merge needs --out")
        merge_shards(args.out)
        return
    shard, n_shards = parse_shard(args.shard)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_cfg = config["models"][args.model]
    data_cfg = dict(config["data"])
    if args.images_dir:
        data_cfg["images_dir"] = args.images_dir
    if args.annotations_dir:
        data_cfg["instances_file"] = os.path.join(args.annotations_dir, "instances_val2017.json")
        data_cfg["captions_file"] = os.path.join(args.annotations_dir, "captions_val2017.json")

    out_dir = args.out or os.path.join("results", "caption_runs", f"{args.model}_n{args.n_images}_s{args.seed}")
    os.makedirs(os.path.join(out_dir, "feats"), exist_ok=True)
    name = "captions.jsonl" if n_shards == 1 else f"captions.shard{shard}of{n_shards}.jsonl"
    captions_path = os.path.join(out_dir, name)

    torch.manual_seed(args.seed)
    chair = ChairExtractor()
    gt = CocoGroundTruth(data_cfg["instances_file"], data_cfg["captions_file"], chair)
    image_ids = sample_images(gt, args.n_images, args.seed)
    finished = done_ids(out_dir)
    todo = [i for i in image_ids[shard::n_shards] if i not in finished]
    if args.limit is not None:
        todo = todo[:args.limit]
    logger.info(f"{len(image_ids)} images sampled, {len(finished)} done, "
                f"{len(todo)} to go in shard {shard}/{n_shards}")

    capturer = LlavaCapturer(model_cfg)
    capturer.load()

    run_info = {
        "model_key": args.model,
        "model": capturer.describe(),
        "prompt": config["prompt"],
        "prompt_template": model_cfg["prompt_template"],
        "max_new_tokens": model_cfg.get("max_new_tokens"),
        "n_images": len(image_ids),
        "seed": args.seed,
        "stat_names": STAT_NAMES,
        "git_commit": git_commit(),
        "data": data_cfg,
    }
    # Shards write identical run.json files; replace atomically so readers never see half a file
    tmp = os.path.join(out_dir, f"run.json.tmp{shard}")
    with open(tmp, "w") as f:
        json.dump(run_info, f, indent=2)
    os.replace(tmp, os.path.join(out_dir, "run.json"))

    desc = f"Captioning [{args.model}]" + (f" shard {shard}/{n_shards}" if n_shards > 1 else "")
    with open(captions_path, "a") as log:
        for image_id in tqdm(todo, desc=desc):
            path = os.path.join(data_cfg["images_dir"], gt.file_names[image_id])
            image = Image.open(path).convert("RGB")

            cap = capturer.generate(image, config["prompt"])
            mentions = gt.label(image_id, chair.extract(cap.caption))

            t0 = time.time()
            if mentions:
                feats = mention_features(capturer, cap, mentions)
                np.savez_compressed(os.path.join(out_dir, "feats", f"{image_id}.npz"), **feats)
            feat_seconds = time.time() - t0

            log.write(json.dumps({
                "image_id": image_id,
                "caption": cap.caption,
                "n_tokens": len(cap.token_ids),
                "gen_seconds": round(cap.gen_seconds, 3),
                "feature_seconds": round(feat_seconds, 3),
                "mentions": [m.to_dict() for m in mentions],
            }) + "\n")
            log.flush()

            del cap
            if capturer.device == "cuda":
                torch.cuda.empty_cache()

    logger.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
