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

The run is resumable: images already in captions.jsonl are skipped, so a
Kaggle/Colab session can be restarted with the same command.

Usage:
    python experiments/caption_run.py --model tiny_llava15 --n_images 3
    python experiments/caption_run.py --model llava15_7b --n_images 1000 \
        --out results/caption_runs/llava15_7b_n1000
"""

import argparse
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


def done_ids(path: str):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["image_id"] for line in f if line.strip()}


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
    args = p.parse_args()

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
    captions_path = os.path.join(out_dir, "captions.jsonl")

    torch.manual_seed(args.seed)
    chair = ChairExtractor()
    gt = CocoGroundTruth(data_cfg["instances_file"], data_cfg["captions_file"], chair)
    image_ids = sample_images(gt, args.n_images, args.seed)
    finished = done_ids(captions_path)
    todo = [i for i in image_ids if i not in finished]
    if args.limit is not None:
        todo = todo[:args.limit]
    logger.info(f"{len(image_ids)} images sampled, {len(finished)} done, {len(todo)} to go")

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
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(run_info, f, indent=2)

    with open(captions_path, "a") as log:
        for image_id in tqdm(todo, desc=f"Captioning [{args.model}]"):
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
