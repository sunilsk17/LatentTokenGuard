"""
experiments/ablation.py
Experiment 3: Ablation Study

Compares three detection modes on POPE-adversarial:
  1. Alignment only   (Module 3 only)
  2. Contrast only    (Module 4 only)
  3. Combined         (Module 3 + 4 weighted)

Also sweeps detection thresholds to show robustness.

Usage:
    python experiments/ablation.py --model phi3 --n_samples 100
"""

import sys
import os
import json
import time
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from models import load_model
from detection.token_extractor import TokenExtractor
from detection.latent_contrast import LatentContrastEngine
from detection.logit_contrast import LogitContrastEngine
from detection.decision import DecisionLayer
from data.pope_loader import POPELoader
from evaluation.metrics import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODES = ["alignment_only", "combined"]
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]


def run_ablation(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(config["experiments"]["output_dir"], exist_ok=True)

    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model, config)
    model.load()

    extractor       = TokenExtractor(use_pos_filter=config["detection"]["use_token_filter"])
    contrast_engine = LatentContrastEngine(config)
    logit_engine    = LogitContrastEngine(config)
    loader          = POPELoader(config, split="adversarial")

    samples = list(loader.iter_samples(split="adversarial", n_samples=args.n_samples))
    logger.info(f"Loaded {len(samples)} POPE-adversarial samples for ablation.")

    # ── Collect raw scores for all samples ────────────────────────────────────
    align_scores   = []
    contrast_scores = []
    gt_labels      = []
    latencies      = []

    for sample in tqdm(samples, desc="Collecting scores"):
        t_start = time.time()
        try:
            output = model.run(sample["image"], sample["question"])
            tok_indices, _ = extractor.extract(output.answer, output.tokens)
            _, align, _ = contrast_engine.compute_alignment(
                output.text_embeddings, output.visual_embeddings, tok_indices
            )
            # Enable logit contrast for ablation
            config_copy = dict(config)
            config_copy["detection"] = dict(config["detection"])
            config_copy["detection"]["use_logit_contrast"] = True
            cont, _ = logit_engine.compute_contrast(
                model, sample["image"], sample["question"],
                output.token_ids, tok_indices, output.logits,
            )
        except Exception as e:
            logger.warning(f"Error: {e}")
            align, cont = 0.5, 0.5

        align_scores.append(align)
        contrast_scores.append(cont if cont else 0.5)
        gt_labels.append(int(sample["is_hallucination"]))
        latencies.append((time.time() - t_start) * 1000)

    model.unload()

    # ── Ablation table: different modes × thresholds ───────────────────────────
    rows = []
    for mode in MODES:
        for thresh in THRESHOLDS:
            decision = DecisionLayer(threshold=thresh)
            evaluator = Evaluator()

            for i, (a, c, gt, lat) in enumerate(
                zip(align_scores, contrast_scores, gt_labels, latencies)
            ):
                if mode == "alignment_only":
                    score = 1.0 - a
                elif mode == "contrast_only":
                    score = 1.0 - c
                else:  # combined
                    score = logit_engine.combine_scores(a, c)

                lbl, _ = decision.classify(score)
                evaluator.add(lbl, bool(gt), score, lat)

            res = evaluator.compute(threshold=thresh, mode=mode, model=args.model, dataset="POPE-adversarial")
            rows.append({
                "Mode": mode,
                "Threshold": thresh,
                "Accuracy": f"{res.accuracy:.4f}",
                "Precision": f"{res.precision:.4f}",
                "Recall": f"{res.recall:.4f}",
                "F1": f"{res.f1:.4f}",
                "AUROC": f"{res.auroc:.4f}",
                "Latency(ms)": f"{res.mean_latency_ms:.1f}",
            })

    df = pd.DataFrame(rows)
    print("\n" + "="*70)
    print("  ABLATION STUDY — POPE-Adversarial")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)

    # Save results
    out = os.path.join(config["experiments"]["output_dir"], f"ablation_{args.model}.json")
    df.to_json(out, orient="records", indent=2)
    logger.info(f"Ablation results saved to: {out}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str, default="phi3")
    parser.add_argument("--config",    type=str, default="config.yaml")
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
