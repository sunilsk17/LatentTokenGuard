"""
experiments/run_pope.py
Experiment 1: Full POPE benchmark evaluation.

Usage:
    python experiments/run_pope.py --model phi3 --split adversarial
    python experiments/run_pope.py --model phi3 --split all
    python experiments/run_pope.py --model phi3 --split adversarial --n_samples 50
"""

import sys
import os
import json
import time
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from tqdm import tqdm

from models import load_model
from detection.token_extractor import TokenExtractor
from detection.latent_contrast import LatentContrastEngine
from detection.logit_contrast import LogitContrastEngine
from detection.decision import DecisionLayer
from data.pope_loader import POPELoader, VALID_SPLITS
from evaluation.metrics import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pope(args):
    # ── Load config ────────────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["experiments"]["n_samples"] = args.n_samples

    os.makedirs(config["experiments"]["output_dir"], exist_ok=True)

    # ── Init components ────────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model}")
    model = load_model(args.model, config)
    model.load()

    extractor   = TokenExtractor(use_pos_filter=config["detection"]["use_token_filter"])
    contrast_engine = LatentContrastEngine(config)
    logit_engine    = LogitContrastEngine(config)
    decision        = DecisionLayer(threshold=args.threshold if args.threshold is not None else config["detection"]["threshold"])
    loader          = POPELoader(config)

    use_logit = config["detection"]["use_logit_contrast"] and args.mode != "alignment_only"

    # Determine splits to run
    splits = VALID_SPLITS if args.split == "all" else [args.split]
    all_results = {}

    for split in splits:
        logger.info(f"\n{'='*50}\nEvaluating POPE [{split}] | mode={args.mode}\n{'='*50}")
        evaluator = Evaluator()
        predictions_log = []

        samples = list(loader.iter_samples(split=split, n_samples=args.n_samples))
        logger.info(f"Running on {len(samples)} samples...")

        for sample in tqdm(samples, desc=f"POPE-{split}"):
            t_start = time.time()

            try:
                # ── Module 1: LVLM inference ───────────────────────────────────
                output = model.run(sample["image"], sample["question"])

                # ── Module 2: Token extraction ─────────────────────────────────
                tok_indices, tok_words = extractor.extract(output.answer, output.tokens)

                # ── Module 3: Latent contrast ──────────────────────────────────
                hall_align, align_score, per_tok = contrast_engine.compute_alignment(
                    output.text_embeddings,
                    output.visual_embeddings,
                    tok_indices,
                )

                # ── Module 4: Logit contrast (optional) ───────────────────────
                contrast_score = None
                if use_logit and args.mode in ("contrast_only", "combined"):
                    contrast_score, _ = logit_engine.compute_contrast(
                        model, sample["image"], sample["question"],
                        output.token_ids, tok_indices, output.logits,
                    )

                # ── Combine scores ─────────────────────────────────────────────
                if args.mode == "alignment_only":
                    final_score = hall_align
                elif args.mode == "contrast_only" and contrast_score is not None:
                    final_score = 1.0 - contrast_score
                else:  # combined
                    final_score = logit_engine.combine_scores(align_score, contrast_score)

                # ── Module 5: Decision ─────────────────────────────────────────
                label, _ = decision.classify(final_score)

            except Exception as e:
                logger.warning(f"Error on sample {sample['question_id']}: {e}")
                final_score = 0.5
                label = "Faithful"

            latency_ms = (time.time() - t_start) * 1000
            evaluator.add(label, sample["is_hallucination"], final_score, latency_ms)

            predictions_log.append({
                "question_id": sample["question_id"],
                "question": sample["question"],
                "gt_label": sample["label"],
                "model_answer": output.answer if "output" in dir() else "",
                "hallucination_score": round(final_score, 4),
                "pred_label": label,
                "correct": (label == "Hallucinated") == sample["is_hallucination"],
                "latency_ms": round(latency_ms, 1),
            })

        result = evaluator.compute(
            threshold=decision.get_threshold(),
            dataset=f"POPE-{split}",
            model=args.model,
            mode=args.mode,
        )
        print(result)
        all_results[split] = result.to_dict()

        # Save per-split predictions
        out_path = os.path.join(
            config["experiments"]["output_dir"],
            f"pope_{split}_{args.model}_{args.mode}.json",
        )
        with open(out_path, "w") as f:
            json.dump({"metadata": result.to_dict(), "predictions": predictions_log}, f, indent=2)
        logger.info(f"Results saved to: {out_path}")

    model.unload()
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run POPE hallucination evaluation.")
    parser.add_argument("--model",    type=str, default="phi3", choices=["phi3", "llava"])
    parser.add_argument("--split",    type=str, default="adversarial",
                        choices=VALID_SPLITS + ["all"])
    parser.add_argument("--mode",     type=str, default="alignment_only",
                        choices=["alignment_only", "contrast_only", "combined"])
    parser.add_argument("--config",   type=str, default="config.yaml")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Limit samples for quick testing")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override default threshold")
    args = parser.parse_args()
    run_pope(args)


if __name__ == "__main__":
    main()
