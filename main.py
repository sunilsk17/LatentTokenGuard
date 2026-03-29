"""
main.py
LatentTokenGuard — Main CLI Entry Point

Usage:
    # Quick test — 5 samples from POPE
    python main.py --model phi3 --dataset pope --n_samples 5

    # Full POPE adversarial evaluation
    python main.py --model phi3 --dataset pope --split adversarial

    # Combined scoring mode
    python main.py --model phi3 --dataset pope --mode combined

    # COCO Indoor custom benchmark
    python main.py --model phi3 --dataset coco_indoor --n_samples 50

    # MMHal-Bench
    python main.py --model phi3 --dataset mmhal --n_samples 30
"""

import sys
import os
import json
import time
import logging
import argparse

import yaml
from tqdm import tqdm

from models import load_model
from detection.token_extractor import TokenExtractor
from detection.latent_contrast import LatentContrastEngine
from detection.logit_contrast import LogitContrastEngine
from detection.decision import DecisionLayer
from data import load_dataset
from evaluation.metrics import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║          LatentTokenGuard 🛡️  v1.0                       ║
║     Hallucination Detection for Vision-Language Models   ║
╚══════════════════════════════════════════════════════════╝
"""


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="LatentTokenGuard: Post-hoc Hallucination Detector for LVLMs"
    )
    p.add_argument("--model",    type=str, default="phi3", choices=["phi3", "llava"],
                   help="LVLM backbone to use")
    p.add_argument("--dataset",  type=str, default="pope",
                   choices=["pope", "mmhal", "coco_indoor"],
                   help="Evaluation dataset")
    p.add_argument("--split",    type=str, default="adversarial",
                   help="Dataset split (pope: adversarial|popular|random)")
    p.add_argument("--mode",     type=str, default="alignment_only",
                   choices=["alignment_only", "contrast_only", "combined"],
                   help="Detection mode")
    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold (default: from config.yaml)")
    p.add_argument("--n_samples", type=int, default=None,
                   help="Limit number of samples (for quick testing)")
    p.add_argument("--output",   type=str, default=None,
                   help="Output JSON path (default: results/<dataset>_<model>_<mode>.json)")
    p.add_argument("--config",   type=str, default="config.yaml",
                   help="Path to config.yaml")
    p.add_argument("--tune_threshold", action="store_true",
                   help="Auto-tune threshold on the evaluation data (oracle mode)")
    p.add_argument("--verbose",  action="store_true",
                   help="Print per-sample details")
    return p


def run_pipeline(args):
    print(BANNER)

    # ── Load config ─────────────────────────────────────────────────────────
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), args.config
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.threshold is not None:
        config["detection"]["threshold"] = args.threshold

    use_logit = (
        config["detection"]["use_logit_contrast"]
        and args.mode in ("contrast_only", "combined")
    )

    os.makedirs(config["experiments"]["output_dir"], exist_ok=True)
    os.makedirs(config["experiments"]["cache_dir"], exist_ok=True)

    # ── Init model & modules ─────────────────────────────────────────────────
    logger.info(f"🔧 Model:   {args.model}")
    logger.info(f"🗂️  Dataset: {args.dataset} [{args.split}]")
    logger.info(f"⚙️  Mode:    {args.mode}")
    logger.info(f"📊 Samples: {args.n_samples or 'all'}")

    model           = load_model(args.model, config)
    extractor       = TokenExtractor(use_pos_filter=config["detection"]["use_token_filter"])
    contrast_engine = LatentContrastEngine(config)
    logit_engine    = LogitContrastEngine(config)
    decision        = DecisionLayer(threshold=config["detection"]["threshold"])

    # ── Load dataset ─────────────────────────────────────────────────────────
    dataset_kwargs = {}
    if args.dataset == "pope":
        dataset_kwargs["split"] = args.split

    dataset = load_dataset(args.dataset, config, **dataset_kwargs)
    model.load()

    evaluator       = Evaluator()
    predictions_log = []
    all_scores      = []
    all_gt          = []

    # ── Iterate samples ──────────────────────────────────────────────────────
    iter_kwargs = {"n_samples": args.n_samples}
    if args.dataset == "pope":
        iter_kwargs["split"] = args.split

    sample_iter = dataset.iter_samples(**iter_kwargs)

    logger.info("\n🚀 Starting inference pipeline...\n")

    for sample in tqdm(sample_iter, desc="Processing"):
        t_start = time.time()
        final_score = 0.5
        label = "Faithful"
        answer = ""

        try:
            # Step 1: LVLM inference
            output = model.run(sample["image"], sample["question"])
            answer = output.answer

            # Step 2: Token extraction
            tok_indices, tok_words = extractor.extract(output.answer, output.tokens)

            # Step 3: Latent alignment
            hall_align, align_score, per_tok = contrast_engine.compute_alignment(
                output.text_embeddings,
                output.visual_embeddings,
                tok_indices,
            )

            # Step 4: Logit contrast (optional)
            contrast_score = None
            if use_logit:
                contrast_score, _ = logit_engine.compute_contrast(
                    model, sample["image"], sample["question"],
                    output.token_ids, tok_indices, output.logits,
                )

            # Step 5: Score combination
            if args.mode == "alignment_only":
                final_score = hall_align
            elif args.mode == "contrast_only" and contrast_score is not None:
                final_score = 1.0 - contrast_score
            else:
                final_score = logit_engine.combine_scores(align_score, contrast_score)

            # Step 6: Decision
            label, _ = decision.classify(final_score)

        except Exception as e:
            logger.warning(f"Sample error: {e}")

        latency_ms = (time.time() - t_start) * 1000
        gt_is_hall = sample.get("is_hallucination", False)

        evaluator.add(label, gt_is_hall, final_score, latency_ms)
        all_scores.append(final_score)
        all_gt.append(int(gt_is_hall))

        log_entry = {
            "question_id": sample.get("question_id", -1),
            "question":    sample.get("question", ""),
            "gt_label":    sample.get("label", ""),
            "model_answer": answer,
            "hallucination_score": round(final_score, 4),
            "pred_label":  label,
            "latency_ms":  round(latency_ms, 1),
        }
        predictions_log.append(log_entry)

        if args.verbose:
            correct = (label == "Hallucinated") == gt_is_hall
            icon = "✅" if correct else "❌"
            print(
                f" {icon} Q: {sample.get('question','')[:60]}\n"
                f"    Answer: '{answer[:50]}' | Score: {final_score:.3f} | {label}"
            )

    # ── Threshold tuning ─────────────────────────────────────────────────────
    if args.tune_threshold:
        best_t, best_f1 = decision.tune_threshold(all_scores, all_gt, metric="f1")
        logger.info(f"Auto-tuned threshold: {best_t:.2f} (F1={best_f1:.4f})")
        # Re-evaluate with tuned threshold
        evaluator.reset()
        for i, (log, score, gt) in enumerate(
            zip(predictions_log, all_scores, all_gt)
        ):
            lbl, _ = decision.classify(score)
            evaluator.add(lbl, bool(gt), score, predictions_log[i]["latency_ms"])

    # ── Compute & print results ───────────────────────────────────────────────
    result = evaluator.compute(
        threshold=decision.get_threshold(),
        dataset=f"{args.dataset}/{args.split}",
        model=args.model,
        mode=args.mode,
    )
    print(result)

    # ── Save output ───────────────────────────────────────────────────────────
    out_path = args.output or os.path.join(
        config["experiments"]["output_dir"],
        f"{args.dataset}_{args.model}_{args.mode}.json",
    )
    with open(out_path, "w") as f:
        json.dump({"metadata": result.to_dict(), "predictions": predictions_log}, f, indent=2)
    logger.info(f"💾 Results saved to: {out_path}")

    model.unload()
    return result


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
