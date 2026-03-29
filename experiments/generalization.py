"""
experiments/generalization.py
Experiment 4: Cross-model generalization test.

Tests LatentTokenGuard on two different LVLMs to show model-agnostic capability.

Usage:
    python experiments/generalization.py --n_samples 100
"""

import sys, os, json, time, logging, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
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


def run_model(model_name, config, samples, mode="alignment_only"):
    model = load_model(model_name, config)
    model.load()
    extractor = TokenExtractor(use_pos_filter=config["detection"]["use_token_filter"])
    contrast_engine = LatentContrastEngine(config)
    logit_engine = LogitContrastEngine(config)
    decision = DecisionLayer(threshold=config["detection"]["threshold"])
    evaluator = Evaluator()

    for sample in tqdm(samples, desc=model_name):
        t0 = time.time()
        try:
            output = model.run(sample["image"], sample["question"])
            tok_idx, _ = extractor.extract(output.answer, output.tokens)
            hall, align, _ = contrast_engine.compute_alignment(
                output.text_embeddings, output.visual_embeddings, tok_idx
            )
            score = hall
            label, _ = decision.classify(score)
        except Exception as e:
            logger.warning(f"Error: {e}"); score = 0.5; label = "Faithful"

        latency = (time.time() - t0) * 1000
        evaluator.add(label, sample["is_hallucination"], score, latency)

    model.unload()
    return evaluator.compute(
        threshold=config["detection"]["threshold"],
        dataset="POPE-adversarial",
        model=model_name,
        mode=mode,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    type=str, default="config.yaml")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--split",     type=str, default="adversarial")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(config["experiments"]["output_dir"], exist_ok=True)
    loader = POPELoader(config)
    samples = list(loader.iter_samples(split=args.split, n_samples=args.n_samples))

    results = {}
    for model_name in ["phi3", "llava"]:
        try:
            logger.info(f"\n{'='*50}\nGeneralization test: {model_name}\n{'='*50}")
            res = run_model(model_name, config, samples)
            results[model_name] = res.to_dict()
            print(res)
        except Exception as e:
            logger.error(f"Model {model_name} failed: {e}")
            results[model_name] = {"error": str(e)}

    out = os.path.join(config["experiments"]["output_dir"], "generalization.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Generalization results saved: {out}")

    # Summary comparison table
    print("\n" + "="*55)
    print("  GENERALIZATION SUMMARY (POPE-adversarial)")
    print("="*55)
    print(f"{'Model':<15} {'F1':>8} {'AUROC':>8} {'Latency(ms)':>13}")
    print("-"*55)
    for m, r in results.items():
        if "error" not in r:
            print(f"{m:<15} {r['f1']:>8.4f} {r['auroc']:>8.4f} {r['mean_latency_ms']:>13.1f}")
    print("="*55)


if __name__ == "__main__":
    main()
