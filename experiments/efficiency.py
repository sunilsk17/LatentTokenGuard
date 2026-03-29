"""
experiments/efficiency.py
Experiment 5: CPU/GPU Efficiency Profiling

Measures:
  - Time per sample (ms)
  - Peak RAM usage (MB)
  - Throughput (samples/sec)

Usage:
    python experiments/efficiency.py --model phi3 --n_samples 30
"""

import sys, os, json, time, logging, argparse, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_ram_mb() -> float:
    """Current process RAM in MB."""
    import psutil, os
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1e6


def get_peak_vram_mb() -> float:
    """Peak GPU VRAM in MB (if CUDA available)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     type=str, default="phi3")
    parser.add_argument("--config",    type=str, default="config.yaml")
    parser.add_argument("--n_samples", type=int, default=30)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(config["experiments"]["output_dir"], exist_ok=True)

    from models import load_model
    from detection.token_extractor import TokenExtractor
    from detection.latent_contrast import LatentContrastEngine
    from detection.decision import DecisionLayer
    from data.pope_loader import POPELoader

    model = load_model(args.model, config)
    extractor = TokenExtractor(use_pos_filter=False)  # Disable for fair speed test
    contrast_engine = LatentContrastEngine(config)
    decision = DecisionLayer(threshold=0.5)
    loader = POPELoader(config)

    ram_before = get_ram_mb()
    logger.info(f"RAM before model load: {ram_before:.1f} MB")

    model.load()

    ram_after_load = get_ram_mb()
    logger.info(f"RAM after model load:  {ram_after_load:.1f} MB (+{ram_after_load - ram_before:.1f} MB)")

    samples = list(loader.iter_samples(split="adversarial", n_samples=args.n_samples))
    latencies = []
    inference_times = []
    contrast_times = []
    peak_rams = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for sample in tqdm(samples, desc="Efficiency benchmark"):
        t_total = time.time()

        # Inference
        t_inf = time.time()
        output = model.run(sample["image"], sample["question"])
        inference_times.append((time.time() - t_inf) * 1000)

        # Contrast
        t_cont = time.time()
        tok_idx, _ = extractor.extract(output.answer, output.tokens)
        hall, align, _ = contrast_engine.compute_alignment(
            output.text_embeddings, output.visual_embeddings, tok_idx
        )
        contrast_times.append((time.time() - t_cont) * 1000)

        latencies.append((time.time() - t_total) * 1000)
        peak_rams.append(get_ram_mb())

    model.unload()
    gc.collect()

    import numpy as np
    results = {
        "model": args.model,
        "device": model.get_device(),
        "n_samples": len(latencies),
        "mean_total_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "mean_inference_ms": float(np.mean(inference_times)),
        "mean_contrast_ms": float(np.mean(contrast_times)),
        "throughput_samples_per_sec": 1000.0 / float(np.mean(latencies)),
        "peak_ram_mb": float(max(peak_rams)),
        "ram_increase_mb": float(max(peak_rams)) - ram_before,
        "peak_vram_mb": get_peak_vram_mb(),
    }

    print("\n" + "="*55)
    print(f"  EFFICIENCY REPORT — {args.model.upper()}")
    print("="*55)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<30}: {v:.2f}")
        else:
            print(f"  {k:<30}: {v}")
    print("="*55)

    out = os.path.join(config["experiments"]["output_dir"], f"efficiency_{args.model}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Efficiency results saved: {out}")


if __name__ == "__main__":
    main()
