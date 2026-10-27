"""
experiments/label_efficiency.py
How many labelled objects do the detectors need?

Both detectors use labels: the latent statistic to choose its layer pair and statistic, the
linear probe to fit its weights (and choose its layer). For each training size n, n objects are
drawn (class ratio kept) from the fitting folds of the same 5-fold cross-fitting split used by
caption_eval.py, both detectors are fit on them, and the held-out fold is scored. Out-of-fold
scores are pooled after mapping each fold to percentiles of its own training scores. Repeated
draws give a mean and standard deviation.

Needs hidden/ in the run folder (experiments/caption_hidden.py).

Usage:
    python experiments/label_efficiency.py --run results/caption_runs/llava15_7b_4bit_n1000_s42 \
        --out results/paper/label_efficiency_llava15_7b.json
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from caption_bench.evaluate import (_percentile_of, fast_auroc, fit_grid_detector,
                                    fit_probe_detector, load_run, position_edges,
                                    stratified_auroc)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LATENT_STATS = ["max", "top4", "top8", "top16", "top32", "global", "peak", "peak_top8"]
SIZES = [50, 100, 200, 400, 800, 1600]


def draw(labels: np.ndarray, n: int, rng) -> np.ndarray:
    """Indices of n rows with the class ratio of `labels`, at least 5 per class."""
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    n_pos = int(np.clip(round(n * len(pos) / len(labels)), 5, len(pos)))
    n_neg = min(n - n_pos, len(neg))
    return np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--config", default="configs/caption_bench.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["evaluation"]
    folds, n_bins = cfg["cv_folds"], cfg.get("position_bins", 10)
    table = load_run(args.run, first_mention_only=True)
    if table.hidden is None:
        sys.exit(f"{args.run} has no hidden/ folder")

    # Same fold assignment as caption_bench.evaluate.crossfit_scores
    rng = np.random.default_rng(cfg["split_seed"])
    ids = np.array(sorted(set(table.all_image_ids.tolist())))
    rng.shuffle(ids)
    fold_of = {i: f for f, chunk in enumerate(np.array_split(ids, folds)) for i in chunk.tolist()}
    fold = np.array([fold_of[i] for i in table.image_ids.tolist()])

    strata = np.zeros(len(table.labels), int)
    for f in range(folds):
        fit = table.subset(fold != f)
        strata[fold == f] = np.digitize(table.positions[fold == f], position_edges(fit.positions, n_bins))

    sizes = [n for n in SIZES if n < (fold != 0).sum()] + ["all"]
    results = {"run": args.run, "n_objects": int(len(table.labels)), "folds": folds, "sizes": {}}
    for n in sizes:
        # Large draws overlap heavily, so fewer repeats are needed there
        repeats = 1 if n == "all" else args.repeats if n <= 400 else max(2, args.repeats // 2)
        rows = {"latent": [], "probe": []}
        for r in range(repeats):
            draw_rng = np.random.default_rng(1000 + r)
            scores = {k: np.zeros(len(table.labels)) for k in rows}
            for f in range(folds):
                fit_all, held = table.subset(fold != f), table.subset(fold == f)
                for attempt in range(20):
                    idx = np.arange(len(fit_all.labels)) if n == "all" else draw(fit_all.labels, n, draw_rng)
                    mask = np.zeros(len(fit_all.labels), bool)
                    mask[idx] = True
                    train = fit_all.subset(mask)
                    try:
                        latent, _ = fit_grid_detector(train, "at", LATENT_STATS)
                        probe, _ = fit_probe_detector(train, seed=r)
                        break
                    except ValueError:      # an inner split with one class only; draw again
                        continue
                else:
                    raise RuntimeError(f"Could not draw a usable training set of size {n}")
                for k, fn in (("latent", latent), ("probe", probe)):
                    scores[k][fold == f] = _percentile_of(fn(train), fn(held))
            for k in rows:
                rows[k].append({"auroc": float(fast_auroc(table.labels, scores[k])),
                                "auroc_pos": float(stratified_auroc(table.labels, scores[k], strata))})
        n_train = int(np.mean([(fold != f).sum() for f in range(folds)])) if n == "all" else n
        results["sizes"][str(n)] = {
            "n_train": n_train,
            **{k: {m: [float(np.mean([x[m] for x in v])), float(np.std([x[m] for x in v]))]
                   for m in ("auroc", "auroc_pos")} for k, v in rows.items()},
        }
        s = results["sizes"][str(n)]
        logger.info(f"n={n_train}: latent {100 * s['latent']['auroc_pos'][0]:.1f} | "
                    f"probe {100 * s['probe']['auroc_pos'][0]:.1f} (AUROC within position bins)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
