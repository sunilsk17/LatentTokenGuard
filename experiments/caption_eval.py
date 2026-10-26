"""
experiments/caption_eval.py
Evaluate object-level hallucination detectors on a caption run.

Writes to <run>/eval/:
  eval.json         all numbers (test split), selections made on calibration
  eval.md           the results table for the paper
  layer_map.png     calibration AUROC of max-patch cosine per (text, image) layer
  roc_test.png      ROC curves of the main detectors on the test split

Usage:
    python experiments/caption_eval.py --run results/caption_runs/llava15_7b_n1000
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import yaml

from caption_bench.evaluate import build_detectors, evaluate_run, load_run, split_images

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROC_DETECTORS = ["NLL", "Token position", "Internal Confidence (calib layer)",
                 "Max-patch cosine, same layer (LTG/ContextualLens)",
                 "Best latent stat (at)", "Best latent stat, position-corrected",
                 "Fusion: NLL + best latent"]


def fmt_ci(value, ci):
    return f"{100 * value:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def write_markdown(result: dict, run_info: dict, path: str):
    s = result["summary"]
    share = s["position_bins"]["test_hallucinated_share"]
    lines = [
        f"# Object hallucination detection: {run_info['model']['model_id']}",
        "",
        f"Images: {s['n_images']} (calib {s['calib']['images']}, test {s['test']['images']}). "
        f"Object mentions: {s['n_mentions']} ({s['n_hallucinated']} hallucinated). "
        f"Test: {s['test']['mentions']} mentions, {s['test']['hallucinated']} hallucinated.",
        "",
        f"CHAIR on test captions: CHAIR_i {100 * result['chair_test']['CHAIR_i']:.1f}, "
        f"CHAIR_s {100 * result['chair_test']['CHAIR_s']:.1f}",
        "",
        f"Hallucinated share by caption-position bin, earliest to latest ({len(share)} bins): "
        + ", ".join(f"{100 * v:.0f}%" for v in share),
        "",
        "| Detector | AUROC [95% CI] | AUROC within position bins [95% CI] | AUPR [95% CI] "
        "| Selected on calib | CHAIR_i after removal | Real objects kept |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in result["detectors"].items():
        sel = r["selection"] or {}
        sel_txt = ", ".join(f"{k}={v}" for k, v in sel.items() if k != "calib_auroc") or "–"
        rm = r["remove"]
        lines.append(
            f"| {name} | {fmt_ci(r['auroc'], r['auroc_ci'])} | {fmt_ci(r['auroc_pos'], r['auroc_pos_ci'])} "
            f"| {fmt_ci(r['aupr'], r['aupr_ci'])} "
            f"| {sel_txt} | {100 * rm['after']['CHAIR_i']:.1f} | {100 * rm['real_kept']:.1f}% |"
        )
    lines += ["", "Positive class = hallucinated object. All selections use the calibration "
              "images only; numbers are on the held-out test images. CIs: bootstrap over images. "
              "Position bins are quantiles of the object's token index on calibration mentions; "
              "the within-bin AUROC is the size-weighted mean over bins."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_layer_map(layer_map: np.ndarray, path: str):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.imshow(layer_map, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(label="Calibration AUROC")
    plt.xlabel("Image-token layer")
    plt.ylabel("Object-token layer")
    plt.title("Max-patch cosine: AUROC by layer pair")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc(table, eval_cfg, path: str):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    calib_ids, test_ids = split_images(table.all_image_ids, eval_cfg["calib_fraction"],
                                       eval_cfg["split_seed"])
    calib = table.subset(np.array([i in calib_ids for i in table.image_ids]))
    test = table.subset(np.array([i in test_ids for i in table.image_ids]))
    dets, _ = build_detectors(calib, eval_cfg.get("position_bins", 10))

    plt.figure(figsize=(6, 5))
    for name in ROC_DETECTORS:
        fpr, tpr, _ = roc_curve(test.labels, dets[name](test))
        plt.plot(fpr, tpr, lw=1.8, label=name)
    plt.plot([0, 1], [0, 1], ":", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Object hallucination detection (test split)")
    plt.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--config", default="configs/caption_bench.yaml")
    p.add_argument("--all_mentions", action="store_true",
                   help="score every mention, not just the first per class")
    args = p.parse_args()

    with open(args.config) as f:
        eval_cfg = yaml.safe_load(f)["evaluation"]
    with open(os.path.join(args.run, "run.json")) as f:
        run_info = json.load(f)

    first_only = eval_cfg["first_mention_only"] and not args.all_mentions
    table = load_run(args.run, first_mention_only=first_only)
    result = evaluate_run(table, eval_cfg)
    result["config"] = {**eval_cfg, "first_mention_only": first_only}

    out = os.path.join(args.run, "eval")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "eval.json"), "w") as f:
        json.dump(result, f, indent=2)

    if "error" in result["summary"]:
        logger.error(result["summary"]["error"])
        print(json.dumps(result["summary"], indent=2))
        return

    write_markdown(result, run_info, os.path.join(out, "eval.md"))
    plot_layer_map(np.array(result["calib_layer_map_max_cosine"]), os.path.join(out, "layer_map.png"))
    plot_roc(table, eval_cfg, os.path.join(out, "roc_test.png"))
    print(open(os.path.join(out, "eval.md")).read())


if __name__ == "__main__":
    main()
