"""
experiments/score_label_audit.py
Compare the reviewer's answers from the label-audit page with the CHAIR labels.

A CHAIR "hallucinated" label is confirmed when the reviewer answered "no" (the object is not
in the image); a "real" label is confirmed by "yes". "Not an object" counts against both labels.
"Unsure" answers are reported and left out of the rates.

Usage:
    python experiments/score_label_audit.py --key <audit>/key.json \
        --answers ~/Downloads/label_audit_answers.json --out results/paper/label_audit.json
"""

import argparse
import json
import os
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson interval for a proportion."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return [centre - half, centre + half]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", required=True)
    p.add_argument("--answers", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    key = json.load(open(args.key))
    answers = {int(k): v for k, v in json.load(open(args.answers)).items()}

    counts = defaultdict(lambda: defaultdict(int))
    for it in key:
        a = answers.get(it["id"])
        if a is None:
            continue
        label = "hallucinated" if it["chair_label"] else "real"
        confirmed = (a == "no") if it["chair_label"] else (a == "yes")
        for group in (label, f"{it['model']} / {label}"):
            counts[group]["answered"] += 1
            counts[group]["unsure"] += a == "unsure"
            counts[group]["not_object"] += a == "not_object"
            counts[group]["confirmed"] += confirmed

    result = {"n_items": len(key), "n_answered": len(answers), "groups": {}}
    for group, c in sorted(counts.items()):
        n = c["answered"] - c["unsure"]
        result["groups"][group] = {**c, "precision": c["confirmed"] / n if n else float("nan"),
                                   "ci": wilson(c["confirmed"], n)}
        g = result["groups"][group]
        print(f"{group:32s} {c['confirmed']:3d}/{n:3d} confirmed = {100 * g['precision']:.1f}% "
              f"[{100 * g['ci'][0]:.1f}, {100 * g['ci'][1]:.1f}]  unsure {c['unsure']}, not an object {c['not_object']}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
