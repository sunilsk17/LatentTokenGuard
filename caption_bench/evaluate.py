"""
caption_bench/evaluate.py
Object-level hallucination detection metrics for a caption run.

Protocol (follows GLSim, NeurIPS 2025, with an explicit held-out split):
  - unit of evaluation = one object mention (first mention per class by default)
  - positive class = hallucinated object
  - images are split into a calibration half and a test half; every choice
    (layers, statistic, sign, threshold) is made on calibration only and all
    reported numbers come from the test half
  - 95% CIs from a bootstrap over test images (mentions of one image move together)
  - hallucinations concentrate late in long captions, so every detector is also
    scored within caption-position deciles, next to a token-position baseline
"""

import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MentionTable:
    image_ids: np.ndarray     # (N,) image of each mention
    labels: np.ndarray        # (N,) 1 = hallucinated
    classes: List[str]
    positions: np.ndarray     # (N,) index of the object's first token in the caption
    nll: np.ndarray
    entropy: np.ndarray
    ic: np.ndarray            # (N, L+1)
    grid_at: np.ndarray       # (N, L+1, L+1, S)
    grid_pre: np.ndarray
    stat_names: List[str]
    all_image_ids: np.ndarray # every captioned image, including ones with no mentions
    n_dropped_nan: int = 0

    def subset(self, mask: np.ndarray) -> "MentionTable":
        return MentionTable(
            self.image_ids[mask], self.labels[mask],
            [c for c, keep in zip(self.classes, mask) if keep],
            self.positions[mask], self.nll[mask], self.entropy[mask], self.ic[mask],
            self.grid_at[mask], self.grid_pre[mask], self.stat_names,
            self.all_image_ids, self.n_dropped_nan,
        )


def load_run(run_dir: str, first_mention_only: bool = True) -> MentionTable:
    with open(os.path.join(run_dir, "run.json")) as f:
        stat_names = json.load(f)["stat_names"]

    rows = {k: [] for k in ("image_ids", "labels", "classes", "positions", "nll",
                            "entropy", "ic", "grid_at", "grid_pre")}
    all_ids = []
    with open(os.path.join(run_dir, "captions.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            all_ids.append(rec["image_id"])
            mentions = rec["mentions"]
            if not mentions:
                continue
            feats = np.load(os.path.join(run_dir, "feats", f"{rec['image_id']}.npz"))
            keep = [i for i, m in enumerate(mentions) if m["first"] or not first_mention_only]
            for i in keep:
                rows["image_ids"].append(rec["image_id"])
                rows["labels"].append(mentions[i]["label"])
                rows["classes"].append(mentions[i]["canonical"])
            rows["positions"].append(feats["token_index"][keep])
            for key in ("nll", "entropy", "ic", "grid_at", "grid_pre"):
                rows[key].append(feats[key][keep])

    stack = lambda key: np.concatenate(rows[key]) if rows[key] else np.empty(0)
    table = MentionTable(
        np.array(rows["image_ids"]), np.array(rows["labels"]), rows["classes"],
        stack("positions"), stack("nll"), stack("entropy"), stack("ic"),
        stack("grid_at"), stack("grid_pre"), stat_names, np.array(sorted(set(all_ids))),
    )
    # A mention whose first token is the final generated token has no "at" state
    finite = np.isfinite(table.grid_at.reshape(len(table.labels), -1)).all(1)
    table = table.subset(finite)
    table.n_dropped_nan = int((~finite).sum())
    return table


def split_images(image_ids: np.ndarray, calib_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(image_ids.tolist())))
    rng.shuffle(ids)
    n_calib = int(round(len(ids) * calib_fraction))
    return set(ids[:n_calib].tolist()), set(ids[n_calib:].tolist())


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def fast_auroc(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """AUROC for every column of `scores` (N, ...) at once, ties averaged."""
    flat = scores.reshape(len(labels), -1)
    ranks = rankdata(flat, axis=0)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    auc = (ranks[pos].sum(0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc.reshape(scores.shape[1:])


def position_edges(positions: np.ndarray, n_bins: int) -> np.ndarray:
    """Inner quantile edges of caption position (computed on calibration mentions)."""
    return np.quantile(positions, np.linspace(0, 1, n_bins + 1)[1:-1])


def stratified_auroc(labels: np.ndarray, scores: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """
    AUROC within each stratum, averaged with stratum size as weight.
    Strata that contain only one class are skipped. Works column-wise like fast_auroc.
    """
    total, weight = 0.0, 0
    for k in np.unique(strata):
        mask = strata == k
        y = labels[mask]
        if y.min() == y.max():
            continue
        total = total + mask.sum() * fast_auroc(y, scores[mask])
        weight += mask.sum()
    return total / weight if weight else np.full(scores.shape[1:], np.nan)


def metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    if len(set(labels.tolist())) < 2:
        return {"auroc": float("nan"), "aupr": float("nan")}
    return {"auroc": float(roc_auc_score(labels, scores)),
            "aupr": float(average_precision_score(labels, scores))}


def bootstrap_ci(labels, scores, image_ids, n_boot: int, seed: int = 0,
                 strata: Optional[np.ndarray] = None):
    """95% CI of AUROC and AUPR (and position-stratified AUROC), resampling images."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(image_ids)
    groups = {i: np.flatnonzero(image_ids == i) for i in uniq}
    aurocs, auprs, strat = [], [], []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.choice(uniq, len(uniq))])
        y = labels[idx]
        if y.min() == y.max():
            continue
        aurocs.append(roc_auc_score(y, scores[idx]))
        auprs.append(average_precision_score(y, scores[idx]))
        if strata is not None:
            strat.append(float(stratified_auroc(y, scores[idx], strata[idx])))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [np.nan, np.nan]
    out = {"auroc_ci": ci(aurocs), "aupr_ci": ci(auprs)}
    if strata is not None:
        out["auroc_pos_ci"] = ci(strat)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Detectors: each is fitted on calibration rows and returns a scoring function
# (higher score = more likely hallucinated)
# ──────────────────────────────────────────────────────────────────────────────

def _derived_stats(grid: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Append peakedness statistics: (max-mean)/std and (top8-mean)/std."""
    get = lambda n: grid[..., names.index(n)]
    std = np.maximum(get("std"), 1e-6)
    peak = (get("max") - get("mean")) / std
    peak8 = (get("top8") - get("mean")) / std
    return np.concatenate([grid, peak[..., None], peak8[..., None]], -1), names + ["peak", "peak_top8"]


def fit_grid_detector(calib: MentionTable, position: str, stats: List[str],
                      diagonal_only: bool = False, strata: Optional[np.ndarray] = None):
    """
    Pick (text layer, image layer, statistic, sign) with the best calibration AUROC.
    With `strata`, the choice uses AUROC within caption-position bins instead.
    """
    grid, names = _derived_stats(getattr(calib, f"grid_{position}"), calib.stat_names)
    cols = [names.index(s) for s in stats]
    if strata is None:
        auc = fast_auroc(calib.labels, grid[..., cols])           # (Lt, Li, len(stats))
    else:
        auc = stratified_auroc(calib.labels, grid[..., cols], strata)
    best_dir = np.maximum(auc, 1 - auc)
    if diagonal_only:
        mask = np.zeros_like(best_dir, dtype=bool)
        d = np.arange(min(best_dir.shape[:2]))
        mask[d, d, :] = True
        best_dir = np.where(mask, best_dir, -1)
    lt, li, s = np.unravel_index(np.argmax(best_dir), best_dir.shape)
    sign = 1.0 if auc[lt, li, s] >= 0.5 else -1.0
    choice = {"state": position, "text_layer": int(lt), "image_layer": int(li),
              "stat": stats[s], "sign": "+" if sign > 0 else "-",
              "calib_auroc": float(best_dir[lt, li, s])}
    col = cols[s]

    def score(table: MentionTable) -> np.ndarray:
        g, _ = _derived_stats(getattr(table, f"grid_{position}"), table.stat_names)
        return sign * g[:, lt, li, col]
    return score, choice


def fit_ic_detector(calib: MentionTable):
    """Internal Confidence: low logit-lens probability → hallucination. Layer from calib."""
    auc = fast_auroc(calib.labels, -calib.ic)
    layer = int(np.argmax(auc))
    return (lambda t: -t.ic[:, layer]), {"layer": layer, "calib_auroc": float(auc[layer])}


def _position_basis(positions: np.ndarray) -> np.ndarray:
    lp = np.log1p(positions.astype(np.float64))
    return np.stack([np.ones_like(lp), lp, lp ** 2, lp ** 3], 1)


def fit_position_correction(calib: MentionTable, base: Callable):
    """
    Subtract the score a real object is expected to get at the same caption position.
    The trend (cubic in log(1 + position)) is fit on real calibration objects only.
    """
    real = calib.labels == 0
    coef = np.linalg.lstsq(_position_basis(calib.positions[real]), base(calib)[real], rcond=None)[0]
    return lambda t: base(t) - _position_basis(t.positions) @ coef


def rank_average(*scores: np.ndarray) -> np.ndarray:
    return np.mean([rankdata(s) / len(s) for s in scores], axis=0)


def build_detectors(calib: MentionTable, position_bins: int = 10
                    ) -> Tuple[Dict[str, Callable], Dict[str, dict]]:
    dets, info = {}, {}
    dets["NLL"] = lambda t: t.nll
    dets["Entropy"] = lambda t: t.entropy
    # Later objects in a caption are more often hallucinated; position alone is a baseline
    dets["Token position"] = lambda t: t.positions.astype(np.float64)
    dets["Internal Confidence (max over layers)"] = lambda t: -t.ic.max(1)
    dets["Internal Confidence (calib layer)"], info["Internal Confidence (calib layer)"] = fit_ic_detector(calib)

    # Original LatentTokenGuard / ContextualLens-style: max cosine, same layer for text and image
    dets["Max-patch cosine, same layer (LTG/ContextualLens)"], info["Max-patch cosine, same layer (LTG/ContextualLens)"] = \
        fit_grid_detector(calib, "at", ["max"], diagonal_only=True)
    dets["Max-patch cosine, layer pair"], info["Max-patch cosine, layer pair"] = \
        fit_grid_detector(calib, "at", ["max"])
    dets["Top-k mean cosine"], info["Top-k mean cosine"] = \
        fit_grid_detector(calib, "at", ["top4", "top8", "top16", "top32"])
    dets["Global cosine"], info["Global cosine"] = fit_grid_detector(calib, "at", ["global"])
    dets["Peakedness"], info["Peakedness"] = fit_grid_detector(calib, "at", ["peak", "peak_top8"])
    dets["Best latent stat (at)"], info["Best latent stat (at)"] = fit_grid_detector(
        calib, "at", ["max", "top4", "top8", "top16", "top32", "global", "peak", "peak_top8"])
    dets["Best latent stat (pre)"], info["Best latent stat (pre)"] = fit_grid_detector(
        calib, "pre", ["max", "top4", "top8", "top16", "top32", "global", "peak", "peak_top8"])

    # Position-controlled: layers and statistic chosen by within-position AUROC, then the
    # expected similarity of a real object at that position is subtracted
    strata = np.digitize(calib.positions, position_edges(calib.positions, position_bins))
    within, info["Best latent stat, position-corrected"] = fit_grid_detector(
        calib, "at", ["max", "top4", "top8", "top16", "top32", "global", "peak", "peak_top8"],
        strata=strata)
    dets["Best latent stat, position-corrected"] = fit_position_correction(calib, within)

    latent = dets["Best latent stat (at)"]
    dets["Fusion: NLL + best latent"] = lambda t: rank_average(t.nll, latent(t))
    ic = dets["Internal Confidence (calib layer)"]
    dets["Fusion: NLL + IC + best latent"] = lambda t: rank_average(t.nll, ic(t), latent(t))
    return dets, info


# ──────────────────────────────────────────────────────────────────────────────
# CHAIR and detect-then-remove
# ──────────────────────────────────────────────────────────────────────────────

def chair_scores(labels: np.ndarray, image_ids: np.ndarray, all_images: np.ndarray,
                 keep: Optional[np.ndarray] = None) -> Dict[str, float]:
    """CHAIR_i over kept mentions; CHAIR_s over all captioned images."""
    keep = np.ones(len(labels), bool) if keep is None else keep
    kept_labels = labels[keep]
    chair_i = float(kept_labels.mean()) if len(kept_labels) else 0.0
    hallu_images = set(image_ids[keep & (labels == 1)].tolist())
    chair_s = len(hallu_images) / max(len(all_images), 1)
    return {"CHAIR_i": chair_i, "CHAIR_s": chair_s}


def detect_then_remove(calib_scores, calib_labels, test: MentionTable, test_scores,
                       test_images: np.ndarray, keep_real: float) -> Dict[str, float]:
    """Remove test mentions above a threshold that keeps `keep_real` of real calib objects."""
    tau = float(np.quantile(calib_scores[calib_labels == 0], keep_real))
    keep = test_scores <= tau
    before = chair_scores(test.labels, test.image_ids, test_images)
    after = chair_scores(test.labels, test.image_ids, test_images, keep)
    real = test.labels == 0
    return {
        "threshold": tau,
        "before": before,
        "after": after,
        "real_kept": float(keep[real].mean()) if real.any() else float("nan"),
        "hallucinated_removed": float((~keep[~real]).mean()) if (~real).any() else float("nan"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_run(table: MentionTable, eval_cfg: dict) -> dict:
    calib_ids, test_ids = split_images(table.all_image_ids, eval_cfg["calib_fraction"],
                                       eval_cfg["split_seed"])
    calib = table.subset(np.array([i in calib_ids for i in table.image_ids]))
    test = table.subset(np.array([i in test_ids for i in table.image_ids]))
    test_images = np.array(sorted(test_ids))
    n_bins = eval_cfg.get("position_bins", 10)

    summary = {
        "n_images": int(len(table.all_image_ids)),
        "n_mentions": int(len(table.labels)),
        "n_hallucinated": int(table.labels.sum()),
        "n_dropped_last_token": table.n_dropped_nan,
        "calib": {"images": len(calib_ids), "mentions": int(len(calib.labels)),
                  "hallucinated": int(calib.labels.sum())},
        "test": {"images": len(test_ids), "mentions": int(len(test.labels)),
                 "hallucinated": int(test.labels.sum())},
    }
    if len(set(calib.labels.tolist())) < 2 or len(set(test.labels.tolist())) < 2:
        summary["error"] = "Need both real and hallucinated objects in calib and test splits"
        return {"summary": summary}

    edges = position_edges(calib.positions, n_bins)
    test_strata = np.digitize(test.positions, edges)
    summary["position_bins"] = {
        "edges": edges.tolist(),
        "test_hallucinated_share": [float(test.labels[test_strata == k].mean())
                                    for k in range(n_bins) if (test_strata == k).any()],
    }

    dets, info = build_detectors(calib, n_bins)
    results = {}
    for name, fn in dets.items():
        s_test = fn(test)
        res = metrics(test.labels, s_test)
        res["auroc_pos"] = float(stratified_auroc(test.labels, s_test, test_strata))
        res.update(bootstrap_ci(test.labels, s_test, test.image_ids, eval_cfg["bootstrap"],
                                strata=test_strata))
        res["selection"] = info.get(name)
        res["remove"] = detect_then_remove(fn(calib), calib.labels, test, s_test,
                                           test_images, eval_cfg["keep_real"])
        results[name] = res

    # Layer map of the plain max-cosine score (fixed sign: low similarity → hallucinated)
    max_col = table.stat_names.index("max")
    layer_map = fast_auroc(calib.labels, -calib.grid_at[..., max_col])

    return {"summary": summary, "detectors": results,
            "calib_layer_map_max_cosine": layer_map.tolist(),
            "chair_test": chair_scores(test.labels, test.image_ids, test_images)}
