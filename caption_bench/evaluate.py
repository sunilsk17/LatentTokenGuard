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
  - a cross-fitted variant (k folds over images) scores every image once as
    test data and compares detectors with a paired bootstrap
  - when object-token hidden states were extracted (experiments/caption_hidden.py),
    a trained linear probe on them is the supervised baseline
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
    hidden: Optional[np.ndarray] = None  # (N, L+1, d) object-token states, if extracted
    hidden_check: Optional[dict] = None  # how well the extracted states reproduce grid_at
    glsim_local: Optional[np.ndarray] = None   # (N, L+1, L+1, K) from experiments/caption_glsim.py
    glsim_global: Optional[np.ndarray] = None  # (N, L+1, L+1)
    model_id: str = ""

    def subset(self, mask: np.ndarray) -> "MentionTable":
        pick = lambda a: None if a is None else a[mask]
        return MentionTable(
            self.image_ids[mask], self.labels[mask],
            [c for c, keep in zip(self.classes, mask) if keep],
            self.positions[mask], self.nll[mask], self.entropy[mask], self.ic[mask],
            self.grid_at[mask], self.grid_pre[mask], self.stat_names,
            self.all_image_ids, self.n_dropped_nan,
            pick(self.hidden), self.hidden_check,
            pick(self.glsim_local), pick(self.glsim_global), self.model_id,
        )


def load_run(run_dir: str, first_mention_only: bool = True, hidden: bool = True) -> MentionTable:
    with open(os.path.join(run_dir, "run.json")) as f:
        run_info = json.load(f)
    stat_names = run_info["stat_names"]

    # Object-token hidden states (experiments/caption_hidden.py) hold first mentions only
    hidden_dir = os.path.join(run_dir, "hidden")
    with_hidden = hidden and first_mention_only and os.path.isdir(hidden_dir)
    # GLSim scores (experiments/caption_glsim.py), also first mentions only
    glsim_dir = os.path.join(run_dir, "glsim")
    with_glsim = first_mention_only and os.path.isdir(glsim_dir)
    max_col = stat_names.index("max")

    rows = {k: [] for k in ("image_ids", "labels", "classes", "positions", "nll", "entropy", "ic",
                            "grid_at", "grid_pre", "hidden", "check", "diag", "glocal", "gglobal")}
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
            if with_hidden:
                path = os.path.join(hidden_dir, f"{rec['image_id']}.npz")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} is missing; finish caption_hidden.py for this run")
                hid = np.load(path)
                if not np.array_equal(hid["token_index"], feats["token_index"][keep]):
                    raise ValueError(f"Object token positions in {path} differ from feats/")
                rows["hidden"].append(hid["at"])
                rows["check"].append(hid["check_max"])
                grid = feats["grid_at"][keep][..., max_col]
                d = np.arange(grid.shape[1])
                rows["diag"].append(grid[:, d, d])
            if with_glsim:
                path = os.path.join(glsim_dir, f"{rec['image_id']}.npz")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} is missing; finish caption_glsim.py for this run")
                gl = np.load(path)
                if not np.array_equal(gl["token_index"], feats["token_index"][keep]):
                    raise ValueError(f"Object token positions in {path} differ from feats/")
                rows["glocal"].append(gl["local"])
                rows["gglobal"].append(gl["global"])

    stack = lambda key: np.concatenate(rows[key]) if rows[key] else np.empty(0)
    table = MentionTable(
        np.array(rows["image_ids"]), np.array(rows["labels"]), rows["classes"],
        stack("positions"), stack("nll"), stack("entropy"), stack("ic"),
        stack("grid_at"), stack("grid_pre"), stat_names, np.array(sorted(set(all_ids))),
        model_id=run_info.get("model", {}).get("model_id", ""),
    )
    if with_glsim:
        table.glsim_local, table.glsim_global = stack("glocal"), stack("gglobal")
    if with_hidden:
        table.hidden = stack("hidden")
        # Same-layer max cosine from the re-read states vs. the one stored at generation
        diff = np.abs(stack("check") - stack("diag"))
        diff = diff[np.isfinite(diff).all(1)]
        table.hidden_check = {"max_abs_diff": float(diff.max()),
                              "median_abs_diff": float(np.median(diff)),
                              "share_above_0.01": float((diff.max(1) > 0.01).mean())}
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


PROBE_C = (1e-3, 1e-2, 1e-1)


def fit_probe_detector(calib: MentionTable, seed: int = 0):
    """
    Trained linear probe on the object-token hidden state (SAPLMA-style, but linear):
    standardized features, L2-regularized logistic regression. The layer and the
    regularization strength are chosen on an inner split of the calibration images
    (half to fit, half to score); the probe is then refit on all calibration rows.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    ids = np.unique(calib.image_ids)
    rng.shuffle(ids)
    inner = np.isin(calib.image_ids, ids[: len(ids) // 2])
    y = calib.labels

    def fit(x, labels, c):
        mu, sd = x.mean(0), x.std(0) + 1e-6
        clf = LogisticRegression(C=c, max_iter=2000).fit((x - mu) / sd, labels)
        return lambda z: clf.decision_function((z - mu) / sd)

    def inner_auroc(layer, c):
        x = calib.hidden[:, layer].astype(np.float32)
        s = fit(x[inner], y[inner], c)(x[~inner])
        return float(roc_auc_score(y[~inner], s))

    by_layer = [inner_auroc(l, PROBE_C[1]) for l in range(calib.hidden.shape[1])]
    layer = int(np.argmax(by_layer))
    by_c = [inner_auroc(layer, c) for c in PROBE_C]
    c = PROBE_C[int(np.argmax(by_c))]
    score = fit(calib.hidden[:, layer].astype(np.float32), y, c)
    return (lambda t: score(t.hidden[:, layer].astype(np.float32))), \
        {"layer": layer, "C": c, "inner_auroc": max(by_c)}


GLSIM_KS = (4, 8, 16, 32)                 # K values stored by experiments/caption_glsim.py
GLSIM_WS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
# Settings published by Park & Li (NeurIPS 2025, Table 5): (image/prompt layer l, object layer l', K, w)
GLSIM_PUBLISHED = {"llava-hf/llava-1.5-7b-hf": (32, 31, 32, 0.6),
                   "llava-hf/llava-1.5-13b-hf": (40, 38, 32, 0.6)}


def _glsim(table: MentionTable, l: int, lo: int, k: int, w: float) -> np.ndarray:
    """GLSim is higher for real objects; negated so that higher means hallucinated."""
    return -(w * table.glsim_global[:, lo, l] + (1 - w) * table.glsim_local[:, lo, l, GLSIM_KS.index(k)])


def fit_glsim_detector(calib: MentionTable):
    """GLSim with (l, l', K, w) chosen by calibration AUROC, like the other layer-selected detectors."""
    best = (-1.0, None)
    for k in GLSIM_KS:
        local = calib.glsim_local[..., GLSIM_KS.index(k)]
        for w in GLSIM_WS:
            auc = fast_auroc(calib.labels, -(w * calib.glsim_global + (1 - w) * local))  # (L+1, L+1)
            lo, l = np.unravel_index(np.argmax(auc), auc.shape)
            if auc[lo, l] > best[0]:
                best = (float(auc[lo, l]), (int(l), int(lo), k, w))
    l, lo, k, w = best[1]
    return (lambda t: _glsim(t, l, lo, k, w)), \
        {"image_layer": l, "object_layer": lo, "K": k, "w": w, "calib_auroc": best[0]}


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

    # GLSim (Park & Li, NeurIPS 2025): needs experiments/caption_glsim.py
    if calib.glsim_local is not None:
        if calib.model_id in GLSIM_PUBLISHED:
            l, lo, k, w = GLSIM_PUBLISHED[calib.model_id]
            dets["GLSim (published settings)"] = lambda t: _glsim(t, l, lo, k, w)
            info["GLSim (published settings)"] = {"image_layer": l, "object_layer": lo, "K": k, "w": w}
        dets["GLSim (settings chosen on calibration)"], info["GLSim (settings chosen on calibration)"] = \
            fit_glsim_detector(calib)

    # Supervised baseline: needs the hidden states from experiments/caption_hidden.py
    if calib.hidden is not None:
        dets["Linear probe on hidden states"], info["Linear probe on hidden states"] = \
            fit_probe_detector(calib)
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
    if table.hidden_check is not None:
        summary["hidden_check"] = table.hidden_check
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


# ──────────────────────────────────────────────────────────────────────────────
# Cross-fitted evaluation
# ──────────────────────────────────────────────────────────────────────────────

# (detector, baseline) pairs compared with a paired bootstrap. The fusion row asks
# whether the latent score adds information to NLL, not whether it replaces it.
PAIRED_COMPARISONS = [
    ("Best latent stat (at)", "NLL"),
    ("Best latent stat (at)", "Internal Confidence (calib layer)"),
    ("Fusion: NLL + best latent", "NLL"),
    # only when hidden states / GLSim scores were extracted
    ("Best latent stat (at)", "Linear probe on hidden states"),
    ("Best latent stat (at)", "GLSim (settings chosen on calibration)"),
    ("Best latent stat (at)", "GLSim (published settings)"),
]


def _percentile_of(reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each score among the reference scores."""
    ref = np.sort(reference)
    return (np.searchsorted(ref, scores, "left") + np.searchsorted(ref, scores, "right")) / (2 * len(ref))


def crossfit_scores(table: MentionTable, folds: int, seed: int, position_bins: int):
    """
    Out-of-fold scores: images are split into `folds` folds; detectors are fit on the
    other folds and score the held-out one, so every mention is scored exactly once.
    Each fold's scores are mapped to percentiles of the fitting folds' scores, which
    keeps folds comparable when they select different layers or statistics.
    Position bins are also set on the fitting folds.
    """
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(table.all_image_ids.tolist())))
    rng.shuffle(ids)
    fold_of = {i: f for f, chunk in enumerate(np.array_split(ids, folds)) for i in chunk.tolist()}
    fold = np.array([fold_of[i] for i in table.image_ids.tolist()])

    scores: Dict[str, np.ndarray] = {}
    selections: Dict[str, list] = {}
    strata = np.zeros(len(table.labels), int)
    for f in range(folds):
        fit, held = table.subset(fold != f), table.subset(fold == f)
        dets, info = build_detectors(fit, position_bins)
        strata[fold == f] = np.digitize(held.positions, position_edges(fit.positions, position_bins))
        for name, fn in dets.items():
            scores.setdefault(name, np.zeros(len(table.labels)))[fold == f] = \
                _percentile_of(fn(fit), fn(held))
            selections.setdefault(name, []).append(info.get(name))
    return scores, strata, selections


def paired_bootstrap(labels, scores_a, scores_b, image_ids, strata, n_boot: int, seed: int = 0):
    """AUROC(a) - AUROC(b), overall and within position bins, resampling images."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(image_ids)
    groups = {i: np.flatnonzero(image_ids == i) for i in uniq}
    both = np.stack([scores_a, scores_b], 1)
    diffs, diffs_pos = [], []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.choice(uniq, len(uniq))])
        y = labels[idx]
        if y.min() == y.max():
            continue
        a, b = fast_auroc(y, both[idx])
        diffs.append(a - b)
        a, b = stratified_auroc(y, both[idx], strata[idx])
        diffs_pos.append(a - b)

    def summarize(point, v):
        v = np.array(v)
        return {"diff": float(point),
                "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                "p_le_0": float(np.mean(v <= 0))}
    point = fast_auroc(labels, both)
    point_pos = stratified_auroc(labels, both, strata)
    return {"overall": summarize(point[0] - point[1], diffs),
            "within_position": summarize(point_pos[0] - point_pos[1], diffs_pos)}


def crossfit_evaluate(table: MentionTable, eval_cfg: dict) -> dict:
    folds = eval_cfg["cv_folds"]
    scores, strata, selections = crossfit_scores(
        table, folds, eval_cfg["split_seed"], eval_cfg.get("position_bins", 10))

    results = {}
    for name, s in scores.items():
        res = metrics(table.labels, s)
        res["auroc_pos"] = float(stratified_auroc(table.labels, s, strata))
        res.update(bootstrap_ci(table.labels, s, table.image_ids, eval_cfg["bootstrap"],
                                strata=strata))
        res["selections"] = selections[name]
        results[name] = res

    comparisons = []
    for a, b in PAIRED_COMPARISONS:
        if a not in scores or b not in scores:
            continue
        comparisons.append({"detector": a, "baseline": b,
                            **paired_bootstrap(table.labels, scores[a], scores[b],
                                               table.image_ids, strata, eval_cfg["bootstrap"])})
    return {"folds": folds, "n_mentions": int(len(table.labels)),
            "n_hallucinated": int(table.labels.sum()),
            "detectors": results, "comparisons": comparisons}
