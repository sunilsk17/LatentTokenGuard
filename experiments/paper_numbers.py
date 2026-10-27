"""
experiments/paper_numbers.py
Every number quoted in the paper, written as LaTeX macros so the text cannot drift from the data.

In the paper:  \\input{numbers}  then  \\N{l7-cv-lat-pos}  ->  79.7

Keys: <model>-<split>-<detector>-<metric>
  model     l7 (LLaVA-1.5-7B), l13 (LLaVA-1.5-13B), q7 (Qwen2.5-VL-7B)
  split     cv (cross-fitted) or ho (held-out test half)
  metric    auc, auclo, auchi, pos, poslo, poshi, aupr; ho also has
            rmib, rmia (CHAIR_i before/after removal), rmsb, rmsa (CHAIR_s), kept, removed
Paired comparisons:  <model>-cmp-<pair>-{d,lo,hi,p} (overall) and -{wd,wlo,whi,wp} (within bins)
Run statistics, POPE and label efficiency are listed in the code below.

Usage:
    python experiments/paper_numbers.py --out <paper>/numbers.tex
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score

MODELS = {"l7": "llava15_7b_4bit_n1000_s42", "l13": "llava15_13b_n1000_s42", "q7": "qwen25vl_7b_n1000_s42"}
LE_KEYS = {"l7": "llava15_7b", "l13": "llava15_13b", "q7": "qwen25vl_7b"}
DETECTORS = {
    "nll": "NLL", "ent": "Entropy", "pos": "Token position",
    "icmax": "Internal Confidence (max over layers)", "ic": "Internal Confidence (calib layer)",
    "maxsame": "Max-patch cosine, same layer (LTG/ContextualLens)", "maxpair": "Max-patch cosine, layer pair",
    "topk": "Top-k mean cosine", "glob": "Global cosine", "peak": "Peakedness",
    "lat": "Best latent stat (at)", "latpre": "Best latent stat (pre)",
    "latpc": "Best latent stat, position-corrected", "fus": "Fusion: NLL + best latent",
    "fus3": "Fusion: NLL + IC + best latent", "probe": "Linear probe on hidden states",
    "glsim": "GLSim (settings chosen on calibration)", "glsimpub": "GLSim (published settings)",
}
PAIRS = {
    ("Best latent stat (at)", "NLL"): "latnll",
    ("Best latent stat (at)", "Internal Confidence (calib layer)"): "latic",
    ("Fusion: NLL + best latent", "NLL"): "fusnll",
    ("Best latent stat (at)", "Linear probe on hidden states"): "latprobe",
    ("Best latent stat (at)", "GLSim (settings chosen on calibration)"): "latglsim",
    ("Best latent stat (at)", "GLSim (published settings)"): "latglsimpub",
}
POPE = {"llava": "results/pope_adversarial_llava_alignment_only.json",
        "phi": "results/pope_adversarial_phi3_alignment_only.json"}


def pct(x):
    return f"{100 * x:.1f}"


def signed(x):
    return f"{100 * x:+.1f}".replace("-", "$-$").replace("+", "$+$")


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def run_numbers(key, run):
    d = f"results/caption_runs/{run}"
    out = {}
    recs = [json.loads(line) for line in open(f"{d}/captions.jsonl")]
    allm = [m for r in recs for m in r["mentions"]]
    e = json.load(open(f"{d}/eval/eval.json"))
    s = e["summary"]
    out.update({
        "images": f"{len(recs):,}", "tokens": f"{np.mean([r['n_tokens'] for r in recs]):.0f}",
        "noobj": str(sum(not r["mentions"] for r in recs)),
        "allobj": f"{len(allm):,}", "allhal": f"{sum(m['label'] for m in allm):,}",
        "chairi": pct(np.mean([m["label"] for m in allm])),
        "chairs": pct(np.mean([any(m["label"] for m in r["mentions"]) for r in recs])),
        "obj": f"{s['n_mentions']:,}", "hal": f"{s['n_hallucinated']:,}",
        "halpct": pct(s["n_hallucinated"] / s["n_mentions"]),
        "calobj": f"{s['calib']['mentions']:,}", "calhal": f"{s['calib']['hallucinated']:,}",
        "testobj": f"{s['test']['mentions']:,}", "testhal": f"{s['test']['hallucinated']:,}",
        "gensec": f"{np.mean([r['gen_seconds'] for r in recs]):.1f}",
    })
    # Share of hallucinated first mentions in the first and last position decile (all images)
    pos, lab = [], []
    for r in recs:
        firsts = [m for m in r["mentions"] if m["first"]]
        if not firsts:
            continue
        f = np.load(f"{d}/feats/{r['image_id']}.npz")
        keep = [i for i, m in enumerate(r["mentions"]) if m["first"]]
        pos += f["token_index"][keep].tolist()
        lab += [m["label"] for m in firsts]
    pos, lab = np.array(pos), np.array(lab)
    bins = np.digitize(pos, np.quantile(pos, np.linspace(0, 1, 11)[1:-1]))
    out["dec1"] = f"{100 * lab[bins == 0].mean():.0f}"
    out["dec10"] = f"{100 * lab[bins == 9].mean():.0f}"

    # AUROC lost from overall to within position bins
    for short in ("lat", "ic", "nll", "glsim", "glsimpub"):
        name = DETECTORS[short]
        if name not in e["crossfit"]["detectors"]:
            continue
        c = e["crossfit"]["detectors"][name]
        out[f"cv-{short}-drop"] = f"{100 * (c['auroc'] - c['auroc_pos']):.1f}"
    for short, name in DETECTORS.items():
        if name not in e["detectors"]:
            continue
        c, h = e["crossfit"]["detectors"][name], e["detectors"][name]
        out.update({
            f"cv-{short}-auc": pct(c["auroc"]), f"cv-{short}-auclo": pct(c["auroc_ci"][0]),
            f"cv-{short}-auchi": pct(c["auroc_ci"][1]), f"cv-{short}-pos": pct(c["auroc_pos"]),
            f"cv-{short}-poslo": pct(c["auroc_pos_ci"][0]), f"cv-{short}-poshi": pct(c["auroc_pos_ci"][1]),
            f"cv-{short}-aupr": pct(c["aupr"]), f"cv-{short}-auprlo": pct(c["aupr_ci"][0]),
            f"cv-{short}-auprhi": pct(c["aupr_ci"][1]),
            f"ho-{short}-auc": pct(h["auroc"]), f"ho-{short}-pos": pct(h["auroc_pos"]),
            f"ho-{short}-aupr": pct(h["aupr"]),
            f"ho-{short}-rmib": pct(h["remove"]["before"]["CHAIR_i"]),
            f"ho-{short}-rmia": pct(h["remove"]["after"]["CHAIR_i"]),
            f"ho-{short}-rmsb": pct(h["remove"]["before"]["CHAIR_s"]),
            f"ho-{short}-rmsa": pct(h["remove"]["after"]["CHAIR_s"]),
            f"ho-{short}-kept": pct(h["remove"]["real_kept"]),
            f"ho-{short}-removed": pct(h["remove"]["hallucinated_removed"]),
        })
    for c in e["crossfit"]["comparisons"]:
        k = PAIRS[(c["detector"], c["baseline"])]
        o, w = c["overall"], c["within_position"]
        out.update({
            f"cmp-{k}-d": signed(o["diff"]), f"cmp-{k}-lo": signed(o["ci"][0]),
            f"cmp-{k}-hi": signed(o["ci"][1]), f"cmp-{k}-p": pval(o["p_le_0"]),
            f"cmp-{k}-wd": signed(w["diff"]), f"cmp-{k}-wlo": signed(w["ci"][0]),
            f"cmp-{k}-whi": signed(w["ci"][1]), f"cmp-{k}-wp": pval(w["p_le_0"]),
            f"cmp-{k}-wabs": f"{100 * abs(w['diff']):.1f}",
        })
    # Layers chosen by the latent statistic across the held-out split and the 5 folds
    sel = e["crossfit"]["detectors"]["Best latent stat (at)"]["selections"] + \
        [e["detectors"]["Best latent stat (at)"]["selection"]]
    t, i = [x["text_layer"] for x in sel], [x["image_layer"] for x in sel]
    stats = sorted({x["stat"] for x in sel}, key=lambda n: int(n[3:]) if n.startswith("top") else 0)
    out["latstats"] = ", ".join(f"$k={n[3:]}$" if n.startswith("top") else n for n in stats)
    out.update({"laytxt": f"{min(t)}--{max(t)}", "layimg": f"{min(i)}--{max(i)}",
                "nlayers": str(json.load(open(f"{d}/run.json"))["model"]["num_layers"])})
    # Choices made on the fitting data in each fold (f1..f5) and on the held-out split (ho)
    def sel_text(name, sel):
        if sel is None:
            return "--"
        if "C" in sel:
            return f"{sel['layer']}, $10^{{{int(round(np.log10(sel['C'])))}}}$"
        if "w" in sel:
            return f"{sel['object_layer']}/{sel['image_layer']}, {sel['K']}, {sel['w']:g}"
        if "layer" in sel:
            return str(sel["layer"])
        stat = sel["stat"].replace("top", "top-")
        return f"{sel['text_layer']}/{sel['image_layer']}, {stat}"
    for short in ("lat", "maxsame", "ic", "probe", "glsim"):
        name = DETECTORS[short]
        if name not in e["detectors"]:
            continue
        for f, sel in enumerate(e["crossfit"]["detectors"][name]["selections"], 1):
            out[f"sel-{short}-f{f}"] = sel_text(name, sel)
        out[f"sel-{short}-ho"] = sel_text(name, e["detectors"][name]["selection"])
    if "hidden_check" in s:
        out["hidmax"] = f"{s['hidden_check']['max_abs_diff']:.3f}"
        out["hidshare"] = f"{100 * s['hidden_check']['share_above_0.01']:.1f}"
        mant, exp = f"{s['hidden_check']['median_abs_diff']:.0e}".split("e")
        out["hidmed"] = f"${mant}\\times10^{{{int(exp)}}}$"
    le = f"results/paper/label_efficiency_{LE_KEYS[key]}.json"
    if os.path.exists(le):
        for n, v in json.load(open(le))["sizes"].items():
            for det in ("latent", "probe"):
                out[f"le-{n}-{det}"] = pct(v[det]["auroc_pos"][0])
    return {f"{key}-{k}": v for k, v in out.items()}


def pope_numbers():
    out = {}
    for key, path in POPE.items():
        preds = json.load(open(path))["predictions"]
        s = np.array([p["hallucination_score"] for p in preds])
        absent = np.array([p["gt_label"] == "no" for p in preds]).astype(int)
        yes = np.array([p["model_answer"].strip().lower().startswith("yes") for p in preds])
        hall = (yes & (absent == 1)).astype(int)
        out.update({
            f"pope-{key}-auc": pct(roc_auc_score(absent, s)),
            f"pope-{key}-answer": pct(roc_auc_score(absent, (~yes).astype(int))),
            f"pope-{key}-nhal": f"{int(hall.sum()):,}", f"pope-{key}-nyes": f"{int(yes.sum()):,}",
            f"pope-{key}-halauc": pct(roc_auc_score(hall[yes], s[yes])),
            # how much the score follows the model's own answer
            f"pope-{key}-ansauc": pct(roc_auc_score((~yes).astype(int), s)),
            f"pope-{key}-noauc": pct(roc_auc_score(absent[~yes], s[~yes])),
            f"pope-{key}-nno": f"{int((~yes).sum()):,}",
        })
        for a_name, a_val in (("present", 0), ("absent", 1)):
            for y_name, y_val in (("yes", True), ("no", False)):
                m = (absent == a_val) & (yes == y_val)
                out[f"pope-{key}-n-{a_name}-{y_name}"] = f"{int(m.sum()):,}"
                out[f"pope-{key}-mean-{a_name}-{y_name}"] = f"{s[m].mean():.3f}"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    nums = pope_numbers()
    for key, run in MODELS.items():
        nums.update(run_numbers(key, run))
    lines = ["% Generated by experiments/paper_numbers.py. Do not edit by hand.",
             "\\makeatletter",
             "\\newcommand{\\N}[1]{\\ifcsname ltg@#1\\endcsname\\csname ltg@#1\\endcsname"
             "\\else\\textbf{??#1}\\fi}"]
    lines += [f"\\expandafter\\def\\csname ltg@{k}\\endcsname{{{v}}}" for k, v in sorted(nums.items())]
    lines.append("\\makeatother")
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {len(nums)} numbers to {args.out}")


if __name__ == "__main__":
    main()
