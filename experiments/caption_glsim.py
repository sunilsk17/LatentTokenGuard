"""
experiments/caption_glsim.py
GLSim scores (Park & Li, NeurIPS 2025) for the objects of a finished caption run.

Each stored caption is read once more in a single forward pass (as in caption_hidden.py). For
every first mention of an object class, with o the object's first token:

  local[l', l, k]   mean cosine between o's state at layer l' and the image patches at layer l,
                    over the K patches whose logit-lens probability of o at layer l is highest
                    (K = KS[k])
  global[l', l]     cosine between o's state at layer l' and the last prompt token at layer l

GLSim is w * global + (1 - w) * local at one (l, l', K, w); higher means real. Keeping every
layer pair lets the evaluation use the published settings and also choose them on calibration
images, as it does for the other detectors.

The logit lens applies the model's final norm before the unembedding, except at the last layer,
whose state Hugging Face already returns normalized (as for Internal Confidence).

Outputs, in --out/glsim:  <image_id>.npz with token_index, token_id, local, global

Usage:
    python experiments/caption_glsim.py --model llava15_7b_4bit \
        --run results/caption_runs/llava15_7b_4bit_n1000_s42 --images_dir <coco>/val2017 --out <dir>
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from caption_bench.capture import make_capturer, token_at_char

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KS = (4, 8, 16, 32)


def parse_shard(text: str):
    k, n = (int(v) for v in text.split("/"))
    if not 0 <= k < n:
        raise ValueError(f"--shard must be K/N with 0 <= K < N, got {text}")
    return k, n


def lens_probs(capturer, image_states: torch.Tensor, token_ids, chunk: int = 128) -> torch.Tensor:
    """(L+1, P, M): logit-lens probability of each object token at every image patch and layer."""
    norm, head = capturer.final_norm(), capturer.lm_head()
    ids = torch.tensor(token_ids, device=image_states.device)
    last = image_states.shape[0] - 1
    out = torch.empty(image_states.shape[0], image_states.shape[1], len(token_ids),
                      device=image_states.device)
    with torch.no_grad():
        for layer in range(image_states.shape[0]):
            states = image_states[layer]
            for s in range(0, states.shape[0], chunk):
                h = states[s:s + chunk]
                h = h if layer == last else norm(h)
                out[layer, s:s + chunk] = F.softmax(head(h).float(), dim=-1)[:, ids]
    return out


def glsim_features(capturer, obj_states, image_states, prompt_state, token_ids):
    """
    obj_states (M, L+1, d), image_states (L+1, P, d), prompt_state (L+1, d)
    -> local (M, L+1, L+1, len(KS)) indexed [object layer, image layer, K], global (M, L+1, L+1)
    """
    probs = lens_probs(capturer, image_states, token_ids)                  # (L+1, P, M)
    o = F.normalize(obj_states.float(), dim=-1)                             # (M, L+1, d)
    v = F.normalize(image_states.float(), dim=-1)                           # (L+1, P, d)
    g = F.normalize(prompt_state.float(), dim=-1)                           # (L+1, d)
    M, n_layers = o.shape[0], o.shape[1]
    k_max = min(max(KS), v.shape[1])
    local = torch.empty(M, n_layers, n_layers, len(KS), device=o.device)
    for layer in range(n_layers):
        top = probs[layer].topk(k_max, dim=0).indices.T                    # (M, k_max), by probability
        sims = torch.einsum("mld,mkd->mlk", o, v[layer][top])               # (M, L+1, k_max)
        cum = sims.cumsum(-1)
        for j, k in enumerate(KS):
            local[:, :, layer, j] = cum[..., min(k, k_max) - 1] / min(k, k_max)
    glob = torch.einsum("mld,nd->mln", o, g)                                # (M, L+1, L+1)
    return local.cpu().numpy().astype(np.float32), glob.cpu().numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/caption_bench.yaml")
    p.add_argument("--model", required=True, help="model key the run was made with")
    p.add_argument("--run", required=True, help="run folder with captions.jsonl and run.json")
    p.add_argument("--images_dir", default=None)
    p.add_argument("--out", default=None, help="output folder (default: the run folder)")
    p.add_argument("--shard", default="0/1")
    args = p.parse_args()
    shard, n_shards = parse_shard(args.shard)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_cfg = config["models"][args.model]
    with open(os.path.join(args.run, "run.json")) as f:
        run_info = json.load(f)
    if run_info["model"]["model_id"] != model_cfg["id"]:
        sys.exit(f"--model {args.model} is {model_cfg['id']}, but the run used "
                 f"{run_info['model']['model_id']}")
    images_dir = args.images_dir or config["data"]["images_dir"]
    out_dir = os.path.join(args.out or args.run, "glsim")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.run, "captions.jsonl")) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = [r for r in records if any(m["first"] for m in r["mentions"])]
    todo = [r for r in records[shard::n_shards]
            if not os.path.exists(os.path.join(out_dir, f"{r['image_id']}.npz"))]
    logger.info(f"{len(records)} images with objects, {len(todo)} to go in shard {shard}/{n_shards}")

    capturer = make_capturer(model_cfg)
    capturer.load()

    for rec in tqdm(todo, desc=f"GLSim [{args.model}] shard {shard}/{n_shards}"):
        image = Image.open(os.path.join(images_dir, f"{rec['image_id']:012d}.jpg")).convert("RGB")
        states, image_states, ids, starts, prompt_state = capturer.teacher_force(
            image, run_info["prompt"], caption=rec["caption"], with_prompt_state=True)
        pos = [token_at_char(starts, m["start"]) for m in rec["mentions"] if m["first"]]
        obj_ids = [ids[j] for j in pos]
        local, glob = glsim_features(capturer, states[:, pos].transpose(0, 1), image_states,
                                     prompt_state, obj_ids)
        np.savez(os.path.join(out_dir, f"{rec['image_id']}.npz"),
                 token_index=np.array(pos, dtype=np.int32), token_id=np.array(obj_ids, dtype=np.int64),
                 local=local, **{"global": glob})
        del states, image_states
        if capturer.device == "cuda":
            torch.cuda.empty_cache()

    logger.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
