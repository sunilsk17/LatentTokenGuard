"""
tests/test_capture_consistency.py
Checks that single-pass capture returns the same hidden states and logits as
a separate full forward pass over prompt + caption (the old two-pass method).

Needs the tiny model: python experiments/make_tiny_llava.py
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from caption_bench.capture import LlavaCapturer
from caption_bench.chair import Mention
from caption_bench.features import STAT_NAMES, mention_features

TINY = os.environ.get("TINY_LLAVA_PATH", os.path.join(ROOT, "cache", "tiny-llava15"))
IMAGE = os.path.join(ROOT, "data", "coco", "val2017", "000000000139.jpg")
PROMPT = "Please describe this image in detail."

pytestmark = pytest.mark.skipif(not (os.path.isdir(TINY) and os.path.exists(IMAGE)),
                                reason="tiny model or COCO image not available")


@pytest.fixture(scope="module")
def run():
    with open(os.path.join(ROOT, "configs", "caption_bench.yaml")) as f:
        cfg = yaml.safe_load(f)["models"]["tiny_llava15"]
    cfg = {**cfg, "id": TINY}
    cap = LlavaCapturer(cfg)
    cap.load()
    image = Image.open(IMAGE).convert("RGB")
    out = cap.generate(image, PROMPT)

    inp = cap.processor(text=cfg["prompt_template"].format(prompt=PROMPT), images=image,
                        return_tensors="pt")
    full_ids = torch.cat([inp["input_ids"], torch.tensor([out.token_ids])], 1)
    with torch.no_grad():
        ref = cap.model(input_ids=full_ids, pixel_values=inp["pixel_values"],
                        attention_mask=torch.ones_like(full_ids), output_hidden_states=True)
    return cap, out, inp, ref


def test_image_states_match_full_pass(run):
    cap, out, inp, ref = run
    P = out.image_states.shape[1]
    assert P == 576
    start = (inp["input_ids"][0] == cap.model.config.image_token_index).nonzero()[0].item()
    expected = torch.stack([h[0, start:start + P] for h in ref.hidden_states])
    assert torch.allclose(expected, out.image_states, atol=1e-4)


def test_token_states_and_logits_match_full_pass(run):
    cap, out, inp, ref = run
    # Prefill length, whether <image> was expanded by the processor or the model
    prefix = ref.hidden_states[0].shape[1] - len(out.token_ids)
    for k in range(out.seq_states.shape[0]):
        expected = torch.stack([h[0, prefix - 1 + k] for h in ref.hidden_states])
        assert torch.allclose(expected, out.seq_states[k], atol=1e-4)
    ref_logits = ref.logits[0, prefix - 1: prefix - 1 + len(out.token_ids)]
    assert torch.allclose(ref_logits, out.logits, atol=1e-4)


def test_features_match_hand_computation(run):
    cap, out, _, _ = run
    j = 2
    m = Mention("x", "dog", out.token_starts[j] + 1, out.token_starts[j] + 2)
    f = mention_features(cap, out, [m])
    assert f["token_index"][0] == j

    lt, li = 2, 3
    sims = F.normalize(out.image_states[li], dim=-1) @ F.normalize(out.seq_states[j + 1][lt], dim=-1)
    assert f["grid_at"][0, lt, li, STAT_NAMES.index("max")] == pytest.approx(sims.max().item(), abs=1e-5)
    assert f["grid_at"][0, lt, li, STAT_NAMES.index("top8")] == pytest.approx(
        sims.topk(8).values.mean().item(), abs=1e-5)

    probs = F.softmax(cap.lm_head()(cap.final_norm()(out.image_states[1])), -1)
    assert f["ic"][0, 1] == pytest.approx(probs[:, out.token_ids[j]].max().item(), rel=1e-4)
    nll = -F.log_softmax(out.logits[j], -1)[out.token_ids[j]].item()
    assert f["nll"][0] == pytest.approx(nll, rel=1e-5)
