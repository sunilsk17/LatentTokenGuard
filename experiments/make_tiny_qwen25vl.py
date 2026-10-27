"""
experiments/make_tiny_qwen25vl.py
Builds a random-weight Qwen2.5-VL with tiny layers for local smoke tests.

It keeps Qwen2.5-VL-7B's real tokenizer, image processor, special-token ids
and M-RoPE layout, so it runs the same code path as the 7B model on a laptop
CPU. Its captions are noise. Needs transformers >= 4.49.

Usage:
    python experiments/make_tiny_qwen25vl.py          # → ./cache/tiny-qwen25vl
    python experiments/caption_run.py --model tiny_qwen25vl --n_images 3
"""

import argparse

import torch
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="checkpoint (or local folder) whose config and processor are reused")
    p.add_argument("--out", default="./cache/tiny-qwen25vl")
    args = p.parse_args()

    processor = AutoProcessor.from_pretrained(args.source)
    cfg = AutoConfig.from_pretrained(args.source).to_dict()

    text = cfg.get("text_config", cfg)
    text.update(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                num_attention_heads=4, num_key_value_heads=2, max_window_layers=4)
    for d in (text, cfg):
        d.pop("layer_types", None)   # per-layer list sized for 28 layers; rebuilt on load
    # M-RoPE splits each head's rotary half (head_dim / 2 = 8) into t/h/w sections
    for key in ("rope_parameters", "rope_scaling"):
        if isinstance(text.get(key), dict) and "mrope_section" in text[key]:
            text[key]["mrope_section"] = [2, 3, 3]
    if "text_config" in cfg:   # older releases keep text settings at the top level too
        for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
                  "num_attention_heads", "num_key_value_heads", "max_window_layers"):
            if k in cfg:
                cfg[k] = text[k]
        if isinstance(cfg.get("rope_scaling"), dict) and "mrope_section" in cfg["rope_scaling"]:
            cfg["rope_scaling"]["mrope_section"] = [2, 3, 3]
    cfg["vision_config"].update(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                                out_hidden_size=64, fullatt_block_indexes=[1])

    config = AutoConfig.for_model(**cfg)
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(config)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"Saved tiny Qwen2.5-VL to {args.out}")


if __name__ == "__main__":
    main()
