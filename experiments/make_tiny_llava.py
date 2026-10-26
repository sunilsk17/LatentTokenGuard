"""
experiments/make_tiny_llava.py
Builds a random-weight LLaVA-1.5 with tiny layers for local smoke tests.

It keeps LLaVA-1.5's real tokenizer, image processor, image-token id and
576-patch layout, so it runs the same code path as the 7B model, but
it is 17 MB and runs on a laptop CPU in seconds. Its captions are noise.

Usage:
    python experiments/make_tiny_llava.py            # → ./cache/tiny-llava15
    python experiments/caption_run.py --model tiny_llava15 --n_images 3
"""

import argparse

import torch
from transformers import (AutoProcessor, CLIPVisionConfig, LlamaConfig, LlavaConfig,
                          LlavaForConditionalGeneration)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./cache/tiny-llava15")
    args = p.parse_args()

    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", use_fast=False)
    vision = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                              num_attention_heads=2, image_size=336, patch_size=14,
                              projection_dim=32)
    text = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                       num_attention_heads=4, num_key_value_heads=4, vocab_size=32064,
                       max_position_embeddings=4096)
    config = LlavaConfig(vision_config=vision, text_config=text, image_token_index=32000,
                         vision_feature_layer=-2, vision_feature_select_strategy="default",
                         pad_token_id=32001)

    torch.manual_seed(0)
    model = LlavaForConditionalGeneration(config)
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)
    print(f"Saved tiny LLaVA-1.5 to {args.out}")


if __name__ == "__main__":
    main()
