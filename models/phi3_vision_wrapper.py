"""
models/phi3_vision_wrapper.py
Phi-3.5 Vision LVLM wrapper for LatentTokenGuard.

Model: microsoft/Phi-3.5-vision-instruct
  - Lightweight multimodal model (~4GB RAM in float32)
  - Supports CPU inference via attn_implementation='eager'
  - Image tokens are embedded directly in the token sequence

Key design:
  - Visual patch embeddings are extracted from hidden_states at image token positions
  - Text embeddings are extracted at generated token positions
  - Both come from the last (or configurable) transformer layer
"""

import torch
import logging
from PIL import Image
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoProcessor

from models.base_wrapper import BaseLVLMWrapper, LVLMOutput

logger = logging.getLogger(__name__)


class Phi3VisionWrapper(BaseLVLMWrapper):
    """Wrapper for microsoft/Phi-3.5-vision-instruct."""

    # Phi-3.5 image placeholder token
    IMAGE_TOKEN = "<|image_1|>"

    def get_model_key(self) -> str:
        return "phi3"

    def get_model_name(self) -> str:
        return "Phi-3.5-Vision-Instruct"

    def _load_model_and_processor(self):
        model_id = self.model_config["id"]
        dtype_str = self.model_config.get("dtype", "float32")
        dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

        logger.info(f"Loading processor from {model_id} ...")
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=4,   # Number of image crops for Phi-3.5-Vision
        )
        attn_impl = self.model_config.get("attn_implementation", "eager")
        
        logger.info(f"Loading model from {model_id} (dtype={dtype}, device={self.device}, attn={attn_impl}) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            _attn_implementation=attn_impl
        )

        if self.device != "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Phi-3.5-Vision loaded.")

    def _forward_pass(self, image: Image.Image, prompt: str) -> LVLMOutput:
        """
        Run inference on (image, prompt) and extract logits + embeddings.

        Strategy for Phi-3.5:
          1. Format prompt with image placeholder
          2. Get inputs via processor
          3. Generate tokens (greedy) collecting hidden_states at each step
          4. Re-run forward pass on full sequence to get all hidden states
          5. Separate visual and text token positions from hidden states
        """
        max_new_tokens = self.model_config.get("max_new_tokens", 128)
        hidden_layer_index = self.config["detection"].get("hidden_layer_index", -1)

        # ── Step 1: Format input ──────────────────────────────────────────────
        messages = [
            {"role": "user", "content": f"{self.IMAGE_TOKEN}\n{prompt}"}
        ]
        formatted = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ── Step 2: Tokenize & process image ─────────────────────────────────
        inputs = self.processor(
            text=formatted,
            images=[image],
            return_tensors="pt",
        )

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ── Step 3: Greedy generation ─────────────────────────────────────────
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        new_token_ids = generated_ids[0, input_len:].tolist()
        answer = self.processor.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        tokens = [
            self.processor.tokenizer.decode([t], skip_special_tokens=True)
            for t in new_token_ids
        ]

        # ── Step 4: Full forward pass on generated sequence ──────────────────
        # We need hidden states for both visual prefix and generated tokens
        with torch.no_grad():
            full_outputs = self.model(
                input_ids=generated_ids,
                pixel_values=inputs.get("pixel_values"),
                image_sizes=inputs.get("image_sizes"),
                output_hidden_states=True,
                return_dict=True,
            )

        # hidden_states: tuple of (num_layers+1) tensors, each (1, full_seq_len, hidden_dim)
        hidden_states = full_outputs.hidden_states  # tuple
        layer_hidden = hidden_states[hidden_layer_index]  # (1, seq_len, hidden_dim)
        layer_hidden = layer_hidden.squeeze(0)             # (seq_len, hidden_dim)

        # ── Step 5: Logits for generated tokens ──────────────────────────────
        # full_outputs.logits: (1, full_seq_len, vocab_size)
        logits = full_outputs.logits.squeeze(0)   # (full_seq_len, vocab_size)
        gen_logits = logits[input_len:]            # (new_tokens, vocab_size)

        # ── Step 6: Separate visual vs text embeddings ────────────────────────
        visual_embeddings, text_embeddings = self._split_visual_text_embeddings(
            inputs["input_ids"].squeeze(0),
            layer_hidden,
            input_len,
        )

        return LVLMOutput(
            answer=answer,
            logits=gen_logits.cpu(),
            text_embeddings=text_embeddings.cpu(),
            visual_embeddings=visual_embeddings.cpu(),
            token_ids=new_token_ids,
            tokens=tokens,
            full_seq_len=generated_ids.shape[1],
        )

    def _split_visual_text_embeddings(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
        input_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split hidden state sequence into visual patch embeddings and text embeddings.

        Phi-3.5-Vision embeds image patches as a contiguous block of special tokens
        (with IDs typically in a specific range set at model load time). We identify
        these by looking for non-standard token IDs beyond the text vocabulary.

        Returns:
            visual_embeddings: (num_visual_patches, hidden_dim)
            text_embeddings:   (num_generated_tokens, hidden_dim)
        """
        vocab_size = self.model.config.vocab_size

        # Visual tokens have IDs >= text vocab_size (image patch tokens)
        prefix_ids = input_ids[:input_len]
        is_visual = (prefix_ids >= vocab_size) | (prefix_ids < 0)

        # Fallback: if no visual tokens detected via ID, use middle segment heuristic
        if is_visual.sum() == 0:
            # Image tokens are typically the longest contiguous non-text region
            # Estimate: first 256 tokens after BOS as visual region
            visual_len = min(256, input_len // 2)
            visual_embeddings = hidden[1:visual_len + 1]   # skip BOS
        else:
            visual_embeddings = hidden[:input_len][is_visual]

        # Text embeddings = generated token positions (after input prefix)
        text_embeddings = hidden[input_len:]  # (num_generated, hidden_dim)

        return visual_embeddings, text_embeddings
