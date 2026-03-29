"""
models/llava_wrapper.py
LLaVA-1.5-7B LVLM wrapper for LatentTokenGuard.

STABILIZATION SUMMARY
─────────────────────
1.  Surgical Buffer Placement: 
    Forces CLIP VisionTower and Llama RotaryEmbedding (inv_freq) buffers to GPU.
2.  Standard Inference Flow:
    Uses standard processor -> model.generate(input_ids, pixel_values).
3.  Correct Indexing for Alignment (F1 Fix):
    Accounts for LLaVA's internal expansion of the single <image> token 
    into 576 visual patches in the hidden-state sequence.
"""

import torch
import logging
from PIL import Image
from typing import Tuple, List, Optional
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

from models.base_wrapper import BaseLVLMWrapper, LVLMOutput

logger = logging.getLogger(__name__)

# LLaVA-1.5-HF expansion constants
# 1 image token -> 576 patches. Shift is +575.
LLAVA_NUM_PATCHES = 576

class LLaVAWrapper(BaseLVLMWrapper):
    """Wrapper for llava-hf/llava-1.5-7b-hf."""

    def get_model_key(self) -> str:
        return "llava"

    def get_model_name(self) -> str:
        return "LLaVA-1.5-7B"

    # ──────────────────────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────────────────────

    def _load_model_and_processor(self):
        model_id = self.model_config["id"]
        logger.info(f"Loading LLaVA-1.5 (4-bit) from {model_id} ...")

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        import transformers.modeling_utils as _mu
        _orig_dispatch = _mu.dispatch_model
        def _safe_dispatch(model, device_map, **kwargs):
            try: return _orig_dispatch(model, device_map, **kwargs)
            except (ValueError, RuntimeError): return model
        
        _mu.dispatch_model = _safe_dispatch
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                quantization_config=quant_config,
                torch_dtype=torch.float16,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                trust_remote_code=self.model_config.get("trust_remote_code", False),
            )
        finally:
            _mu.dispatch_model = _orig_dispatch

        self.model.eval()
        device = "cuda:0"

        # ── THE SILVER BULLET: Force all Buffers to GPU ───────────────────────
        logger.info(f"Forcing vision_tower and projector to {device}...")
        self.model.vision_tower.to(device)
        self.model.multi_modal_projector.to(device)

        logger.info("Moving language_model.rotary_emb buffers to GPU...")
        for name, module in self.model.language_model.named_modules():
            if "rotary_emb" in name:
                module.to(device)

        # Final Verification
        try:
            vis_pos_dev = self.model.vision_tower.vision_model.embeddings.position_ids.device
            rot_dev = self.model.language_model.model.layers[0].self_attn.rotary_emb.inv_freq.device
            logger.info(f"Verified Devices: Vision-Pos={vis_pos_dev} | Language-Rotary={rot_dev}")
        except Exception:
            logger.warning("Device verification check failed.")

        logger.info(f"{self.get_model_name()} loaded successfully.")

    # ──────────────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────────────

    def _get_input_device(self) -> torch.device:
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def _forward_pass(self, image: Image.Image, prompt: str) -> LVLMOutput:
        max_new_tokens = self.model_config.get("max_new_tokens", 128)
        hidden_layer_index = self.config["detection"].get("hidden_layer_index", -1)
        device = self._get_input_device()

        # 1. Format & Process
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        # 2. Standard Generation
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # 3. Full contiguous forward pass for hidden states
        # IMPORTANT: LlavaForConditionalGeneration handles image expansion internally.
        # hidden_states.shape[1] = prompt_len - 1 + 576 + new_tokens
        with torch.no_grad():
            full_outputs = self.model(
                input_ids=generated_ids,
                pixel_values=inputs["pixel_values"],
                attention_mask=torch.ones_like(generated_ids),
                output_hidden_states=True,
                return_dict=True,
            )

        # 4. Extract generated tokens
        new_token_ids = generated_ids[0, prompt_len:].tolist()
        answer = self.processor.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        tokens = [self.processor.tokenizer.decode([t], skip_special_tokens=True) for t in new_token_ids]

        # 5. Indexing Logic for LLaVA-1.5 Latent space
        # Prefix length (expanded) = unexpanded_prompt_len + expansion_shift
        # Expansion shift = (576 patches - 1 image token) = 575
        prefix_len_expanded = prompt_len + (LLAVA_NUM_PATCHES - 1)
        
        hidden_states = full_outputs.hidden_states[hidden_layer_index].squeeze(0)  # [full_len, dim]
        logits        = full_outputs.logits.squeeze(0)                             # [full_len, vocab]

        # Alignment slices
        # Text embeddings are the tokens *after* the prefix (generated tokens)
        text_embeddings = hidden_states[prefix_len_expanded:]
        # Logits are for predicting tokens, so take position (prefix_len_expanded - 1) to (end - 1)
        gen_logits = logits[prefix_len_expanded - 1 : -1]

        # Visual patches are at the position of the <image> token in input_ids
        img_token_idx = self.model.config.image_token_index
        hits = (inputs["input_ids"][0] == img_token_idx).nonzero(as_tuple=True)[0]
        img_start = hits[0].item() if len(hits) > 0 else 0
        visual_embeddings = hidden_states[img_start : img_start + LLAVA_NUM_PATCHES]

        return LVLMOutput(
            answer=answer,
            logits=gen_logits.cpu(),
            text_embeddings=text_embeddings.cpu(),
            visual_embeddings=visual_embeddings.cpu(),
            token_ids=new_token_ids,
            tokens=tokens,
            full_seq_len=hidden_states.shape[0],
        )
