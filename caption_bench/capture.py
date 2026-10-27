"""
caption_bench/capture.py
Single-pass caption generation with hidden-state capture.

One call to `generate()` returns all inputs the detectors use:
  - the caption and its token ids
  - hidden states of every layer for the image tokens (from the prefill step)
  - hidden states of every layer for each generated token
  - the logits that produced each generated token

There is no second forward pass, so detection cost is the cost of the
similarity maths on tensors that generation already computed.

`teacher_force()` re-reads an existing caption in one forward pass; it is
used to extract hidden states for the trained-probe baseline after a run.

Supported model families (the `family` key of a model config):
  - "llava" (default): any `LlavaForConditionalGeneration` checkpoint
    (llava-hf/llava-1.5-7b-hf, -13b-hf, the tiny random test model), with
    both transformers conventions for image tokens: expanded inside the
    model (<= 4.46) or by the processor (newer)
  - "qwen2_5_vl": Qwen2.5-VL (transformers >= 4.49). The processor expands
    the image placeholder into one token per merged 28x28 patch, so the
    number of image tokens depends on the image size.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Capture:
    """Tensors and text recorded during one generation."""
    caption: str
    token_ids: List[int]          # generated ids (special tokens included)
    token_starts: List[int]       # char offset in `caption` where each token begins
    image_states: torch.Tensor    # (L+1, P, d) hidden states at image positions
    seq_states: torch.Tensor      # (T, L+1, d): [0] = last prompt position,
                                  #              [k] = generated token k-1
    logits: torch.Tensor          # (T, V) logits that produced generated token t
    gen_seconds: float


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LlavaCapturer:
    """
    Loads a LLaVA-family checkpoint and runs captured generation.

    Usage:
        cap = LlavaCapturer(model_cfg)
        out = cap.generate(image, "Please describe this image in detail.")
    """

    def __init__(self, model_cfg: dict):
        self.cfg = model_cfg
        self.device = pick_device(model_cfg.get("device", "auto"))
        self.model = None
        self.processor = None

    # ──────────────────────────────────────────────────────────────────────────
    # Loading
    # ──────────────────────────────────────────────────────────────────────────

    def _model_class(self):
        from transformers import LlavaForConditionalGeneration
        return LlavaForConditionalGeneration

    def load(self):
        from transformers import AutoProcessor

        model_id = self.cfg["id"]
        load_in_4bit = self.cfg.get("load_in_4bit", False) and self.device == "cuda"
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[self.cfg.get("dtype", "float16")]
        if self.device == "cpu":
            dtype = torch.float32

        logger.info(f"Loading {model_id} on {self.device} (dtype={dtype}, 4bit={load_in_4bit})")
        # The slow (sentencepiece) tokenizer also loads on older tokenizers releases
        self.processor = AutoProcessor.from_pretrained(
            model_id, use_fast=self.cfg.get("use_fast_tokenizer", True))

        kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}

        self.model = self._model_class().from_pretrained(model_id, **kwargs)
        if not load_in_4bit:
            self.model.to(self.device)
        self.model.eval()
        self.load_in_4bit = load_in_4bit
        self.dtype = dtype

    def describe(self) -> dict:
        """Settings recorded with every run, so results are traceable."""
        import transformers
        return {
            "model_id": self.cfg["id"],
            "family": self.cfg.get("family", "llava"),
            "device": self.device,
            "dtype": str(self.dtype),
            "load_in_4bit": self.load_in_4bit,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "num_layers": self.num_layers(),
        }

    def num_layers(self) -> int:
        return self.model.config.text_config.num_hidden_layers

    # ──────────────────────────────────────────────────────────────────────────
    # Logit lens helpers (used by the Internal Confidence baseline)
    # ──────────────────────────────────────────────────────────────────────────

    def final_norm(self) -> torch.nn.Module:
        """The language model's last RMSNorm; its location moved across versions."""
        for path in ("language_model.model.norm", "model.language_model.norm",
                     "language_model.norm"):
            module = self.model
            try:
                for attr in path.split("."):
                    module = getattr(module, attr)
                return module
            except AttributeError:
                continue
        raise AttributeError("Could not locate the language model's final norm")

    def lm_head(self) -> torch.nn.Module:
        return self.model.get_output_embeddings()

    # ──────────────────────────────────────────────────────────────────────────
    # Generation
    # ──────────────────────────────────────────────────────────────────────────

    def generate(self, image: Image.Image, prompt: str) -> Capture:
        import time

        text = self.cfg["prompt_template"].format(prompt=prompt)
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.get("max_new_tokens", 512),
                do_sample=False,
                num_beams=1,
                output_hidden_states=True,
                output_logits=True,
                return_dict_in_generate=True,
                **self.cfg.get("generate_kwargs", {}),
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
        gen_seconds = time.time() - t0

        input_ids = inputs["input_ids"][0]
        prompt_len = input_ids.shape[0]
        gen_ids = out.sequences[0, prompt_len:].tolist()

        # hidden_states: one tuple per step, each with L+1 tensors (1, seq, d)
        prefill = out.hidden_states[0]
        image_pos = self._image_positions(input_ids, prefill[0].shape[1])
        image_states = torch.stack([h[0, image_pos] for h in prefill])          # (L+1, P, d)

        steps = [torch.stack([h[0, -1] for h in step]) for step in out.hidden_states]
        seq_states = torch.stack(steps)                                          # (T, L+1, d)
        logits = torch.stack([l[0] for l in out.logits])                         # (T, V)

        caption, starts = self._decode_with_offsets(gen_ids)
        return Capture(caption, gen_ids, starts, image_states, seq_states, logits, gen_seconds)

    def teacher_force(self, image: Image.Image, prompt: str, caption: Optional[str] = None,
                      token_ids: Optional[List[int]] = None, with_prompt_state: bool = False):
        """
        One forward pass over prompt + an existing caption (no generation).
        The caption is given as text (re-tokenized) or as the generated token ids.
        Returns (caption_states (L+1, T, d), image_states (L+1, P, d), token_ids, token_starts),
        where caption_states[:, k] is the state at caption token k, i.e. Capture.seq_states[k + 1].
        With with_prompt_state, the last prompt token's states (L+1, d) are appended.
        """
        text = self.cfg["prompt_template"].format(prompt=prompt)
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        if token_ids is None:
            token_ids = self.processor.tokenizer(caption, add_special_tokens=False)["input_ids"]
        n_cap = len(token_ids)
        prompt_ids = inputs["input_ids"]
        full_ids = torch.cat([prompt_ids, torch.tensor([token_ids], dtype=prompt_ids.dtype)], 1)
        extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
        # Per-token inputs (Qwen's mm_token_type_ids) cover the prompt; caption tokens are text (0)
        for k, v in extra.items():
            if v.dim() == 2 and v.shape == prompt_ids.shape:
                extra[k] = torch.cat([v, v.new_zeros(1, n_cap)], 1)
        extra = {k: v.to(self.model.device) for k, v in extra.items()}
        if "pixel_values" in extra:
            extra["pixel_values"] = extra["pixel_values"].to(self.dtype)
        full_ids = full_ids.to(self.model.device)

        with torch.no_grad():
            out = self.model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                             output_hidden_states=True, **extra)
        hidden = out.hidden_states                                               # L+1 x (1, S, d)
        # Prefill length, whether <image> was expanded by the processor or the model
        prefix = hidden[0].shape[1] - n_cap
        caption_states = torch.stack([h[0, prefix:] for h in hidden])            # (L+1, T, d)
        image_pos = self._image_positions(prompt_ids[0].to(self.model.device), prefix)
        image_states = torch.stack([h[0, image_pos] for h in hidden])            # (L+1, P, d)
        _, starts = self._decode_with_offsets(token_ids)
        if with_prompt_state:
            prompt_state = torch.stack([h[0, prefix - 1] for h in hidden])         # (L+1, d)
            return caption_states, image_states, token_ids, starts, prompt_state
        return caption_states, image_states, token_ids, starts

    def _image_positions(self, input_ids: torch.Tensor, prefill_len: int) -> torch.Tensor:
        """Positions of the image tokens in the prefill sequence."""
        config = self.model.config
        image_token = getattr(config, "image_token_index", None)
        if image_token is None:                       # renamed in newer transformers
            image_token = config.image_token_id
        hits = (input_ids == image_token).nonzero(as_tuple=True)[0]
        if len(hits) == 0:
            raise ValueError("No image token found in the prompt")
        if len(hits) > 1:
            # Processor already expanded <image> into one token per patch
            return hits
        # Model expands the single <image> token internally
        n_patches = prefill_len - input_ids.shape[0] + 1
        start = hits[0].item()
        return torch.arange(start, start + n_patches, device=input_ids.device)

    def _decode_with_offsets(self, ids: List[int]):
        """Decode, recording where each token's text begins in the final string."""
        tok = self.processor.tokenizer
        kw = {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
        caption = tok.decode(ids, **kw)
        starts = [len(tok.decode(ids[:i], **kw)) for i in range(len(ids))]
        return caption, starts


class QwenVLCapturer(LlavaCapturer):
    """
    Qwen2.5-VL. Generation, capture and the logit lens are shared with LLaVA:
    the image tokens are found by `image_token_id` after processor expansion,
    and the final norm sits at model.language_model.norm.
    """

    def _model_class(self):
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration


CAPTURERS = {"llava": LlavaCapturer, "qwen2_5_vl": QwenVLCapturer}


def make_capturer(model_cfg: dict) -> LlavaCapturer:
    family = model_cfg.get("family", "llava")
    if family not in CAPTURERS:
        raise ValueError(f"Unknown model family '{family}', expected one of {sorted(CAPTURERS)}")
    return CAPTURERS[family](model_cfg)


def token_at_char(token_starts: List[int], char: int) -> int:
    """Index of the token whose text covers character `char`."""
    idx = 0
    for i, s in enumerate(token_starts):
        if s <= char:
            idx = i
        else:
            break
    return idx
