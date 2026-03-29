"""
models/base_wrapper.py
Abstract base class for LVLM wrappers.

All model wrappers must inherit from BaseLVLMWrapper and implement:
  - load(): Load model + processor from HuggingFace
  - run(image, prompt): Forward pass returning internals
  - get_model_name(): Return model identifier string
"""

from abc import ABC, abstractmethod
from typing import Optional
from PIL import Image
import torch
import logging

logger = logging.getLogger(__name__)


class LVLMOutput:
    """
    Structured output from a single LVLM forward pass.

    Attributes:
        answer          (str): Generated text answer
        logits          (Tensor): (seq_len, vocab_size) — logits over generated tokens
        text_embeddings (Tensor): (seq_len, hidden_dim) — last hidden layer of text tokens
        visual_embeddings (Tensor): (num_visual_patches, hidden_dim) — visual patch states
        token_ids       (list[int]): Generated token IDs
        tokens          (list[str]): Decoded token strings
        full_seq_len    (int): Total sequence length (prefix + generated)
    """

    def __init__(
        self,
        answer: str,
        logits: torch.Tensor,
        text_embeddings: torch.Tensor,
        visual_embeddings: torch.Tensor,
        token_ids: list,
        tokens: list,
        full_seq_len: int,
    ):
        self.answer = answer
        self.logits = logits
        self.text_embeddings = text_embeddings
        self.visual_embeddings = visual_embeddings
        self.token_ids = token_ids
        self.tokens = tokens
        self.full_seq_len = full_seq_len

    def __repr__(self):
        return (
            f"LVLMOutput(\n"
            f"  answer='{self.answer[:60]}...'\n"
            f"  logits.shape={self.logits.shape}\n"
            f"  text_embeddings.shape={self.text_embeddings.shape}\n"
            f"  visual_embeddings.shape={self.visual_embeddings.shape}\n"
            f"  tokens={self.tokens[:8]}...\n"
            f")"
        )


class BaseLVLMWrapper(ABC):
    """
    Abstract base class for LVLM inference wrappers.

    Subclasses implement _load_model_and_processor() and _forward_pass()
    to provide model-specific loading and inference logic.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_config = config["models"][self.get_model_key()]
        self.device = self._resolve_device()
        self.model = None
        self.processor = None
        self._loaded = False

    def _resolve_device(self) -> str:
        device_cfg = self.model_config.get("device", "auto")
        if device_cfg == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_cfg

    def load(self):
        """Load model and processor. Called once before inference."""
        if self._loaded:
            return
        logger.info(f"Loading {self.get_model_name()} on device={self.device} ...")
        self._load_model_and_processor()
        self._loaded = True
        logger.info(f"Model loaded successfully.")

    def run(self, image: Image.Image, prompt: str) -> LVLMOutput:
        """
        Main inference entry point.

        Args:
            image: PIL.Image
            prompt: Question / instruction string

        Returns:
            LVLMOutput with answer, embeddings, logits, tokens
        """
        if not self._loaded:
            self.load()
        return self._forward_pass(image, prompt)

    @abstractmethod
    def _load_model_and_processor(self):
        """Load self.model and self.processor."""
        pass

    @abstractmethod
    def _forward_pass(self, image: Image.Image, prompt: str) -> LVLMOutput:
        """Run inference and return LVLMOutput."""
        pass

    @abstractmethod
    def get_model_key(self) -> str:
        """Return config key (e.g., 'phi3', 'llava')."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return human-readable model name."""
        pass

    def get_device(self) -> str:
        return self.device

    def unload(self):
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False
        logger.info(f"Model {self.get_model_name()} unloaded.")
