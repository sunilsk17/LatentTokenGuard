"""
detection/logit_contrast.py
Module 4: Logit Contrast Engine (Optional — doubles inference time)

Intuition:
  If a model says "there is a microwave" but the image shows no microwave,
  the token probability of "microwave" should NOT meaningfully increase
  when the image is provided vs. when it's absent.

  Low P(token | image) - P(token | no image) → token is not visually grounded
  → hallucination signal

Method:
  1. Run LVLM with original image  → P1(token | image, prompt)
  2. Run LVLM with blank/grey image → P2(token | prompt only)
  3. contrast_i = P1_i - P2_i  for each content token
  4. contrast_score = mean(max(0, contrast_i))
  5. hallucination_score = 1 - contrast_score

Reference: VCD (CVPR 2024) uses a similar distorted-image contrast for
  decoding-time mitigation. Our method uses it for post-hoc detection.
"""

import torch
import torch.nn.functional as F
import logging
from PIL import Image
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _make_blank_image(size: Tuple[int, int], color: int = 128) -> Image.Image:
    """Create a grey RGB image of the same size as the original."""
    return Image.new("RGB", size, color=(color, color, color))


class LogitContrastEngine:
    """
    Computes token-level logit contrast between image-conditioned and
    image-free model runs.
    """

    def __init__(self, config: dict):
        self.alpha = config["detection"].get("alpha", 0.7)
        self.beta = config["detection"].get("beta", 0.3)
        self.blank_color = 128   # Gray blank image

    def compute_contrast(
        self,
        model_wrapper,
        image: Image.Image,
        prompt: str,
        token_ids: List[int],
        token_indices: List[int],
        logits_with_image: torch.Tensor,   # (seq_len, vocab_size) — already computed
    ) -> Tuple[float, List[float]]:
        """
        Compute logit contrast score for generated content tokens.

        Args:
            model_wrapper:      Any BaseLVLMWrapper instance
            image:              Original PIL Image
            prompt:             Original prompt
            token_ids:          All generated token IDs
            token_indices:      Indices of content tokens to analyse
            logits_with_image:  Pre-computed logits from the full forward pass

        Returns:
            contrast_score  (float): Mean positive contrast ∈ [0, 1]
            per_token_contrast (list): Per-token contrast values
        """
        if not token_indices or not token_ids:
            logger.warning("Empty token_indices — cannot compute logit contrast.")
            return 0.5, []

        # ── Run model with blank image ─────────────────────────────────────────
        blank_image = _make_blank_image(image.size, self.blank_color)
        logger.debug("Running model on blank image for logit contrast...")

        try:
            blank_output = model_wrapper.run(blank_image, prompt)
            logits_without_image = blank_output.logits   # (seq_len_blank, vocab_size)
        except Exception as e:
            logger.warning(f"Failed blank-image run: {e}. Returning neutral contrast.")
            return 0.5, []

        # ── Compute per-content-token contrast ────────────────────────────────
        min_len = min(logits_with_image.shape[0], logits_without_image.shape[0], len(token_ids))
        contrasts = []

        for idx in token_indices:
            if idx >= min_len:
                continue

            token_id = token_ids[idx]

            # Probabilities at this token position
            p_with = F.softmax(logits_with_image[idx], dim=-1)[token_id].item()
            p_without = F.softmax(logits_without_image[idx], dim=-1)[token_id].item()

            # Positive contrast: image increases token probability
            contrast = p_with - p_without
            contrasts.append(contrast)

            logger.debug(
                f"  token_idx={idx} | p_with={p_with:.4f} | "
                f"p_without={p_without:.4f} | contrast={contrast:.4f}"
            )

        if not contrasts:
            return 0.5, []

        # ── Aggregate: mean of positive contrasts ─────────────────────────────
        # Only count positive contrasts (image-grounded tokens)
        positive_contrasts = [max(0.0, c) for c in contrasts]
        max_p = max(positive_contrasts) if positive_contrasts else 0.0

        # Normalize by max possible contrast (1.0 - 0.0 = 1.0)
        contrast_score = sum(positive_contrasts) / len(positive_contrasts)  # ∈ [0, 1]

        logger.debug(f"Logit contrast score: {contrast_score:.4f}")
        return contrast_score, contrasts

    def combine_scores(
        self,
        alignment_score: float,
        contrast_score: Optional[float] = None,
    ) -> float:
        """
        Combine alignment and contrast into final hallucination score.

        H = α·(1 - alignment) + β·(1 - contrast)
          = 1 - [α·alignment + β·contrast]

        If contrast_score is None, uses alignment only (α=1.0).
        """
        if contrast_score is None:
            return 1.0 - alignment_score

        h_align = 1.0 - alignment_score
        h_contrast = 1.0 - contrast_score
        combined = self.alpha * h_align + self.beta * h_contrast
        return float(combined)
