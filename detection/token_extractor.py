"""
detection/token_extractor.py
Module 2: Content Token Extractor

Purpose:
  Filter the LVLM-generated tokens to keep only content-bearing tokens
  (nouns, proper nouns, adjectives, verbs) that can be visually grounded.

Why this matters:
  Function words ("is", "the", "and") have no visual referent.
  Only content words should be checked for visual alignment.

Methods:
  - POS tagging via spaCy (default, CPU-friendly)
  - Attention-based filtering (optional, future work)
"""

import logging
from typing import List, Tuple, Optional
import torch

logger = logging.getLogger(__name__)

# Target POS tags for visually grounded content
CONTENT_POS_TAGS = {"NOUN", "PROPN", "ADJ", "VERB"}

# Tokens to always skip regardless of POS
SKIP_TOKENS = {"yes", "no", "there", "is", "are", "the", "a", "an", "it", "they"}


class TokenExtractor:
    """
    Extracts content-bearing token indices from LVLM generated output.

    Usage:
        extractor = TokenExtractor()
        indices = extractor.extract(answer_text, tokens)
    """

    def __init__(self, use_pos_filter: bool = True):
        self.use_pos_filter = use_pos_filter
        self._nlp = None

        if use_pos_filter:
            self._load_spacy()

    def _load_spacy(self):
        """Lazy-load spaCy model."""
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy 'en_core_web_sm' loaded.")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm\n"
                "Falling back to all-tokens mode."
            )
            self.use_pos_filter = False

    def extract(
        self,
        answer: str,
        tokens: List[str],
        embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], List[str]]:
        """
        Identify indices of content-bearing tokens in the generated sequence.

        Args:
            answer:     Full generated answer string
            tokens:     List of decoded token strings (e.g., ['▁There', '▁is', '▁a', '▁chair'])
            embeddings: (optional) Token embedding tensor — unused here, for future extensions

        Returns:
            (indices, content_tokens):
                indices:        List of integer indices into `tokens`
                content_tokens: Corresponding decoded strings
        """
        if not tokens:
            return [], []

        if not self.use_pos_filter or self._nlp is None:
            # Fallback: return all token indices
            return list(range(len(tokens))), tokens

        # ── Run spaCy POS tagging on the answer text ──────────────────────────
        doc = self._nlp(answer)

        # Build a set of content words from spaCy parse
        content_words = set()
        for token in doc:
            if token.pos_ in CONTENT_POS_TAGS and token.text.lower() not in SKIP_TOKENS:
                content_words.add(token.text.lower())
                content_words.add(token.lemma_.lower())

        logger.debug(f"Content words from spaCy: {content_words}")

        # ── Map content words back to subword tokens ───────────────────────────
        # Handle BPE/sentencepiece prefixes (▁ or Ġ)
        indices = []
        content_tokens = []

        for i, tok in enumerate(tokens):
            clean = tok.strip("▁Ġ ").lower()
            if clean in content_words and clean not in SKIP_TOKENS:
                indices.append(i)
                content_tokens.append(tok)

        # Fallback if nothing matched (e.g., very short answers like "No.")
        if not indices:
            logger.debug("No content tokens found — falling back to all tokens.")
            return list(range(len(tokens))), tokens

        logger.debug(f"Content token indices: {indices} → {content_tokens}")
        return indices, content_tokens

    def extract_batch(
        self,
        answers: List[str],
        tokens_batch: List[List[str]],
    ) -> List[Tuple[List[int], List[str]]]:
        """Process a batch of answers."""
        return [
            self.extract(answer, tokens)
            for answer, tokens in zip(answers, tokens_batch)
        ]
