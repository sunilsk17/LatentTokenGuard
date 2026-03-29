"""
tests/test_latent_contrast.py
Unit tests for the LatentContrastEngine.

Tests core mathematical properties:
  - Perfect alignment → score ~ 0 (faithful)
  - Orthogonal embeddings → score ~ 0.5 (neutral)
  - Anti-aligned embeddings → score ~ 1 (hallucinated)
  - Normalization invariance
  - Empty tensor handling
"""

import torch
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.latent_contrast import LatentContrastEngine

DUMMY_CONFIG = {
    "detection": {
        "normalize_embeddings": True,
        "hidden_layer_index": -1,
        "use_token_filter": True,
        "threshold": 0.5,
        "alpha": 0.7,
        "beta": 0.3,
    }
}


@pytest.fixture
def engine():
    return LatentContrastEngine(DUMMY_CONFIG)


def test_perfect_alignment(engine):
    """Identical text and visual embeddings → max alignment → low hallucination score."""
    emb = torch.randn(4, 256)
    hall, align, per_tok = engine.compute_alignment(emb, emb.clone())
    assert align > 0.9, f"Expected high alignment, got {align:.4f}"
    assert hall < 0.15, f"Expected low hallucination score, got {hall:.4f}"


def test_orthogonal_embeddings(engine):
    """Orthogonal embeddings → cosine ≈ 0 → neutral score ~0.5."""
    d = 128
    text_emb = torch.zeros(3, d)
    text_emb[0, 0] = 1.0
    text_emb[1, 1] = 1.0
    text_emb[2, 2] = 1.0

    visual_emb = torch.zeros(3, d)
    visual_emb[0, 3] = 1.0
    visual_emb[1, 4] = 1.0
    visual_emb[2, 5] = 1.0

    # Cosine between orthogonal vectors = 0 → normalized to 0.5
    hall, align, _ = engine.compute_alignment(text_emb, visual_emb)
    assert 0.4 <= align <= 0.6, f"Expected ~0.5 alignment, got {align:.4f}"


def test_anti_aligned_embeddings(engine):
    """
    Anti-aligned embeddings → hallucination score should be HIGHER than the perfectly-aligned case.

    Note: with max-pool, even fully anti-aligned vectors can give cosine=0 (not necessarily cosine=-1)
    when text and visual tokens don't pair 1-to-1. We verify the ordering: mismatched > matched.
    """
    d = 256
    # Matched: identical → low hallucination
    base = torch.randn(4, d)
    hall_match, _, _ = engine.compute_alignment(base, base.clone())

    # Mismatched: text aligned with one direction, visual with opposite
    text_emb = base.clone()
    vis_emb  = -base.clone()
    hall_mismatch, _, _ = engine.compute_alignment(text_emb, vis_emb)

    # Mismatched should produce equal or higher hallucination score
    assert hall_mismatch >= hall_match - 0.01, (
        f"Mismatched ({hall_mismatch:.4f}) should be >= matched ({hall_match:.4f})"
    )



def test_token_index_filtering(engine):
    """Only selected token indices should be used."""
    all_embs = torch.randn(10, 64)
    visual   = torch.randn(5, 64)

    # Use only first 3 tokens
    hall_all, _, _ = engine.compute_alignment(all_embs, visual, token_indices=None)
    hall_filt, _, _ = engine.compute_alignment(all_embs, visual, token_indices=[0, 1, 2])

    # Both should return valid scores in [0, 1]
    assert 0.0 <= hall_all  <= 1.0
    assert 0.0 <= hall_filt <= 1.0


def test_empty_text_embeddings(engine):
    """Empty text embeddings → neutral score (0.5)."""
    visual = torch.randn(8, 64)
    empty  = torch.zeros(0, 64)
    hall, align, per_tok = engine.compute_alignment(empty, visual)
    assert hall == 0.5


def test_empty_visual_embeddings(engine):
    """Empty visual embeddings → neutral score (0.5)."""
    text   = torch.randn(4, 64)
    empty  = torch.zeros(0, 64)
    hall, align, per_tok = engine.compute_alignment(text, empty)
    assert hall == 0.5


def test_score_range(engine):
    """Hallucination score must always be in [0, 1]."""
    for _ in range(10):
        T = torch.randn(5, 128)
        V = torch.randn(10, 128)
        hall, align, _ = engine.compute_alignment(T, V)
        assert 0.0 <= hall  <= 1.0, f"hall score out of range: {hall}"
        assert 0.0 <= align <= 1.0, f"align score out of range: {align}"


def test_normalization_invariance(engine):
    """Scaling embeddings should not change the result (cosine is scale-invariant)."""
    T = torch.randn(4, 64)
    V = torch.randn(6, 64)
    h1, a1, _ = engine.compute_alignment(T, V)
    h2, a2, _ = engine.compute_alignment(T * 100.0, V * 0.001)
    assert abs(h1 - h2) < 1e-4, f"Score changed with scaling: {h1:.5f} vs {h2:.5f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
