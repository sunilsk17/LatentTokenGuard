"""
tests/test_pipeline.py
End-to-end pipeline test using synthetic data (no real model needed).

Tests the full detection pipeline with mocked LVLM output:
  Token Extractor → Latent Contrast Engine → Decision Layer
"""

import torch
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_wrapper import LVLMOutput
from detection.token_extractor import TokenExtractor
from detection.latent_contrast import LatentContrastEngine
from detection.decision import DecisionLayer
from evaluation.metrics import Evaluator, evaluate

CONFIG = {
    "detection": {
        "normalize_embeddings": True,
        "hidden_layer_index": -1,
        "use_token_filter": True,
        "threshold": 0.5,
        "alpha": 0.7,
        "beta": 0.3,
        "use_logit_contrast": False,
    }
}


def make_mock_output(answer: str, aligned: bool = True, hidden_dim: int = 64) -> LVLMOutput:
    """Create synthetic LVLMOutput with controllable alignment."""
    num_tokens = max(len(answer.split()), 4)
    num_patches = 16

    if aligned:
        # Text and visual embeddings point in the same direction → faithful
        base = torch.randn(hidden_dim)
        text_embs = base.unsqueeze(0).repeat(num_tokens, 1) + torch.randn(num_tokens, hidden_dim) * 0.1
        vis_embs  = base.unsqueeze(0).repeat(num_patches, 1) + torch.randn(num_patches, hidden_dim) * 0.1
    else:
        # Anti-aligned → hallucinated
        base = torch.randn(hidden_dim)
        text_embs = base.unsqueeze(0).repeat(num_tokens, 1)
        vis_embs  = -base.unsqueeze(0).repeat(num_patches, 1)

    tokens = answer.split()
    return LVLMOutput(
        answer=answer,
        logits=torch.randn(num_tokens, 1000),
        text_embeddings=text_embs,
        visual_embeddings=vis_embs,
        token_ids=list(range(num_tokens)),
        tokens=tokens,
        full_seq_len=num_tokens + 10,
    )


def pipeline(output: LVLMOutput, config: dict = CONFIG):
    extractor = TokenExtractor(use_pos_filter=False)
    engine = LatentContrastEngine(config)
    decision = DecisionLayer(threshold=config["detection"]["threshold"])

    indices, _ = extractor.extract(output.answer, output.tokens)
    hall, align, per_tok = engine.compute_alignment(
        output.text_embeddings, output.visual_embeddings, indices
    )
    label, score = decision.classify(hall)
    return label, score, align


def test_faithful_output_classified_correctly():
    output = make_mock_output("There is a chair and a sofa", aligned=True)
    label, score, align = pipeline(output)
    assert label == "Faithful", f"Expected Faithful, got {label} (score={score:.3f})"
    assert score < 0.5


def test_hallucinated_output_classified_correctly():
    output = make_mock_output("There is a microwave and a refrigerator", aligned=False)
    label, score, align = pipeline(output)
    assert label == "Hallucinated", f"Expected Hallucinated, got {label} (score={score:.3f})"
    assert score > 0.5


def test_evaluator_accumulates_correctly():
    ev = Evaluator()
    ev.add("Hallucinated", True, 0.8, 100.0)   # TP
    ev.add("Faithful",     False, 0.2, 90.0)   # TN
    ev.add("Hallucinated", False, 0.7, 95.0)   # FP
    ev.add("Faithful",     True, 0.3, 85.0)    # FN

    result = ev.compute(threshold=0.5, dataset="test", model="mock", mode="alignment_only")
    assert result.tp == 1
    assert result.tn == 1
    assert result.fp == 1
    assert result.fn == 1
    assert result.n_samples == 4
    assert abs(result.accuracy - 0.5) < 1e-6


def test_decision_threshold_tuning():
    from detection.decision import DecisionLayer
    scores = [0.2, 0.4, 0.6, 0.8, 0.3, 0.7]
    labels = [0,   0,   1,   1,   0,   1  ]  # 0=faithful, 1=hallucinated
    d = DecisionLayer(threshold=0.5)
    best_t, best_f1 = d.tune_threshold(scores, labels)
    assert 0.0 < best_t < 1.0
    assert 0.0 <= best_f1 <= 1.0


def test_evaluate_convenience():
    result = evaluate(
        predictions=["Hallucinated", "Faithful", "Hallucinated"],
        ground_truth=[True, False, False],
        scores=[0.8, 0.2, 0.7],
        dataset="mock",
        model="mock",
        mode="test",
    )
    assert result.n_samples == 3
    assert 0.0 <= result.f1 <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
