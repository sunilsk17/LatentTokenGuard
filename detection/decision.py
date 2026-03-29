"""
detection/decision.py
Module 5: Decision Layer

Converts a hallucination score ∈ [0, 1] into a binary label.

Also provides:
  - Threshold tuner: find optimal threshold on a validation set (F1-maximizing)
  - Soft label: probability-style confidence output
"""

import logging
import numpy as np
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

LABEL_HALLUCINATED = "Hallucinated"
LABEL_FAITHFUL = "Faithful"


class DecisionLayer:
    """
    Converts hallucination scores to binary labels.

    Usage:
        decision = DecisionLayer(threshold=0.5)
        label, confidence = decision.classify(0.73)
        # → ("Hallucinated", 0.73)
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def classify(self, hallucination_score: float) -> Tuple[str, float]:
        """
        Classify a single sample.

        Args:
            hallucination_score: ∈ [0, 1], higher → more hallucinated

        Returns:
            (label, confidence)
            label:      "Hallucinated" or "Faithful"
            confidence: The raw hallucination score (for AUROC)
        """
        label = (
            LABEL_HALLUCINATED
            if hallucination_score > self.threshold
            else LABEL_FAITHFUL
        )
        return label, hallucination_score

    def classify_batch(
        self, scores: List[float]
    ) -> List[Tuple[str, float]]:
        """Classify a list of hallucination scores."""
        return [self.classify(s) for s in scores]

    def tune_threshold(
        self,
        scores: List[float],
        ground_truth: List[int],   # 1=Hallucinated, 0=Faithful
        metric: str = "f1",
    ) -> Tuple[float, float]:
        """
        Find the threshold that maximizes F1 (or accuracy) on a validation set.

        Args:
            scores:       List of hallucination scores
            ground_truth: Binary labels (1=hallucinated, 0=faithful)
            metric:       "f1" | "accuracy"

        Returns:
            (best_threshold, best_metric_value)
        """
        from sklearn.metrics import f1_score, accuracy_score

        best_threshold = 0.5
        best_score = 0.0
        candidates = np.arange(0.1, 0.95, 0.05)

        for t in candidates:
            preds = [1 if s > t else 0 for s in scores]
            if metric == "f1":
                score = f1_score(ground_truth, preds, zero_division=0)
            else:
                score = accuracy_score(ground_truth, preds)

            if score > best_score:
                best_score = score
                best_threshold = float(t)

        logger.info(
            f"Best threshold: {best_threshold:.2f} → {metric}={best_score:.4f}"
        )
        self.threshold = best_threshold
        return best_threshold, best_score

    def get_threshold(self) -> float:
        return self.threshold

    def set_threshold(self, threshold: float):
        self.threshold = threshold
        logger.info(f"Threshold updated to {threshold:.3f}")

    @staticmethod
    def pope_label_to_binary(label: str) -> int:
        """
        Convert POPE dataset label to binary.
        POPE: 'yes' = object present = Faithful (0)
              'no'  = object absent  = could be Hallucinated (1) if model said yes
        """
        return 0 if label.strip().lower() == "yes" else 1
