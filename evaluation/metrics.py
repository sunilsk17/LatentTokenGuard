"""
evaluation/metrics.py
Module 7: Evaluation Engine

Computes:
  - Accuracy
  - Precision, Recall, F1 Score (main metric)
  - AUROC (robustness metric)
  - Latency per sample
  - Confusion matrix

Also provides pretty-print and JSON serialization.
"""

import time
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics."""
    accuracy:   float = 0.0
    precision:  float = 0.0
    recall:     float = 0.0
    f1:         float = 0.0
    auroc:      float = 0.0
    tp:         int = 0
    fp:         int = 0
    tn:         int = 0
    fn:         int = 0
    n_samples:  int = 0
    mean_latency_ms: float = 0.0
    total_time_s: float = 0.0
    threshold:  float = 0.5
    dataset:    str = ""
    model:      str = ""
    mode:       str = ""
    extra:      Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  LatentTokenGuard Evaluation Results\n"
            f"{'='*55}\n"
            f"  Dataset:   {self.dataset}\n"
            f"  Model:     {self.model}\n"
            f"  Mode:      {self.mode}\n"
            f"  Threshold: {self.threshold:.3f}\n"
            f"  Samples:   {self.n_samples}\n"
            f"{'─'*55}\n"
            f"  Accuracy:  {self.accuracy:.4f}  ({self.accuracy*100:.2f}%)\n"
            f"  Precision: {self.precision:.4f}\n"
            f"  Recall:    {self.recall:.4f}\n"
            f"  F1 Score:  {self.f1:.4f}  ← main metric\n"
            f"  AUROC:     {self.auroc:.4f}\n"
            f"{'─'*55}\n"
            f"  TP={self.tp}  FP={self.fp}  TN={self.tn}  FN={self.fn}\n"
            f"{'─'*55}\n"
            f"  Latency:   {self.mean_latency_ms:.1f} ms/sample\n"
            f"  Total:     {self.total_time_s:.1f} s\n"
            f"{'='*55}\n"
        )


class Evaluator:
    """
    Collects predictions and computes all evaluation metrics.

    Usage:
        ev = Evaluator()
        for sample in dataset:
            t0 = time.time()
            pred_label, score = model.predict(sample)
            latency = (time.time() - t0) * 1000
            ev.add(pred_label, sample["is_hallucination"], score, latency)
        result = ev.compute(dataset="pope", model="phi3", mode="alignment_only")
        print(result)
    """

    def __init__(self):
        self._pred_labels: List[str] = []
        self._true_labels: List[int] = []   # 1 = Hallucinated, 0 = Faithful
        self._scores: List[float] = []      # Hallucination scores for AUROC
        self._latencies: List[float] = []   # ms per sample

    def add(
        self,
        pred_label: str,
        is_hallucination: bool,
        hallucination_score: float,
        latency_ms: float = 0.0,
    ):
        """Record one prediction."""
        self._pred_labels.append(pred_label)
        self._true_labels.append(int(is_hallucination))
        self._scores.append(float(hallucination_score))
        self._latencies.append(float(latency_ms))

    def compute(
        self,
        threshold: float = 0.5,
        dataset: str = "",
        model: str = "",
        mode: str = "",
    ) -> EvaluationResult:
        """Compute and return all metrics."""
        if len(self._pred_labels) == 0:
            logger.warning("No predictions recorded.")
            return EvaluationResult(dataset=dataset, model=model, mode=mode)

        y_pred = [1 if p == "Hallucinated" else 0 for p in self._pred_labels]
        y_true = self._true_labels
        scores = self._scores

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # AUROC: needs both classes present
        try:
            auroc = roc_auc_score(y_true, scores)
        except ValueError:
            auroc = 0.0
            logger.warning("AUROC could not be computed (single class in y_true).")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        mean_lat = float(np.mean(self._latencies)) if self._latencies else 0.0
        total_time = float(sum(self._latencies)) / 1000.0

        result = EvaluationResult(
            accuracy=float(acc),
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
            auroc=float(auroc),
            tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
            n_samples=len(y_true),
            mean_latency_ms=mean_lat,
            total_time_s=total_time,
            threshold=threshold,
            dataset=dataset,
            model=model,
            mode=mode,
        )
        return result

    def reset(self):
        """Reset collected predictions."""
        self.__init__()


def evaluate(
    predictions: List[str],
    ground_truth: List[bool],
    scores: List[float],
    latencies: Optional[List[float]] = None,
    **kwargs,
) -> EvaluationResult:
    """
    Convenience function for batch evaluation.

    Args:
        predictions:  List of "Hallucinated" | "Faithful" strings
        ground_truth: List of bool (True = hallucinated)
        scores:       List of hallucination scores ∈ [0, 1]
        latencies:    Optional list of per-sample latency in ms
    """
    ev = Evaluator()
    for i, (pred, gt, score) in enumerate(zip(predictions, ground_truth, scores)):
        lat = latencies[i] if latencies else 0.0
        ev.add(pred, gt, score, lat)
    return ev.compute(**kwargs)
